"""
greek_adaptive_rewriter.py

LLM-driven Greek passage adapter that targets lemma-coverage while preserving meaning.
This module orchestrates:
  - Lemmatizer (judge)
  - LemmaSalienceRanker (unknown ranking + coverage)
  - Iterative LLM rewrites with lemma/surface ban enforcement

You provide an LLM backend by implementing LLMClient.generate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from platform import system
from typing import Iterable, List, Optional, Sequence, Set, Tuple
import re
import unicodedata
from typing import Protocol, runtime_checkable, Dict
from lemma_salience import LemmaSalienceRanker, LemmaStats
import rewrite_prompts  # NEW: Import our prompting module


@runtime_checkable
class LemmatizerLike(Protocol):
    def lemmatize_passage(self, passage: str) -> list[tuple[str, str, str, str]]:
        """
        Must match your Lemmatizer.lemmatize_passage output:
        (original_surface, lemma, upos, source)
        """


# -----------------------------
# Utilities (mirror lemmatizer normalization logic)
# -----------------------------

# Free POS tags that don't count as unknown content words
FREE_UPOS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON"}

_TOKEN_EDGE_PUNCT_RE = re.compile(
    r"^[^\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+|[^\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+$"
)

def _strip_edge_punct(tok: str) -> str:
    return _TOKEN_EDGE_PUNCT_RE.sub("", tok.strip())

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip().lower()

def _is_free(upos: str) -> bool:
    return (upos or "").upper() in FREE_UPOS

def _unique_preserve_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# LLM Client abstraction
# -----------------------------

class LLMClient:
    """
    Implement generate() using whatever backend you want (OpenAI, local model, etc.).

    Expected behavior:
      - Input: system + user strings (or ignore system if your backend doesn't support it)
      - Output: a single Greek passage (no commentary)
    """
    def generate(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise NotImplementedError


# -----------------------------
# Core configuration
# -----------------------------

class RewriteMode(str, Enum):
    SURGICAL = "surgical"
    SIMPLIFY_THEN_ENFORCE = "simplify_then_enforce"
    RETELL = "retell"
    NOOB = "noob"  # NEW: aggressive simplification for beginners
    ULTRA_NOOB = "ultra_noob"  # NEW: extreme simplification

@dataclass
class IterationPolicy:
    max_rounds: int = 8
    stagnation_rounds: int = 2          # consecutive rounds without meaningful improvement
    min_delta_coverage: float = 0.003   # 0.3%
    ban_add_cap: int = 12               # max new banned lemmas per failed round
    ban_add_floor: int = 2              # min new banned lemmas per failed round
    ban_add_per_tokens: int = 40        # +1 ban per this many tokens
    auto_escalate: bool = True

@dataclass
class RewriteConfig:
    target_coverage: float = 0.95
    prefer_pos: Set[str] = field(default_factory=lambda: {"NOUN", "VERB", "ADJ", "ADV"})
    temperature: float = 0.2
    iteration: IterationPolicy = field(default_factory=IterationPolicy)
    # When we escalate to retell, how long should it be?
    retell_target_sentences: int = 6
    # For ultra_noob mode, how many sentences in the summary?
    ultra_noob_target_sentences: int = 3
    # Minimum vocabulary threshold
    min_vocab_threshold: float = 0.3  # Must know 30% of top N frequent lemmas
    top_n_frequent: int = 500  # Check against top 500 most frequent lemmas


# -----------------------------
# Result types
# -----------------------------

@dataclass
class RoundSnapshot:
    round_index: int
    mode: RewriteMode
    coverage: float
    total_tokens: int
    violations_lemmas: List[str]
    violations_surface: List[str]
    banned_lemmas_size: int
    banned_surface_size: int
    essential_lemmas: List[str]
    text: str

@dataclass
class RewriteResult:
    final_text: str
    final_coverage: float
    initial_coverage: float
    mode_used: RewriteMode
    rounds: List[RoundSnapshot]
    essential_lemmas: List[str]
    banned_lemmas: List[str]
    banned_surface_forms: List[str]


# -----------------------------
# Orchestrator
# -----------------------------

class GreekAdaptiveRewriter:
    def __init__(
        self,
        *,
        llm: LLMClient,
        ranker: LemmaSalienceRanker,
        config: Optional[RewriteConfig] = None,
        lemmatizer: Optional[LemmatizerLike] = None,  # NEW
        top_frequent_lemmas: Optional[List[str]] = None,  # Top N frequent lemmas for threshold check
    ):
        self.llm = llm
        self.ranker = ranker
        self.cfg = config or RewriteConfig()
        self.lemmatizer = lemmatizer
        self.top_frequent_lemmas = top_frequent_lemmas or []
        self._ban_tick = 0  # monotonically increasing "recency" counter

        self._ban_lemma_score: Dict[str, int] = {}
        self._ban_lemma_last: Dict[str, int] = {}

        self._ban_surface_score: Dict[str, int] = {}
        self._ban_surface_last: Dict[str, int] = {}
        
        # Track violation history for automatic keep-set expansion
        self._violation_history: Dict[str, int] = {}
        
        # Map surface forms to lemmas for better tracking
        self._surface_to_lemma: Dict[str, str] = {}

    # ---- public API ----


    def adapt(
        self,
        passage: str,
        *,
        known_lemmas: Set[str],
        target_coverage: Optional[float] = None,
        temperature: float = 0.2,
        mode: Optional[RewriteMode] = None,
    ) -> RewriteResult:
        """
        Adapt passage to target coverage using iterative lemma/surface enforcement.
        """
        target = float(target_coverage if target_coverage is not None else self.cfg.target_coverage)
        
        # Check minimum vocabulary threshold
        if self.top_frequent_lemmas and len(self.top_frequent_lemmas) > 0:
            known_frequent = len([lemma for lemma in self.top_frequent_lemmas if lemma in known_lemmas])
            frequent_coverage = known_frequent / len(self.top_frequent_lemmas)
            
            if frequent_coverage < self.cfg.min_vocab_threshold:
                # Return early with error message
                missing_essentials = [lemma for lemma in self.top_frequent_lemmas[:50] if lemma not in known_lemmas]  # Show top 50 missing
                error_msg = (
                    f"Insufficient vocabulary foundation. You know only {frequent_coverage:.1%} of the {len(self.top_frequent_lemmas)} "
                    f"most frequent Greek lemmas (minimum required: {self.cfg.min_vocab_threshold:.0%}). "
                    f"Focus on learning these essential words first: {', '.join(missing_essentials[:20])}..."
                )
                return RewriteResult(
                    final_text=error_msg,
                    final_coverage=0.0,
                    initial_coverage=0.0,
                    mode_used=RewriteMode.ULTRA_NOOB,
                    rounds=[],
                    essential_lemmas=[],
                    banned_lemmas=[],
                    banned_surface_forms=[],
                )

        # initial analysis
        init_analysis = self.ranker.analyze(passage, known_lemmas=known_lemmas)
        init_cov = float(init_analysis.coverage if init_analysis.coverage is not None else 1.0)

        chosen_mode = mode or self._choose_mode(init_cov)

        unknown_ranked = self.ranker.filter_known(init_analysis, known_lemmas)
        keep_stats = self._choose_essential_set(init_analysis, unknown_ranked, known_lemmas, target, chosen_mode)
        essential_lemmas = [s.lemma for s in keep_stats]

        banned_lemmas, banned_surface = self._initial_bans(init_analysis, known_lemmas, essential_lemmas)

        rounds: List[RoundSnapshot] = []
        best_text = passage
        best_cov = init_cov
        best_mode = chosen_mode  # NEW: best-so-far provenance
        stagnation = 0

        best_valid_text: Optional[str] = None
        best_valid_cov: float = -1.0
        best_valid_mode: RewriteMode = chosen_mode
        
        # Track for early stopping
        consecutive_drops = 0

        current_text = passage
        current_mode = chosen_mode
        last_viol_count = 10**9
        consecutive_no_viol_improve = 0

        for r in range(self.cfg.iteration.max_rounds):
            system, user = self._build_prompt(
                current_mode,
                current_text,
                target,
                essential_lemmas,
                banned_lemmas,
                banned_surface,
            )

            print(f"[adapt] round={r} mode={current_mode} essential={len(essential_lemmas)} banned_lemmas={len(banned_lemmas)} banned_surface={len(banned_surface)}")
            print(f"[adapt] prompt chars: system={len(system)} user={len(user)}")

            out = self.llm.generate(
                system=system,
                user=user,
                temperature=temperature,
            ).strip()

            

            if not out:
                # If backend returns empty, treat as failure: keep current text and break.
                break

            print(f"[adapt] LLM returned {len(out)} chars")

            known_plus = set(known_lemmas) | set(essential_lemmas)
            analysis_eff = self.ranker.analyze(out, known_lemmas=known_plus)
            cov_eff = float(analysis_eff.coverage if analysis_eff.coverage is not None else 1.0)
            analysis_base = self.ranker.analyze(out, known_lemmas=known_lemmas)
            cov_base = float(analysis_base.coverage if analysis_base.coverage is not None else 1.0)
            analysis_out = analysis_base  # now defined

            violations_lemmas, violations_surface = self._find_violations(
                out,
                analysis_out,
                known_lemmas,
                essential_lemmas,
                banned_lemmas,
                banned_surface,
            )

            is_valid = (not violations_lemmas) and (not violations_surface)

            if is_valid and cov_eff > best_valid_cov:
                best_valid_text = out
                best_valid_cov = cov_eff
                best_valid_mode = current_mode

            rounds.append(
                RoundSnapshot(
                    round_index=r,
                    mode=current_mode,
                    coverage=cov_eff,
                    total_tokens=int(getattr(analysis_out, "total_tokens", 0) or 0),
                    violations_lemmas=violations_lemmas,
                    violations_surface=violations_surface,
                    banned_lemmas_size=len(banned_lemmas),
                    banned_surface_size=len(banned_surface),
                    essential_lemmas=list(essential_lemmas),
                    text=out,
                )
            )

            # success
            if cov_eff >= target and is_valid:
                best_text, best_cov = out, cov_eff
                best_mode = current_mode
                if best_valid_text is not None:
                    current_text = best_valid_text
                else:
                    current_text = best_text
                break

            # Early stopping check 1: Coverage dropped from best valid
            viol_count = len(violations_lemmas) + len(violations_surface)

            # Track last few rounds
            
            if viol_count < last_viol_count:
                consecutive_no_viol_improve = 0
            else:
                consecutive_no_viol_improve += 1
            last_viol_count = viol_count

            # Only do coverage-drop early stop AFTER we've seen a valid solution
            if best_valid_text is not None:
                if cov_eff < best_valid_cov:
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0

                # Stop only if we're not improving violations either
                if consecutive_drops >= 3 and consecutive_no_viol_improve >= 3:
                    break


            # track best (but keep the previous best so stagnation is computed correctly)
            prev_best_cov = best_cov

            if cov_eff > best_cov:
                best_text, best_cov = out, cov_eff
                best_mode = current_mode  # NEW

            # stagnation logic: did our *best so far* improve meaningfully this round?
            delta_best = best_cov - prev_best_cov
            if r > 0 and delta_best < self.cfg.iteration.min_delta_coverage:
                stagnation += 1
            else:
                stagnation = 0

            # Track violations for auto-expansion - track ALL violations, not just those outside keep-set
            for lemma in violations_lemmas:
                self._violation_history[lemma] = self._violation_history.get(lemma, 0) + 1
                

            # NEW: Track surface form violations and map them to lemmas
            for sf in violations_surface:
                # Try to find the lemma for this surface form
                lemma = self._surface_to_lemma.get(sf)
                if lemma:
                    self._violation_history[lemma] = self._violation_history.get(lemma, 0) + 1
                    print(f"[adapt] Surface violation '{sf}' maps to lemma '{lemma}' (count={self._violation_history[lemma]})")
                    
            
            # expand bans using violations from THIS output
            banned_lemmas, banned_surface = self._expand_bans(
                analysis=analysis_out,
                out_text=out,
                known_lemmas=known_lemmas,
                essential_lemmas=essential_lemmas,
                banned_lemmas=banned_lemmas,
                banned_surface=banned_surface,
                violations_lemmas=violations_lemmas,
                violations_surface=violations_surface,
            )
            
            # No need to filter basic vocabulary anymore

            # Immediate escalation for big coverage gaps
            coverage_gap = target - cov_eff
            should_escalate_now = False
            
            if current_mode in [RewriteMode.NOOB, RewriteMode.ULTRA_NOOB]:
                # For noob modes, escalate after just 1 round of stagnation
                if stagnation >= 1:
                    should_escalate_now = True
                # Or if NOOB achieves less than 80% coverage
                elif current_mode == RewriteMode.NOOB and cov_eff < 0.80:
                    should_escalate_now = True
            else:
                # For other modes, check if gap is > 30%
                if coverage_gap > 0.30 and current_mode != RewriteMode.ULTRA_NOOB:
                    should_escalate_now = True
                # Or use normal stagnation logic
                elif self.cfg.iteration.auto_escalate and stagnation >= self.cfg.iteration.stagnation_rounds:
                    should_escalate_now = True

            if should_escalate_now:
                current_mode = self._escalate_mode(current_mode)
                stagnation = 0

            current_text = out

        final_text = best_valid_text if best_valid_text is not None else best_text
        final_cov = best_valid_cov if best_valid_text is not None else best_cov
        final_mode = best_valid_mode if best_valid_text is not None else best_mode

        return RewriteResult(
            final_text=final_text,
            final_coverage=final_cov,
            initial_coverage=init_cov,
            mode_used=final_mode,
            rounds=rounds,
            essential_lemmas=list(essential_lemmas),
            banned_lemmas=sorted(banned_lemmas),
            banned_surface_forms=sorted(banned_surface),
        )
    
    def _bump_lemma_ban(self, lemma: str, amount: int = 1) -> None:
        if not lemma:
            return
        self._ban_tick += 1
        self._ban_lemma_score[lemma] = self._ban_lemma_score.get(lemma, 0) + amount
        self._ban_lemma_last[lemma] = self._ban_tick

    def _bump_surface_ban(self, sf: str, amount: int = 1) -> None:
        if not sf:
            return
        self._ban_tick += 1
        self._ban_surface_score[sf] = self._ban_surface_score.get(sf, 0) + amount
        self._ban_surface_last[sf] = self._ban_tick

    def _top_banned_lemmas(self, banned: Set[str], cap: int) -> List[str]:
        items = []
        for l in banned:
            items.append((
                self._ban_lemma_score.get(l, 1),
                self._ban_lemma_last.get(l, 0),
                l,
            ))
        items.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [l for _score, _last, l in items[:cap]]

    def _top_banned_surface(self, banned: Set[str], cap: int) -> List[str]:
        items = []
        for s in banned:
            items.append((
                self._ban_surface_score.get(s, 1),
                self._ban_surface_last.get(s, 0),
                s,
            ))
        items.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [s for _score, _last, s in items[:cap]]

    # ---- mode/keep/bans policy ----

    def _choose_mode(self, coverage: float) -> RewriteMode:
        if coverage >= 0.88:
            return RewriteMode.SURGICAL
        elif coverage >= 0.70:
            return RewriteMode.SIMPLIFY_THEN_ENFORCE
        elif coverage >= 0.60:
            return RewriteMode.RETELL
        elif coverage >= 0.50:  # Changed back to 0.30
            return RewriteMode.NOOB
        else:
            # Start with ULTRA_NOOB if coverage < 30%
            return RewriteMode.ULTRA_NOOB

    def _escalate_mode(self, mode: RewriteMode) -> RewriteMode:
        escalation_path = [
            RewriteMode.SURGICAL,
            RewriteMode.SIMPLIFY_THEN_ENFORCE,
            RewriteMode.RETELL,
            RewriteMode.NOOB,
            RewriteMode.ULTRA_NOOB,
        ]
        
        try:
            current_idx = escalation_path.index(mode)
            if current_idx < len(escalation_path) - 1:
                return escalation_path[current_idx + 1]
        except ValueError:
            pass
        
        return RewriteMode.ULTRA_NOOB

    def _choose_essential_set(
        self,
        analysis,
        unknown_ranked: List[LemmaStats],
        known_lemmas: Set[str],
        target: float,
        mode: Optional[RewriteMode] = None,
    ) -> List[LemmaStats]:
        # Use a token-budget for allowed unknowns, not "k lemmas".
        T0 = int(getattr(analysis, "total_tokens", 0) or 0)
        if T0 <= 0:
            return []

        # Allowed unknown tokens = (1 - target) * T0
        # Add small slack so the model isn't fighting punctuation/tokenization noise.
        base_budget = int(round((1.0 - float(target)) * T0))
        slack = 2

        # Mode knobs (optional): in ultra-noob you can allow more unknown tokens.
        if mode == RewriteMode.ULTRA_NOOB:
            budget = max(base_budget, 18)  # or whatever you like
        elif mode == RewriteMode.NOOB:
            budget = max(base_budget, 14)
        else:
            budget = base_budget

        budget = max(0, budget + slack)

        keep = self.ranker.choose_keep_set_knapsack(
            unknown_ranked,
            token_budget=budget,
            prefer_pos=self.cfg.prefer_pos,
        )
        return keep

    def _initial_bans(
        self,
        analysis,
        known_lemmas: Set[str],
        essential_lemmas: Sequence[str],
    ) -> Tuple[Set[str], Set[str]]:
        """
        Ban every unknown lemma not in keep-set, plus a couple observed surface forms per lemma.
        """
        present_lemmas = set(s.lemma for s in analysis.ranked if s.lemma)
        banned_lemmas: Set[str] = set()
        for lemma in present_lemmas:
            if lemma in known_lemmas:
                continue
            # We don't ban free UPOS lemmas via lemma-only knowledge; rely on UPOS stats.
            st = analysis.lemma_stats.get(lemma)
            if st and _is_free(st.upos):
                continue
            if lemma in essential_lemmas:
                continue
            banned_lemmas.add(lemma)

        banned_surface: Set[str] = set()
        for lemma in banned_lemmas:
            st = analysis.lemma_stats.get(lemma)
            if not st:
                continue
            # Track ALL surface forms, not just a few
            for sf in st.surface_forms:
                norm_sf = _norm(_strip_edge_punct(sf))
                if norm_sf:
                    banned_surface.add(norm_sf)
                    # Map surface form to lemma for tracking
                    self._surface_to_lemma[norm_sf] = lemma

        return banned_lemmas, banned_surface

    def _ban_add_budget(self, total_tokens: int) -> int:
        # Remove the cap - allow banning as many lemmas as needed
        extra = int(total_tokens / max(1, self.cfg.iteration.ban_add_per_tokens))
        # No more cap - return all violations if needed
        return max(self.cfg.iteration.ban_add_floor, self.cfg.iteration.ban_add_floor + extra)

    def _expand_bans(
        self,
        *,
        analysis,
        out_text: str,
        known_lemmas: Set[str],
        essential_lemmas: Sequence[str],
        banned_lemmas: Set[str],
        banned_surface: Set[str],
        violations_lemmas: Sequence[str],
        violations_surface: Sequence[str],
    ) -> Tuple[Set[str], Set[str]]:
        """
        Add ALL violating lemmas (no budget limit); add observed surface forms for those lemmas.
        """
        banned_lemmas = set(banned_lemmas)
        banned_surface = set(banned_surface)

        # Anything that showed up as a violation gets "hotter"
        for lemma in violations_lemmas:
            self._bump_lemma_ban(lemma, amount=3)  # heavier weight for lemma violations

        for sf in violations_surface:
            self._bump_surface_ban(sf, amount=2)

        # Ban ALL violations that aren't in essential_lemmas
        for lemma in violations_lemmas:
            if lemma in essential_lemmas:
                continue
            if lemma in known_lemmas:
                continue
            st = analysis.lemma_stats.get(lemma)
            if st and _is_free(st.upos):
                continue
            
            # Add to banned set
            banned_lemmas.add(lemma)
            self._bump_lemma_ban(lemma, amount=2)

            # Also ban its surface forms
            if st:
                for sf in st.surface_forms:  # Ban ALL surface forms, not just a few
                    norm_sf = _norm(_strip_edge_punct(sf))
                    if norm_sf:
                        banned_surface.add(norm_sf)
                        self._bump_surface_ban(norm_sf, amount=1)
                        # Map surface form to lemma for tracking
                        self._surface_to_lemma[norm_sf] = lemma

        # Surface violations: add directly (already normalized by _find_violations)
        for sf in violations_surface:
            banned_surface.add(sf)
            self._bump_surface_ban(sf, amount=2)

        return banned_lemmas, banned_surface

    # ---- violation checks ----

    def _find_violations(
        self,
        text: str,
        analysis,
        known_lemmas: Set[str],
        essential_lemmas: Sequence[str],
        banned_lemmas: Set[str],
        banned_surface: Set[str],
    ) -> Tuple[List[str], List[str]]:
        essential_set = set(essential_lemmas)

        # Lemma violations: unknown content lemmas that are not in keep_set
        present_unknown: Set[str] = set()
        for st in analysis.ranked:
            lemma = st.lemma
            if not lemma:
                continue
            if lemma in known_lemmas:
                continue
            if _is_free(st.upos):
                continue
            present_unknown.add(lemma)

        violations_lemmas = sorted([l for l in present_unknown if l not in essential_set])

        # Also ensure banned lemmas are absent (even if keep-set accidentally contains them)
        violations_lemmas = _unique_preserve_order([l for l in violations_lemmas if l in present_unknown] + [l for l in sorted(banned_lemmas) if l in present_unknown])

        # Surface violations: exact token-level matches against banned surface forms (normalized)
        toks = self._surface_tokens_norm(text)
        present_surface = set(toks)
        violations_surface = sorted([sf for sf in banned_surface if sf and sf in present_surface])

        return violations_lemmas, violations_surface

    # ---- prompt builder ----

    def _build_prompt(
        self,
        mode: RewriteMode,
        passage: str,
        target: float,
        essential_lemmas: Sequence[str],
        banned_lemmas: Set[str],
        banned_surface: Set[str],
    ) -> Tuple[str, str]:
        # keep prompt sizes bounded
        essential_lemmas = list(essential_lemmas)

        banned_lemmas_list = self._top_banned_lemmas(banned_lemmas, cap=500)
        banned_surface_list = self._top_banned_surface(banned_surface, cap=500)

        if mode == RewriteMode.SURGICAL:
            return rewrite_prompts.prompt_surgical(passage, target, essential_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.SIMPLIFY_THEN_ENFORCE:
            return rewrite_prompts.prompt_simplify_then_enforce(passage, target, essential_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.RETELL:
            return rewrite_prompts.prompt_retell(passage, target, essential_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.NOOB:
            return rewrite_prompts.prompt_noob(passage, target, essential_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.ULTRA_NOOB:
            return rewrite_prompts.prompt_ultra_noob(passage, target, essential_lemmas, banned_lemmas_list, banned_surface_list)
        else:
            # Fallback to retell mode
            return rewrite_prompts.prompt_retell(passage, target, essential_lemmas, banned_lemmas_list, banned_surface_list)

    def _surface_tokens_norm(self, text: str) -> List[str]:
            # Prefer the lemmatizer’s token stream (robust to punctuation/tokenization quirks)
            if self.lemmatizer is not None:
                try:
                    rows = self.lemmatizer.lemmatize_passage(text)
                    toks = []
                    for (surface_orig, _lemma, _upos, _src) in rows:
                        s = _norm(_strip_edge_punct(surface_orig))
                        if s:
                            toks.append(s)
                    return toks
                except Exception:
                    # Don’t let tokenization failure crash adaptation; fall back.
                    pass

            # Fallback: regex-based tokens (still better than split()).
            # This finds word-like spans (Greek/Latin/digits/underscore) and then normalizes.
            raw = re.findall(r"[\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+", text)
            return [t for t in (_norm(x) for x in raw) if t]