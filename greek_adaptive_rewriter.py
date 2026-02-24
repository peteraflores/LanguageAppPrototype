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
from lemma_salience import LemmaSalienceRanker, LemmaStats, FREE_UPOS
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
    surface_examples_per_lemma: int = 2 # how many observed surface forms to ban per lemma
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
    # Track violations for auto-expansion
    violation_threshold: int = 2  # Add to keep-set after this many violations
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
    keep_lemmas: List[str]
    text: str

@dataclass
class RewriteResult:
    final_text: str
    final_coverage: float
    initial_coverage: float
    mode_used: RewriteMode
    rounds: List[RoundSnapshot]
    keep_lemmas: List[str]
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
                    keep_lemmas=[],
                    banned_lemmas=[],
                    banned_surface_forms=[],
                )

        # initial analysis
        init_analysis = self.ranker.analyze(passage, known_lemmas=known_lemmas)
        init_cov = float(init_analysis.coverage if init_analysis.coverage is not None else 1.0)

        chosen_mode = mode or self._choose_mode(init_cov)

        unknown_ranked = self.ranker.filter_known(init_analysis, known_lemmas)
        keep_stats = self._choose_keep_set(init_analysis, unknown_ranked, known_lemmas, target, mode=chosen_mode)
        keep_lemmas = [s.lemma for s in keep_stats]

        banned_lemmas, banned_surface = self._initial_bans(init_analysis, known_lemmas, keep_lemmas)

        rounds: List[RoundSnapshot] = []
        best_text = passage
        best_cov = init_cov
        best_mode = chosen_mode  # NEW: best-so-far provenance
        stagnation = 0

        best_valid_text: Optional[str] = None
        best_valid_cov: float = -1.0
        best_valid_mode: RewriteMode = chosen_mode

        current_text = passage
        current_mode = chosen_mode

        for r in range(self.cfg.iteration.max_rounds):
            system, user = self._build_prompt(
                current_mode,
                current_text,
                target,
                keep_lemmas,
                banned_lemmas,
                banned_surface,
            )

            print(f"[adapt] round={r} mode={current_mode} keep={len(keep_lemmas)} banned_lemmas={len(banned_lemmas)} banned_surface={len(banned_surface)}")
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

            analysis = self.ranker.analyze(out, known_lemmas=known_lemmas)
            cov = float(analysis.coverage if analysis.coverage is not None else 1.0)
            violations_lemmas, violations_surface = self._find_violations(out, analysis, known_lemmas, keep_lemmas, banned_lemmas, banned_surface)

            is_valid = (not violations_lemmas) and (not violations_surface)

            if is_valid and cov > best_valid_cov:
                best_valid_text = out
                best_valid_cov = cov
                best_valid_mode = current_mode

            rounds.append(
                RoundSnapshot(
                    round_index=r,
                    mode=current_mode,
                    coverage=cov,
                    total_tokens=analysis.total_tokens,
                    violations_lemmas=violations_lemmas,
                    violations_surface=violations_surface,
                    banned_lemmas_size=len(banned_lemmas),
                    banned_surface_size=len(banned_surface),
                    keep_lemmas=list(keep_lemmas),
                    text=out,
                )
            )

            # success
            if cov >= target and is_valid:
                best_text, best_cov = out, cov
                best_mode = current_mode
                # best_valid_* will already update above
                current_text = out
                break

            # track best (but keep the previous best so stagnation is computed correctly)
            prev_best_cov = best_cov

            if cov > best_cov:
                best_text, best_cov = out, cov
                best_mode = current_mode  # NEW

            # stagnation logic: did our *best so far* improve meaningfully this round?
            delta_best = best_cov - prev_best_cov
            if r > 0 and delta_best < self.cfg.iteration.min_delta_coverage:
                stagnation += 1
            else:
                stagnation = 0

            # Track violations for auto-expansion
            for lemma in violations_lemmas:
                if lemma not in keep_lemmas:
                    self._violation_history[lemma] = self._violation_history.get(lemma, 0) + 1
                    
                    # Auto-add to keep-set if violated too many times
                    if self._violation_history[lemma] >= self.cfg.violation_threshold:
                        print(f"[adapt] Auto-adding {lemma} to keep-set after {self._violation_history[lemma]} violations")
                        keep_lemmas.append(lemma)
                        # Reset violation count
                        self._violation_history[lemma] = 0
            
            # expand bans using violations from THIS output
            banned_lemmas, banned_surface = self._expand_bans(
                analysis=analysis,
                out_text=out,
                known_lemmas=known_lemmas,
                keep_lemmas=keep_lemmas,
                banned_lemmas=banned_lemmas,
                banned_surface=banned_surface,
                violations_lemmas=violations_lemmas,
                violations_surface=violations_surface,
            )
            
            # No need to filter basic vocabulary anymore

            # Immediate escalation for big coverage gaps
            coverage_gap = target - cov
            should_escalate_now = False
            
            if current_mode in [RewriteMode.NOOB, RewriteMode.ULTRA_NOOB]:
                # For noob modes, escalate after just 1 round of stagnation
                if stagnation >= 1:
                    should_escalate_now = True
                # Or if NOOB achieves less than 80% coverage
                elif current_mode == RewriteMode.NOOB and cov < 0.80:
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
            keep_lemmas=list(keep_lemmas),
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
        elif coverage >= 0.50:
            return RewriteMode.RETELL
        elif coverage >= 0.30:  # Changed back to 0.30
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

    def _choose_keep_set(
        self,
        analysis,
        unknown_ranked: List[LemmaStats],
        known_lemmas: Set[str],
        target: float,
        mode: Optional[RewriteMode] = None,
    ) -> List[LemmaStats]:
        # For noob modes, use adjusted keep-sets
        if mode == RewriteMode.ULTRA_NOOB:
            k = 5  # Increased from 2 to 5-6 new lemmas for ultra-noob
        elif mode == RewriteMode.NOOB:
            k = 6  # Increased from 4 to 6 new lemmas for noob
        else:
            # Original K policy: scale with length + gap
            T = int(getattr(analysis, "total_tokens", 0) or 0)
            c = float(getattr(analysis, "coverage", 1.0) or 1.0)
            gap = max(0.0, target - c)

            k_base = int(round(T / 35.0)) if T else 3
            k_base = max(3, min(12, k_base))

            k = k_base
            if gap >= 0.20:
                k += 6
            elif gap >= 0.10:
                k += 3
            elif gap < 0.05:
                k -= 1

            k = max(2, min(18, k))  # allow a bit more headroom

        keep = self.ranker.choose_keep_set(unknown_ranked, k=k, prefer_pos=self.cfg.prefer_pos)
        return keep

    def _initial_bans(
        self,
        analysis,
        known_lemmas: Set[str],
        keep_lemmas: Sequence[str],
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
            if lemma in keep_lemmas:
                continue
            banned_lemmas.add(lemma)

        banned_surface: Set[str] = set()
        for lemma in banned_lemmas:
            st = analysis.lemma_stats.get(lemma)
            if not st:
                continue
            for sf in list(st.surface_forms)[: self.cfg.iteration.surface_examples_per_lemma]:
                banned_surface.add(_norm(_strip_edge_punct(sf)))

        return banned_lemmas, banned_surface

    def _ban_add_budget(self, total_tokens: int) -> int:
        extra = int(total_tokens / max(1, self.cfg.iteration.ban_add_per_tokens))
        return max(self.cfg.iteration.ban_add_floor, min(self.cfg.iteration.ban_add_cap, self.cfg.iteration.ban_add_floor + extra))

    def _expand_bans(
        self,
        *,
        analysis,
        out_text: str,
        known_lemmas: Set[str],
        keep_lemmas: Sequence[str],
        banned_lemmas: Set[str],
        banned_surface: Set[str],
        violations_lemmas: Sequence[str],
        violations_surface: Sequence[str],
    ) -> Tuple[Set[str], Set[str]]:
        """
        Add top offending lemmas by token_count; add observed surface forms for those lemmas.
        """
        banned_lemmas = set(banned_lemmas)
        banned_surface = set(banned_surface)

        budget = self._ban_add_budget(int(analysis.total_tokens or 0))

        # Anything that showed up as a violation gets “hotter”
        for lemma in violations_lemmas:
            self._bump_lemma_ban(lemma, amount=3)  # heavier weight for lemma violations

        for sf in violations_surface:
            self._bump_surface_ban(sf, amount=2)

        # Build offender stats (token_count desc)
        offenders: List[Tuple[int, str]] = []
        for lemma in violations_lemmas:
            if lemma in keep_lemmas:
                continue
            if lemma in known_lemmas:
                continue
            st = analysis.lemma_stats.get(lemma)
            if st and _is_free(st.upos):
                continue
            tc = int(st.token_count) if st else 1
            offenders.append((tc, lemma))

        offenders.sort(reverse=True)
        to_add = [lemma for _tc, lemma in offenders if lemma not in banned_lemmas][:budget]

        for lemma in to_add:
            banned_lemmas.add(lemma)
            self._bump_lemma_ban(lemma, amount=2)

            st = analysis.lemma_stats.get(lemma)
            if st:
                for sf in list(st.surface_forms)[: self.cfg.iteration.surface_examples_per_lemma]:
                    norm_sf = _norm(_strip_edge_punct(sf))
                    if norm_sf:
                        banned_surface.add(norm_sf)
                        self._bump_surface_ban(norm_sf, amount=1)

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
        keep_lemmas: Sequence[str],
        banned_lemmas: Set[str],
        banned_surface: Set[str],
    ) -> Tuple[List[str], List[str]]:
        keep_set = set(keep_lemmas)

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

        violations_lemmas = sorted([l for l in present_unknown if l not in keep_set])

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
        keep_lemmas: Sequence[str],
        banned_lemmas: Set[str],
        banned_surface: Set[str],
    ) -> Tuple[str, str]:
        # keep prompt sizes bounded
        keep_lemmas = list(keep_lemmas)
        banned_lemmas_list = self._top_banned_lemmas(banned_lemmas, cap=500)
        banned_surface_list = self._top_banned_surface(banned_surface, cap=500)

        if mode == RewriteMode.SURGICAL:
            return rewrite_prompts.prompt_surgical(passage, target, keep_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.SIMPLIFY_THEN_ENFORCE:
            return rewrite_prompts.prompt_simplify_then_enforce(passage, keep_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.RETELL:
            return rewrite_prompts.prompt_retell(passage, keep_lemmas, banned_lemmas_list, banned_surface_list, target_sentences=self.cfg.retell_target_sentences)
        elif mode == RewriteMode.NOOB:
            return rewrite_prompts.prompt_noob(passage, target, keep_lemmas, banned_lemmas_list, banned_surface_list)
        elif mode == RewriteMode.ULTRA_NOOB:
            return rewrite_prompts.prompt_ultra_noob(passage, keep_lemmas, banned_lemmas_list, banned_surface_list, target_sentences=self.cfg.ultra_noob_target_sentences)
        else:
            # Fallback to retell mode
            return rewrite_prompts.prompt_retell(passage, keep_lemmas, banned_lemmas_list, banned_surface_list, target_sentences=self.cfg.retell_target_sentences)

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