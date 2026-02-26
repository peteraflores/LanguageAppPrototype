import stanza
from ufal.udpipe import Model as UDPipeModel, Pipeline as UDPipePipeline
import unicodedata
import csv
import re
import os
from datetime import datetime

_TOKEN_EDGE_PUNCT_RE = re.compile(
    r"^[^\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+|[^\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+$"
)

# Exclusion filters (keep consistent with how you built the lexicon)
_DIGIT_RE = re.compile(r"\d")
_ACRONYM_RE = re.compile(r"^(?:[A-ZΑ-Ω]{2,}|(?:[A-ZΑ-Ω]\.){2,}|[A-ZΑ-Ω]{2,}\.)+$")
_SKIP_UPOS = {"PUNCT", "NUM", "SYM", "X"}


def _strip_edge_punct(tok: str) -> str:
    return _TOKEN_EDGE_PUNCT_RE.sub("", tok.strip())


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip().lower()

class Lemmatizer:
    def __init__(
        self,
        surface_lexicon_path: str,
        udpipe_model_path: str,
        use_lexicon: bool = True,
        skip_propn: bool = True,
        needs_review_path: str = "needs_review_instances.csv",
        stanza_model_dir: str | None = None,
        stanza_download_method=None,
        lemma_frequency_path: str = "lemma_frequency.csv",
    ):
        self.use_lexicon = use_lexicon
        self.skip_propn = skip_propn

        self.lexicon_path = surface_lexicon_path
        self.needs_review_path = needs_review_path

        # Load trusted lexicon
        self.surface_lexicon = self.load_surface_lexicon_csv(surface_lexicon_path)

        # Fast membership check
        self._lexicon_triples = set()
        for s, pairs in self.surface_lexicon.items():
            for (l, u) in pairs:
                self._lexicon_triples.add((s, l, u))

        # Load lemma frequencies
        self.lemma_frequencies = self._load_lemma_frequencies(lemma_frequency_path)

        # Ensure review file exists
        if not os.path.exists(self.needs_review_path):
            with open(self.needs_review_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp",
                    "token_index",
                    "surface_orig",
                    "surface_norm",
                    "stanza_lemma",
                    "stanza_upos",
                    "udpipe_lemma",
                    "udpipe_upos",
                    "decision",
                    "sentence",
                ])

        # Initialize NLP engines
        self.nlp = stanza.Pipeline(
            lang="el",
            processors="tokenize,pos,lemma",
            tokenize_no_ssplit=False,
            use_gpu=False,
            verbose=False,
            model_dir=stanza_model_dir,              # NEW
            download_method=stanza_download_method,  # NEW
        )
        self.udpipe_model = UDPipeModel.load(udpipe_model_path)
        if self.udpipe_model is None:
            raise FileNotFoundError(
                f"UDPipe model failed to load. Check udpipe_model_path={udpipe_model_path!r} "
                f"(wrong path or file missing)."
            )
        self.udpipe = UDPipePipeline(
            self.udpipe_model,
            "tokenize",
            UDPipePipeline.DEFAULT,
            UDPipePipeline.DEFAULT,
            "conllu",
        )
        

    def load_surface_lexicon_csv(self, path: str) -> dict[str, set[tuple[str, str]]]:
        """
        CSV columns: surface, lemma, upos
        """
        surface2: dict[str, set[tuple[str, str]]] = {}
        if not os.path.exists(path):
            return surface2
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                surface = _norm(row["surface"])
                lemma = _norm(row["lemma"])
                upos = row["upos"].strip().upper()
                if not surface or not lemma or not upos:
                    continue
                surface2.setdefault(surface, set()).add((lemma, upos))
        return surface2

    def _load_lemma_frequencies(self, path: str) -> dict[str, int]:
        """Load lemma frequencies from CSV file."""
        frequencies = {}
        if not os.path.exists(path):
            return frequencies
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lemma = _norm(row["lemma"])
                frequency = int(row["frequency"])
                frequencies[lemma] = frequency
        return frequencies

    def _lexicon_lookup(self, surface: str, stanza_upos: str | None) -> tuple[str, str] | set[tuple[str, str]] | None:
        """
        Returns either:
        - None if no match found
        - (lemma, upos) tuple if single match
        - set of (lemma, upos) tuples if multiple UPOS options (always consult both models)
        """
        if not self.use_lexicon:
            return None
        s = _norm(surface)
        options = self.surface_lexicon.get(s)
        if not options:
            return None
            
        # If only one option, return it directly
        if len(options) == 1:
            lemma, upos = list(options)[0]
            return lemma, upos
        
        # Multiple options exist - always return all options for multi-model disambiguation
        return options

    def _parse_udpipe_conllu(self, conllu_text: str) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for line in conllu_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue
            form, lemma, upos = cols[1], cols[2], cols[3]
            out.append((form, lemma, upos))
        return out

    def _is_coherent(self, lemma: str, upos: str) -> bool:
        """Binary check: does lemma have expected ending for its UPOS?"""
        if not lemma:
            return False
            
        upos = upos.upper()
        
        if upos == "VERB":
            return lemma.endswith(('ω', 'ώ', 'ομαι', 'μαι', 'ούμαι', 'ιέμαι'))
        
        elif upos == "NOUN":
            return lemma.endswith(('ος', 'ός', 'α', 'ά', 'ο', 'ό', 'η', 'ή', 'ι', 'ί', 'μα'))
        
        elif upos == "ADJ":
            return lemma.endswith(('ος', 'ός', 'ης', 'ής', 'ύς', 'υς', 'ικος', 'ικός'))
        
        elif upos == "ADV":
            return lemma.endswith(('α', 'ά', 'ως', 'ώς'))
        
        # For other UPOS, consider all coherent
        return True

    def _character_similarity(self, surface: str, lemma: str) -> float:
        """Simple character overlap ratio based on common prefix"""
        if not surface or not lemma:
            return 0.0
            
        common_prefix_len = 0
        for i, (c1, c2) in enumerate(zip(surface, lemma)):
            if c1 == c2:
                common_prefix_len = i + 1
            else:
                break
        return common_prefix_len / max(len(surface), len(lemma))

    def _decide_from_lexicon_options(
        self,
        surface: str,
        lexicon_options: set[tuple[str, str]],
        stanza_lemma: str,
        stanza_upos: str,
        udpipe_lemma: str,
        udpipe_upos: str
    ) -> tuple[str, str, str]:
        """
        Decide which lexicon entry to use when multiple UPOS options exist.
        Returns (lemma, upos, decision_reason)
        """
        # Check if models match any lexicon option
        stanza_match = None
        udpipe_match = None
        
        stanza_norm = (_norm(stanza_lemma), stanza_upos.upper()) if stanza_lemma else None
        udpipe_norm = (_norm(udpipe_lemma), udpipe_upos.upper()) if udpipe_lemma else None
        
        for lemma, upos in lexicon_options:
            if stanza_norm and (lemma, upos) == stanza_norm:
                stanza_match = (lemma, upos)
            if udpipe_norm and (lemma, upos) == udpipe_norm:
                udpipe_match = (lemma, upos)
        
        # Case 1: Only one model matches a lexicon option
        if stanza_match and not udpipe_match:
            return (stanza_match[0], stanza_match[1], "LEXICON_SINGLE_MODEL_MATCH")
        elif udpipe_match and not stanza_match:
            return (udpipe_match[0], udpipe_match[1], "LEXICON_SINGLE_MODEL_MATCH")
        
        # Case 2: Both models match lexicon options - use frequency
        elif stanza_match and udpipe_match:
            stanza_freq = self.lemma_frequencies.get(stanza_match[0], 0)
            udpipe_freq = self.lemma_frequencies.get(udpipe_match[0], 0)
            
            if stanza_freq >= udpipe_freq:
                return (stanza_match[0], stanza_match[1], "LEXICON_BOTH_MATCH_FREQ")
            else:
                return (udpipe_match[0], udpipe_match[1], "LEXICON_BOTH_MATCH_FREQ")
        
        # Case 3: Neither model matches - pick most frequent from lexicon
        else:
            best_option = None
            best_freq = -1
            
            for lemma, upos in lexicon_options:
                freq = self.lemma_frequencies.get(lemma, 0)
                if freq > best_freq:
                    best_freq = freq
                    best_option = (lemma, upos)
            
            if best_option:
                return (best_option[0], best_option[1], "LEXICON_NO_MATCH_FREQ")
            else:
                # Fallback: just take the first option
                first_option = list(lexicon_options)[0]
                return (first_option[0], first_option[1], "LEXICON_NO_MATCH_FREQ")

    def _decide_lemma(
        self,
        surface: str,
        stanza_lemma: str,
        stanza_upos: str,
        udpipe_lemma: str,
        udpipe_upos: str
    ) -> tuple[str, str, str]:
        """
        Decide which lemma/UPOS to use when models disagree.
        Returns (lemma, upos, decision_reason)
        """
        # Case A: Full agreement
        if stanza_lemma == udpipe_lemma and stanza_upos.upper() == udpipe_upos.upper():
            return (stanza_lemma, stanza_upos.upper(), "FULL_AGREEMENT")
        
        # Case B: UPOS agreement only
        elif stanza_upos.upper() == udpipe_upos.upper():
            stanza_coherent = self._is_coherent(stanza_lemma, stanza_upos)
            udpipe_coherent = self._is_coherent(udpipe_lemma, udpipe_upos)
            
            if stanza_coherent and not udpipe_coherent:
                return (stanza_lemma, stanza_upos.upper(), "UPOS_AGREE_COHERENT_STANZA")
            elif udpipe_coherent and not stanza_coherent:
                return (udpipe_lemma, udpipe_upos.upper(), "UPOS_AGREE_COHERENT_UDPIPE")
            else:
                # Both coherent or both incoherent - pick closest to surface
                if self._character_similarity(surface, stanza_lemma) >= self._character_similarity(surface, udpipe_lemma):
                    return (stanza_lemma, stanza_upos.upper(), "UPOS_AGREE_SURFACE_SIMILAR")
                else:
                    return (udpipe_lemma, udpipe_upos.upper(), "UPOS_AGREE_SURFACE_SIMILAR")
        
        # Case C: Any disagreement (UPOS or both)
        else:
            stanza_coherent = self._is_coherent(stanza_lemma, stanza_upos)
            udpipe_coherent = self._is_coherent(udpipe_lemma, udpipe_upos)
            
            if stanza_coherent and not udpipe_coherent:
                return (stanza_lemma, stanza_upos.upper(), "DISAGREE_COHERENT_STANZA")
            elif udpipe_coherent and not stanza_coherent:
                return (udpipe_lemma, udpipe_upos.upper(), "DISAGREE_COHERENT_UDPIPE")
            else:
                # Both coherent or both incoherent - pick closest to surface
                if self._character_similarity(surface, stanza_lemma) >= self._character_similarity(surface, udpipe_lemma):
                    return (stanza_lemma, stanza_upos.upper(), "DISAGREE_SURFACE_SIMILAR")
                else:
                    return (udpipe_lemma, udpipe_upos.upper(), "DISAGREE_SURFACE_SIMILAR")

    def _keep_token(self, surface_clean: str, upos: str) -> bool:
        if not surface_clean:
            return False
        if upos.upper() in _SKIP_UPOS:
            return False
        if _DIGIT_RE.search(surface_clean):
            return False
        # acronym detection should use original surface; clean is okay as proxy
        if _ACRONYM_RE.match(surface_clean) or _ACRONYM_RE.match(surface_clean.upper()):
            return False
        return True

    def _append_needs_review(
        self,
        token_index: int,
        surface_orig: str,
        surface_norm: str,
        stanza_lemma: str | None,
        stanza_upos: str | None,
        udpipe_lemma: str | None,
        udpipe_upos: str | None,
        decision: str,
        sentence: str,
    ) -> None:
        with open(self.needs_review_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    token_index,
                    surface_orig,
                    surface_norm,
                    stanza_lemma or "",
                    (stanza_upos or "").upper(),
                    udpipe_lemma or "",
                    (udpipe_upos or "").upper(),
                    decision,
                    sentence,
                ]
            )


    def lemmatize_passage(self, passage: str) -> list[tuple[str, str, str, str]]:
        doc = self.nlp(passage)
        udpipe_sents = self._parse_udpipe_conllu_sents(self.udpipe.process(passage))

        results: list[tuple[str, str, str, str]] = []
        stream_index = 0  # index among *kept* (learnable) tokens only

        for si, sent in enumerate(doc.sentences):
            ud_tokens = udpipe_sents[si] if si < len(udpipe_sents) else []
            ud_i = 0

            for word in sent.words:
                original_surface = word.text
                surface_clean = _strip_edge_punct(original_surface)
                if not surface_clean:
                    # Not a meaningful token; don't advance stream index.
                    continue

                stanza_upos = (word.upos or "").upper()

                # Optionally skip proper nouns from the learnable stream.
                if self.skip_propn and stanza_upos == "PROPN":
                    results.append((original_surface, _norm(surface_clean), "PROPN", "Stanza"))
                    continue

                # Apply global token filters (digits, acronyms, punctuation-ish UPOS, etc.)
                if not self._keep_token(surface_clean, stanza_upos):
                    continue

                # This token counts toward the learnable stream.
                token_index = stream_index
                stream_index += 1

                surface_norm = _norm(surface_clean)

                # Trusted lexicon lookup
                lex = self._lexicon_lookup(surface_clean, stanza_upos)
                
                # Handle lexicon results
                if lex and isinstance(lex, tuple):
                    # Single match found
                    lemma, upos = lex
                    results.append((original_surface, lemma, upos, "Lexicon"))
                    continue
                elif lex and isinstance(lex, set):
                    # Multiple UPOS options - need to consult models
                    stanza_lemma = _norm(word.lemma) if word.lemma else ""
                    stanza_upos_norm = stanza_upos or ""
                    
                    udpipe_lemma, udpipe_upos, ud_i, ud_matched = self._udpipe_match_next(
                        ud_tokens, ud_i, surface_norm, lookahead=6
                    )
                    
                    # Use the multi-option decision logic
                    lemma, upos, decision_reason = self._decide_from_lexicon_options(
                        surface_norm,
                        lex,  # the set of options
                        stanza_lemma,
                        stanza_upos_norm,
                        udpipe_lemma,
                        udpipe_upos
                    )
                    
                    # Don't log - we're still using the lexicon
                    
                    results.append((original_surface, lemma, upos, "Lexicon"))
                    continue

                # No lexicon match - proceed with model-based lemmatization
                stanza_lemma = _norm(word.lemma) if word.lemma else ""
                stanza_upos_norm = stanza_upos or ""

                udpipe_lemma, udpipe_upos, ud_i, ud_matched = self._udpipe_match_next(
                    ud_tokens, ud_i, surface_norm, lookahead=6
                )

                # Decide what we output, and log issues for review
                if stanza_lemma and ud_matched and udpipe_lemma:
                    # Both have lemmas and we successfully aligned
                    lemma, upos, decision_reason = self._decide_lemma(
                        surface_norm,
                        stanza_lemma,
                        stanza_upos_norm,
                        udpipe_lemma,
                        udpipe_upos
                    )
                    
                    # Log the decision
                    self._append_needs_review(
                        token_index,
                        original_surface,
                        surface_norm,
                        stanza_lemma,
                        stanza_upos_norm,
                        udpipe_lemma,
                        udpipe_upos,
                        decision_reason,
                        sent.text,
                    )
                    
                    # Map decision reasons to simpler sources for results
                    if "STANZA" in decision_reason:
                        source = "Stanza"
                    elif "UDPIPE" in decision_reason:
                        source = "UDPipe"
                    elif decision_reason == "FULL_AGREEMENT":
                        source = "Agree(Stanza+UDPipe)"
                    else:
                        source = decision_reason
                    
                    results.append((original_surface, lemma, upos, source))
                    continue

                # If UDPipe didn't align, that's an alignment/tokenization issue—not "UDPipe had no analysis".
                if stanza_lemma and not ud_matched:
                    decision = "UDPIPE_ALIGN_MISS"
                    self._append_needs_review(
                        token_index,
                        original_surface,
                        surface_norm,
                        stanza_lemma,
                        stanza_upos_norm,
                        "",
                        "",
                        decision,
                        sent.text,
                    )
                    results.append((original_surface, stanza_lemma, stanza_upos_norm.upper(), "Stanza"))
                    continue

                # UDPipe aligned but produced no lemma (rare, but distinct from align miss)
                if stanza_lemma and ud_matched and not udpipe_lemma:
                    decision = "UDPIPE_EMPTY_LEMMA"
                    self._append_needs_review(
                        token_index,
                        original_surface,
                        surface_norm,
                        stanza_lemma,
                        stanza_upos_norm,
                        "",
                        udpipe_upos,
                        decision,
                        sent.text,
                    )
                    results.append((original_surface, stanza_lemma, stanza_upos_norm.upper(), "Stanza"))
                    continue

                # Only UDPipe has a lemma (and we aligned)
                if udpipe_lemma and not stanza_lemma and ud_matched:
                    decision = "ONLY_UDPIPE"
                    self._append_needs_review(
                        token_index,
                        original_surface,
                        surface_norm,
                        "",
                        "",
                        udpipe_lemma,
                        udpipe_upos,
                        decision,
                        sent.text,
                    )
                    results.append((original_surface, udpipe_lemma, udpipe_upos, "UDPipe"))
                    continue

                # No usable analysis from either (or stanza lemma missing and udpipe didn't match)
                decision = "NO_ANALYSIS"
                self._append_needs_review(
                    token_index,
                    original_surface,
                    surface_norm,
                    stanza_lemma or "",
                    stanza_upos_norm,
                    udpipe_lemma or "",
                    udpipe_upos or "",
                    decision,
                    sent.text,
                )
                results.append((original_surface, surface_norm, stanza_upos_norm.upper(), "Fallback"))

        return results


    def _parse_udpipe_conllu_sents(self, conllu_text: str) -> list[list[tuple[str, str, str]]]:
        sents: list[list[tuple[str, str, str]]] = []
        cur: list[tuple[str, str, str]] = []
        for line in conllu_text.splitlines():
            line = line.strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue
            form, lemma, upos = cols[1], cols[2], cols[3]
            cur.append((form, lemma, upos))
        if cur:
            sents.append(cur)
        return sents
    
    def _udpipe_match_next(
        self,
        ud_tokens: list[tuple[str, str, str]],
        start_i: int,
        target_surface_norm: str,
        lookahead: int = 6,
    ) -> tuple[str, str, int, bool]:
        """
        Returns (ud_lemma_norm, ud_upos, next_index, matched).

        - matched=False means we could not align the Stanza token to any UDPipe token
        within the lookahead window (alignment/tokenization mismatch).
        - matched=True with ud_lemma_norm=="" means we aligned, but UDPipe did not provide a lemma.

        If no match in window, returns ("","", start_i, False) (i.e., don't advance).
        """
        n = len(ud_tokens)
        for j in range(start_i, min(n, start_i + lookahead)):
            form, lemma, upos = ud_tokens[j]
            form_clean = _strip_edge_punct(form)
            if not form_clean:
                continue
            if not self._keep_token(form_clean, upos):
                continue
            if _norm(form_clean) == target_surface_norm:
                ud_lemma = _norm(lemma) if lemma and lemma != "_" else ""
                ud_upos = (upos or "").upper()
                return ud_lemma, ud_upos, j + 1, True
        return "", "", start_i, False
