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

    def _lexicon_lookup(self, surface: str, stanza_upos: str | None) -> tuple[str, str] | None:
        if not self.use_lexicon:
            return None
        s = _norm(surface)
        options = self.surface_lexicon.get(s)
        if not options:
            return None
        if stanza_upos:
            stanza_upos = stanza_upos.upper()
            for lemma, upos in options:
                if upos == stanza_upos:
                    return lemma, upos
        lemma, upos = sorted(options)[0]
        return lemma, upos

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

                # Trusted lexicon fast-path
                lex = self._lexicon_lookup(surface_clean, stanza_upos)
                if lex:
                    lemma, upos = lex
                    results.append((original_surface, lemma, upos, "Lexicon"))
                    continue

                stanza_lemma = _norm(word.lemma) if word.lemma else ""
                stanza_upos_norm = stanza_upos or ""

                udpipe_lemma, udpipe_upos, ud_i, ud_matched = self._udpipe_match_next(
                    ud_tokens, ud_i, surface_norm, lookahead=6
                )

                # Decide what we output, and log issues for review
                if stanza_lemma and ud_matched and udpipe_lemma:
                    # Both have lemmas and we successfully aligned
                    if stanza_lemma == udpipe_lemma and stanza_upos_norm.upper() == udpipe_upos:
                        # Log agreement to needs_review
                        self._append_needs_review(
                            token_index,
                            original_surface,
                            surface_norm,
                            stanza_lemma,
                            stanza_upos_norm,
                            udpipe_lemma,
                            udpipe_upos,
                            "AGREE",
                            sent.text,
                        )
                        results.append((original_surface, stanza_lemma, udpipe_upos, "Agree(Stanza+UDPipe)"))
                    else:
                        decision = "DISAGREE_LEMMA" if stanza_lemma != udpipe_lemma else "DISAGREE_UPOS"
                        self._append_needs_review(
                            token_index,
                            original_surface,
                            surface_norm,
                            stanza_lemma,
                            stanza_upos_norm,
                            udpipe_lemma,
                            udpipe_upos,
                            decision,
                            sent.text,
                        )
                        # runtime choice: prefer Stanza
                        results.append((original_surface, stanza_lemma, stanza_upos_norm.upper(), "Stanza"))
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
