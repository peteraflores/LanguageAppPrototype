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


def _ensure_csv_header(path: str, header: list[str]) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)


class Lemmatizer:
    def __init__(
        self,
        surface_lexicon_path: str,
        udpipe_model_path: str,
        use_lexicon: bool = True,
        skip_propn: bool = True,
        auto_promote_agree: bool = True,
        agree_min_count: int = 3,
        agree_counts_path: str = "agree_counts.csv",
        needs_review_path: str = "needs_review_instances.csv",
    ):
        self.use_lexicon = use_lexicon
        self.skip_propn = skip_propn

        self.lexicon_path = surface_lexicon_path
        self.needs_review_path = needs_review_path
        self.agree_counts_path = agree_counts_path

        self.auto_promote_agree = auto_promote_agree
        self.agree_min_count = max(1, int(agree_min_count))

        # Load trusted lexicon
        self.surface_lexicon = self.load_surface_lexicon_csv(surface_lexicon_path)

        # Fast membership check
        self._lexicon_triples = set()
        for s, pairs in self.surface_lexicon.items():
            for (l, u) in pairs:
                self._lexicon_triples.add((s, l, u))

        # Load persistent agreement counts
        self._agree_counts = self._load_agree_counts(self.agree_counts_path)
        self._agree_counts_dirty = False

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
        )
        self.udpipe_model = UDPipeModel.load(udpipe_model_path)
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

    def _maybe_promote_agree(self, surface_norm: str, lemma_norm: str, upos: str) -> bool:
        """
        Promote only when Stanza and UDPipe agree (caller enforces) and the agreement
        has been observed >= agree_min_count times (persisted in agree_counts.csv).

        This is the ONLY promotion entry point. It increments the in-memory counter and
        marks agree-counts as dirty; it does not write agree_counts.csv. The caller
        should flush counts periodically (e.g., once per passage).

        Returns True if the (surface, lemma, upos) triple was appended to the trusted lexicon.
        """
        triple = (surface_norm, lemma_norm, upos)
        if triple in self._lexicon_triples:
            return False

        # Increment persistent agreement count (kept in memory until flushed)
        self._agree_counts[triple] = self._agree_counts.get(triple, 0) + 1
        self._agree_counts_dirty = True

        if self._agree_counts[triple] < self.agree_min_count:
            return False

        # Append to trusted lexicon (single code path)
        self._append_to_lexicon(surface_norm, lemma_norm, upos)
        # Drop counter now that it's promoted
        self._agree_counts.pop(triple, None)
        self._agree_counts_dirty = True
        return True

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
                        if self.auto_promote_agree:
                            self._maybe_promote_agree(surface_norm, stanza_lemma, udpipe_upos)
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

        # Flush agree-counts once per passage (avoid per-token disk writes)
        if getattr(self, "_agree_counts_dirty", False):
            self._save_agree_counts()
            self._agree_counts_dirty = False

        return results

    def _load_agree_counts(self, path: str) -> dict[tuple[str, str, str], int]:
        counts: dict[tuple[str, str, str], int] = {}
        if not os.path.exists(path):
            return counts
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                s = row["surface"].strip()
                l = row["lemma"].strip()
                u = row["upos"].strip().upper()
                c = int(row["count"])
                counts[(s, l, u)] = c
        return counts

    def _save_agree_counts(self) -> None:
        # write atomically so you don’t corrupt the file on crash
        tmp = self.agree_counts_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["surface", "lemma", "upos", "count"])
            for (s, l, u), c in sorted(self._agree_counts.items()):
                w.writerow([s, l, u, c])
        os.replace(tmp, self.agree_counts_path)

    def _append_to_lexicon(self, surface: str, lemma: str, upos: str) -> None:
        surface = _norm(surface)
        lemma = _norm(lemma)
        upos = (upos or "").strip().upper()
        if not surface or not lemma or not upos:
            return

        triple = (surface, lemma, upos)
        if triple in self._lexicon_triples:
            return

        write_header = not os.path.exists(self.lexicon_path)
        with open(self.lexicon_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["surface", "lemma", "upos"])
            w.writerow([surface, lemma, upos])

        self.surface_lexicon.setdefault(surface, set()).add((lemma, upos))
        self._lexicon_triples.add(triple)
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
