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
            tokenize_no_ssplit=True,
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
        Promote only when Stanza and UDPipe agree (caller enforces) and
        the agreement has been observed >= agree_min_count times this run.
        Returns True if appended to trusted lexicon.
        """
        triple = (surface_norm, lemma_norm, upos)
        if triple in self._lexicon_triples:
            return False

        # count agreement occurrences this run
        self._agree_counts[triple] = self._agree_counts.get(triple, 0) + 1
        if self._agree_counts[triple] < self.agree_min_count:
            return False

        # append to trusted lexicon
        write_header = not os.path.exists(self.lexicon_path)
        with open(self.lexicon_path, "a", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            if write_header:
                writer.writerow(["surface", "lemma", "upos"])
            writer.writerow([surface_norm, lemma_norm, upos])

        # update in-memory
        self.surface_lexicon.setdefault(surface_norm, set()).add((lemma_norm, upos))
        self._lexicon_triples.add(triple)
        return True

    def lemmatize_passage(self, passage: str) -> list[tuple[str, str, str, str]]:
        doc = self.nlp(passage)

        # Run UDPipe once for the whole passage (analysis source)
        udpipe_tokens = self._parse_udpipe_conllu(self.udpipe.process(passage))
        ud_i = 0

        results: list[tuple[str, str, str, str]] = []
        token_index = 0

        for sent in doc.sentences:
            for word in sent.words:
                original_surface = word.text
                surface_clean = _strip_edge_punct(original_surface)
                if not surface_clean:
                    continue

                stanza_upos = (word.upos or "").upper()

                # Optional: treat proper nouns separately
                if self.skip_propn and stanza_upos == "PROPN":
                    results.append((original_surface, _norm(surface_clean), "PROPN", "Stanza"))
                    token_index += 1
                    continue

                # Apply same token filters you used when building the lexicon
                if not self._keep_token(surface_clean, stanza_upos or ""):
                    token_index += 1
                    continue

                surface_norm = _norm(surface_clean)

                # Lexicon fast path
                lex = self._lexicon_lookup(surface_clean, stanza_upos)
                if lex:
                    lemma, upos = lex
                    results.append((original_surface, lemma, upos, "Lexicon"))
                    token_index += 1
                    continue

                # Get Stanza analysis
                stanza_lemma = _norm(word.lemma) if word.lemma else ""
                stanza_upos_norm = stanza_upos or ""

                # Get UDPipe analysis aligned by surface (best-effort)
                udpipe_lemma = ""
                udpipe_upos = ""

                # advance until we find a matching surface_norm or we run out
                target = surface_norm
                while ud_i < len(udpipe_tokens):
                    form, lemma, upos = udpipe_tokens[ud_i]
                    ud_i += 1

                    form_clean = _strip_edge_punct(form)
                    if not form_clean:
                        continue
                    if not self._keep_token(form_clean, upos):
                        continue

                    if _norm(form_clean) == target:
                        udpipe_lemma = _norm(lemma) if lemma and lemma != "_" else ""
                        udpipe_upos = (upos or "").upper()
                        break

                # Decide what we output, and log disagreements
                decision = ""
                if stanza_lemma and udpipe_lemma:
                    if stanza_lemma == udpipe_lemma and stanza_upos_norm.upper() == udpipe_upos:
                        decision = "AGREE"

                        triple = (surface_norm, stanza_lemma, udpipe_upos)

                        if self.auto_promote_agree and triple not in self._lexicon_triples:
                            new_count = self._bump_agree_count(*triple)

                            if new_count >= self.agree_min_count:
                                self._append_to_lexicon(*triple)

                            self._save_agree_counts()

                        results.append(
                            (original_surface, stanza_lemma, udpipe_upos, "Agree(Stanza+UDPipe)")
                        )
                    else:
                        # disagreement
                        if stanza_lemma != udpipe_lemma:
                            decision = "DISAGREE_LEMMA"
                        else:
                            decision = "DISAGREE_UPOS"
                        self._append_needs_review(
                            token_index,
                            original_surface,
                            surface_norm,
                            stanza_lemma,
                            stanza_upos_norm,
                            udpipe_lemma,
                            udpipe_upos,
                            decision,
                            passage,
                        )
                        # runtime choice: prefer Stanza (you can flip this)
                        results.append((original_surface, stanza_lemma, stanza_upos_norm.upper(), "Stanza"))
                elif stanza_lemma and not udpipe_lemma:
                    decision = "ONLY_STANZA"
                    self._append_needs_review(
                        token_index,
                        original_surface,
                        surface_norm,
                        stanza_lemma,
                        stanza_upos_norm,
                        "",
                        "",
                        decision,
                        passage,
                    )
                    results.append((original_surface, stanza_lemma, stanza_upos_norm.upper(), "Stanza"))
                elif udpipe_lemma and not stanza_lemma:
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
                        passage,
                    )
                    results.append((original_surface, udpipe_lemma, udpipe_upos, "UDPipe"))
                else:
                    decision = "NO_ANALYSIS"
                    self._append_needs_review(
                        token_index,
                        original_surface,
                        surface_norm,
                        "",
                        stanza_upos_norm,
                        "",
                        "",
                        decision,
                        passage,
                    )
                    results.append((original_surface, surface_norm, stanza_upos_norm.upper(), "Fallback"))

                token_index += 1

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

    def _bump_agree_count(self, surface: str, lemma: str, upos: str) -> int:
        key = (surface, lemma, upos)
        self._agree_counts[key] = self._agree_counts.get(key, 0) + 1
        return self._agree_counts[key]

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
        write_header = not os.path.exists(self.lexicon_path)
        with open(self.lexicon_path, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["surface", "lemma", "upos"])
            w.writerow([surface, lemma, upos])

        self.surface_lexicon.setdefault(surface, set()).add((lemma, upos))
        self._lexicon_triples.add((surface, lemma, upos))