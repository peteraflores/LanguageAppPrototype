from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import csv
import os
import unicodedata
import math

FREE_UPOS: Set[str] = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON"}

def _is_free_upos(upos: str) -> bool:
    return (upos or "").upper() in FREE_UPOS

def _norm_lemma(s: str) -> str:
    # Match lemmatizer._norm behavior: NFC + strip + lowercase
    return unicodedata.normalize("NFC", str(s)).strip().lower()


def load_known_lemmas_csv(
    path: str,
    *,
    keep_upos: Optional[Set[str]] = None,
    drop_free_upos_rows: bool = False,
) -> Set[str]:
    """
    Read known_lemma.csv with schema: lemma,upos

    Returns a Set[str] of normalized lemmas.

    Options:
      - keep_upos: if provided, only include rows whose UPOS is in this set.
                  Example: {"NOUN","VERB","ADJ","ADV"}
      - drop_free_upos_rows: if True, rows whose UPOS is in FREE_UPOS are ignored.
        (You already treat FREE_UPOS as "known for coverage" in analyze(), so this is
         mostly useful if you want the file to represent only content-lemma knowledge.)
    """
    known: Set[str] = set()

    if not path:
        return known
    if not os.path.exists(path):
        return known

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Be strict about expected headers; fail loud if the file is wrong,
        # because silent partial loads are a debugging nightmare.
        if not reader.fieldnames:
            return known

        # Allow minor header variations but insist lemma exists.
        # (You said schema is lemma,upos, so we treat that as canonical.)
        fieldnames = {h.strip().lower() for h in reader.fieldnames if h}
        if "lemma" not in fieldnames:
            raise ValueError(f"{path!r} is missing required header 'lemma'. Found: {reader.fieldnames!r}")

        for row in reader:
            if not row:
                continue

            lemma_raw = row.get("lemma", "")
            upos_raw = row.get("upos", "")

            lemma = _norm_lemma(lemma_raw)
            if not lemma:
                continue

            upos = (upos_raw or "").strip().upper()

            if drop_free_upos_rows and _is_free_upos(upos):
                continue

            if keep_upos is not None:
                # If the CSV row has no UPOS, it won't pass the filter.
                if upos not in keep_upos:
                    continue

            known.add(lemma)

    return known

def load_known_lemma_map_csv(path: str) -> Dict[str, str]:
    """
    Read known_lemma.csv and return Dict[lemma_norm -> UPOS].
    If lemma repeats, the last row wins.
    """
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        fieldnames = {h.strip().lower() for h in reader.fieldnames if h}
        if "lemma" not in fieldnames:
            raise ValueError(f"{path!r} is missing required header 'lemma'. Found: {reader.fieldnames!r}")

        for row in reader:
            lemma = _norm_lemma(row.get("lemma", ""))
            if not lemma:
                continue
            upos = (row.get("upos", "") or "").strip().upper()
            out[lemma] = upos

    return out


def load_lemma_frequency_csv(path: str) -> Dict[str, int]:
    """
    Load lemma_frequency.csv with columns: lemma, frequency, rank
    Returns dict mapping normalized lemma -> frequency count
    """
    frequency_map: Dict[str, int] = {}
    if not path or not os.path.exists(path):
        return frequency_map
    
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = _norm_lemma(row.get("lemma", ""))
            if not lemma:
                continue
            try:
                freq = int(row.get("frequency", 0))
                frequency_map[lemma] = freq
            except ValueError:
                continue
    
    return frequency_map


@dataclass(frozen=True)
class LemmaStats:
    lemma: str
    upos: str
    token_count: int
    sentence_count: int
    first_sentence_index: int
    surface_forms: Tuple[str, ...]
    score: float


@dataclass(frozen=True)
class PassageAnalysis:
    total_tokens: int
    lemma_stats: Dict[str, LemmaStats]
    ranked: List[LemmaStats]

    # optional, only meaningful if you pass known_lemmas
    known_tokens: Optional[int] = None
    coverage: Optional[float] = None


class LemmaSalienceRanker:
    """
    Ranks lemmas by "importance/salience" within a passage based on:
      - SUBTLEX frequency (if provided)
      - dispersion across sentences
      - frequency in passage
      - POS bonus (content words)
      - early appearance bonus

    Does NOT require a known-lemma list.
    """

    def __init__(
        self,
        lemmatizer,
        *,
        w_sentence_count: float = 3.0,
        w_token_count: float = 1.0,
        w_frequency: float = 10.0,  # Weight for SUBTLEX frequency
        early_bonus_s0: float = 1.0,
        early_bonus_s1: float = 1.0,
        frequency_map: Optional[Dict[str, int]] = None,
    ):
        self.lemmatizer = lemmatizer
        self.w_sentence_count = w_sentence_count
        self.w_token_count = w_token_count
        self.w_frequency = w_frequency
        self.early_bonus_s0 = early_bonus_s0
        self.early_bonus_s1 = early_bonus_s1
        self.frequency_map = frequency_map or {}

    def analyze(
        self,
        passage: str,
        known_lemmas: Optional[Set[str]] = None,
    ) -> PassageAnalysis:
        """
        If known_lemmas is provided, compute coverage over kept tokens and
        still return ranking over ALL lemmas (use filter_known to get unknowns).
        """
        # Get sentence segmentation (we rely on your stanza pipeline)
        doc = self.lemmatizer.nlp(passage)

        # Lemmatize passage into the "kept token stream" you already defined
        # Each element: (surface_orig, lemma_norm, upos, source)
        rows = self.lemmatizer.lemmatize_passage(passage)

        # Map kept-token index -> sentence index (must match lemmatizer's kept-token policy)
        # This assumes your lemmatizer's token_index increments only for kept tokens (and not PROPN if skipped)
        token_to_sentence: Dict[int, int] = {}
        t = 0
        for s_i, sent in enumerate(doc.sentences):
            for w in sent.words:
                surf = getattr(w, "text", "") or ""
                surf_clean = surf.strip()
                if not surf_clean:
                    continue
                upos = (getattr(w, "upos", "") or "").upper()

                if self.lemmatizer.skip_propn and upos == "PROPN":
                    continue
                if not self.lemmatizer._keep_token(surf_clean, upos):
                    continue

                token_to_sentence[t] = s_i
                t += 1

        # Collect stats for ALL lemmas
        data = defaultdict(lambda: {
            "upos": "",
            "token_count": 0,
            "sentences": set(),
            "first_sentence": None,
            "surface_forms": set(),
        })

        total_tokens = 0
        known_tokens = 0

        for idx, (surface, lemma, upos, _source) in enumerate(rows):
            total_tokens += 1
            if known_lemmas is not None and (lemma in known_lemmas or _is_free_upos(upos)):
                known_tokens += 1

            s_i = token_to_sentence.get(idx, 0)

            d = data[lemma]
            d["upos"] = upos
            d["token_count"] += 1
            d["sentences"].add(s_i)
            d["surface_forms"].add(surface)

            if d["first_sentence"] is None:
                d["first_sentence"] = s_i

        stats_list: List[LemmaStats] = []
        for lemma, d in data.items():
            sentence_count = len(d["sentences"])
            token_count = d["token_count"]
            first_sentence = int(d["first_sentence"] if d["first_sentence"] is not None else 0)
            upos = (d["upos"] or "").upper()

            # Get SUBTLEX frequency if available
            subtlex_freq = self.frequency_map.get(lemma, 0)
            
            # Use raw frequency directly for continuous scoring
            # Normalize by dividing by 1000 to keep scores in reasonable range
            frequency_score = subtlex_freq / 1000.0
            
            score = (
                self.w_frequency * frequency_score
                + self.w_sentence_count * sentence_count
                + self.w_token_count * token_count
                + self._pos_bonus(upos)
                + self._early_bonus(first_sentence)
            )

            stats_list.append(
                LemmaStats(
                    lemma=lemma,
                    upos=upos,
                    token_count=token_count,
                    sentence_count=sentence_count,
                    first_sentence_index=first_sentence,
                    surface_forms=tuple(sorted(d["surface_forms"])),
                    score=float(score),
                )
            )

        stats_list.sort(key=lambda x: x.score, reverse=True)
        stats_dict = {s.lemma: s for s in stats_list}

        coverage = None
        if known_lemmas is not None:
            coverage = (known_tokens / total_tokens) if total_tokens else 1.0

        return PassageAnalysis(
            total_tokens=total_tokens,
            lemma_stats=stats_dict,
            ranked=stats_list,
            known_tokens=(known_tokens if known_lemmas is not None else None),
            coverage=coverage,
        )

    def filter_known(self, analysis: PassageAnalysis, known_lemmas: Set[str]) -> List[LemmaStats]:
        """Return ranked stats excluding known lemmas AND free UPOS."""
        return [
            s for s in analysis.ranked
            if (s.lemma not in known_lemmas) and (not _is_free_upos(s.upos))
        ]

    def choose_keep_set(
        self,
        unknown_ranked: List[LemmaStats],
        *,
        k: int,
        prefer_pos: Optional[Set[str]] = None,
    ) -> List[LemmaStats]:
        """
        Pick the top-k unknown lemmas, optionally restricting to preferred POS.
        Example: prefer_pos={"NOUN","VERB","ADJ","ADV"}.
        """
        if prefer_pos:
            filtered = [s for s in unknown_ranked if s.upos in prefer_pos]
            if len(filtered) >= k:
                return filtered[:k]
            # if not enough, top off with remaining
            rest = [s for s in unknown_ranked if s not in filtered]
            return filtered + rest[: max(0, k - len(filtered))]
        return unknown_ranked[:k]

    def build_known_from_corpus(
        self,
        texts: Iterable[str],
        *,
        top_n: int = 1000,
    ) -> Set[str]:
        """
        Bootstrap a dummy known-lemma set by lemma frequency over a corpus.
        This is *not* a true learner model; it’s for testing/ranking/debug.
        """
        counts = Counter()
        for text in texts:
            rows = self.lemmatizer.lemmatize_passage(text)
            for _surface, lemma, _upos, _source in rows:
                if lemma:
                    counts[lemma] += 1
        return {lemma for (lemma, _c) in counts.most_common(top_n)}

    def load_texts_from_paths(self, paths: Iterable[str], *, suffixes: Tuple[str, ...] = (".txt",)) -> List[str]:
        """
        Convenience: read a bunch of .txt files (or directories containing .txt).
        """
        out: List[str] = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file() and child.suffix.lower() in suffixes:
                        out.append(child.read_text(encoding="utf-8", errors="ignore"))
            elif path.is_file():
                out.append(path.read_text(encoding="utf-8", errors="ignore"))
        return out

    def _pos_bonus(self, upos: str) -> float:
        if upos in {"NOUN", "VERB"}:
            return 3.0
        if upos in {"ADJ", "ADV"}:
            return 2.0
        if upos == "PROPN":
            return 1.0
        return 0.0

    def _early_bonus(self, sentence_index: int) -> float:
        if sentence_index == 0:
            return self.early_bonus_s0
        if sentence_index == 1:
            return self.early_bonus_s1
        return 0.0