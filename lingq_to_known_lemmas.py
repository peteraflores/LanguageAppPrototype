import csv
import re
import sys
import unicodedata
from collections import defaultdict

import stanza

GREEK_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]")

# Token splitter for LingQ "terms" (keeps Greek tokens; splits on whitespace and common punctuation)
SPLIT_RE = re.compile(r"[^\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+")

SKIP_UPOS = {"PUNCT"}  # for term lists, be much less aggressive than your passage filters


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()


def norm_lower(s: str) -> str:
    return norm(s).lower()


def guess_term_column(fieldnames):
    # Prefer obvious names
    for c in ["term", "Term", "expression", "Expression", "word", "Word", "text", "Text"]:
        if c in fieldnames:
            return c
    # Otherwise pick the column with the most Greek letters in first N rows
    return None


def extract_tokens(term: str):
    term = norm(term)
    parts = [p for p in SPLIT_RE.split(term) if p]
    # Keep tokens that contain at least one Greek letter OR are alphabetic (to not drop Greeklish if you decide later)
    toks = [p for p in parts if GREEK_RE.search(p) or p.isalpha()]
    return toks


def main(in_csv: str, out_csv: str, unknown_csv: str = "lingq_unlemmatized.csv"):
    # Stanza in pretokenized mode: we provide tokens directly
    nlp = stanza.Pipeline(
        lang="el",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=True,
        use_gpu=False,
        verbose=False,
    )

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("No header row found in input CSV.")

        term_col = guess_term_column(reader.fieldnames)
        rows = list(reader)

        if term_col is None:
            # fallback: find a column with most Greek
            best = None
            best_score = -1
            for c in reader.fieldnames:
                score = sum(1 for r in rows[:200] if GREEK_RE.search(norm(r.get(c, ""))))
                if score > best_score:
                    best_score = score
                    best = c
            term_col = best

        if term_col is None:
            raise RuntimeError("Could not determine term column.")

    lemma_counts = defaultdict(int)
    lemma_upos = {}  # lemma -> upos (first seen)
    lemma_examples = defaultdict(set)

    unknown_rows = []

    for r in rows:
        term = norm(r.get(term_col, ""))
        if not term:
            continue

        toks = extract_tokens(term)
        if not toks:
            unknown_rows.append((term, "NO_TOKENS"))
            continue

        # Feed as one "sentence" with pretokenized tokens
        doc = nlp([toks])

        any_ok = False
        for sent in doc.sentences:
            for w in sent.words:
                upos = (w.upos or "").upper()
                if upos in SKIP_UPOS:
                    continue

                lemma = norm_lower(w.lemma if w.lemma else w.text)
                if not lemma:
                    continue

                any_ok = True
                lemma_counts[(lemma, upos)] += 1
                lemma_examples[(lemma, upos)].add(norm(w.text))

        if not any_ok:
            unknown_rows.append((term, "NO_ANALYSIS"))

    # Write known lemmas
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lemma", "upos"])
        for (lemma, upos), c in sorted(lemma_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
            examples = ", ".join(list(sorted(lemma_examples[(lemma, upos)]))[:5])
            w.writerow([lemma, upos])

    # Write failures for inspection
    with open(unknown_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "reason"])
        for term, reason in unknown_rows:
            w.writerow([term, reason])

    print(f"Read rows: {len(rows)}")
    print(f"Unique (lemma,upos): {len(lemma_counts)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote failures: {unknown_csv}")
    print(f"Detected term column: {term_col}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python lingq_to_known_lemmas.py lingqs.csv known_lemmas.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])