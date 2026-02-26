import csv
import re
import sys
import os
from collections import defaultdict

from lemmatizer import Lemmatizer, normalize_text

GREEK_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]")

# Token splitter for LingQ "terms" (keeps Greek tokens; splits on whitespace and common punctuation)
SPLIT_RE = re.compile(r"[^\wΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώϊΐϋΰ]+")

SKIP_UPOS = {"PUNCT"}  # for term lists, be much less aggressive than your passage filters


def norm(s: str) -> str:
    # Use lemmatizer's normalize function without lowercase
    return normalize_text(s).upper() if s else ""  # Preserve case by re-uppercasing


def norm_lower(s: str) -> str:
    return normalize_text(s)


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
    # Initialize lemmatizer instead of creating our own Stanza pipeline
    udpipe_model_path = "greek-gdt-ud-2.5-191206.udpipe"
    stanza_model_dir = os.path.join(os.getcwd(), "stanza_resources")
    surface_lexicon_path = "surface_lemma_lexicon.csv"
    
    lemmatizer = Lemmatizer(
        surface_lexicon_path=surface_lexicon_path,
        udpipe_model_path=udpipe_model_path,
        stanza_model_dir=stanza_model_dir,
        use_lexicon=True,
        skip_propn=False  # Don't skip proper nouns for LingQ terms
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

        # Join tokens and use lemmatizer
        term_text = " ".join(toks)
        
        try:
            results = lemmatizer.lemmatize_passage(term_text)
            any_ok = False
            
            for surface_orig, lemma, upos, source in results:
                if upos in SKIP_UPOS:
                    continue
                    
                if not lemma:
                    continue
                    
                any_ok = True
                lemma_counts[(lemma, upos)] += 1
                lemma_examples[(lemma, upos)].add(surface_orig)
        except Exception:
            # If lemmatization fails, mark as unknown
            any_ok = False

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