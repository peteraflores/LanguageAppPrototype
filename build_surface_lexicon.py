import csv
import re

files = [
    "el_gdt-ud-train.conllu",
    "el_gdt-ud-dev.conllu",
    "el_gdt-ud-test.conllu",
]

entries = set()

digit_re = re.compile(r"\d")

# Matches acronym/abbrev-like tokens such as:
# CNN, BP., U.S., ΕΕ., ΝΔ., etc.
# (2+ letters, may contain dots, no lowercase)
acronym_re = re.compile(r"^(?:[A-ZΑ-Ω]{2,}|(?:[A-ZΑ-Ω]\.){2,}|[A-ZΑ-Ω]{2,}\.)+$")

skip_upos = {"PUNCT", "NUM", "SYM", "X"}

for filename in files:
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 6:
                continue

            token_id = cols[0]
            if "-" in token_id or "." in token_id:
                continue

            surface = cols[1]
            lemma = cols[2]
            upos = cols[3]
            feats = cols[5]  # FEATS column

            if upos in skip_upos:
                continue
            if surface == "_" or lemma == "_":
                continue
            if digit_re.search(surface) or digit_re.search(lemma):
                continue

            # Drop abbreviations explicitly marked by UD
            if feats != "_" and "Abbr=Yes" in feats:
                continue

            # Drop acronym-ish tokens (CNN, BP., ΕΕ., etc.)
            # Check *original* surface before lowercasing
            if acronym_re.match(surface):
                continue

            # Normalize to lowercase for lookup use
            surface_l = surface.lower()
            lemma_l = lemma.lower()

            entries.add((surface_l, lemma_l, upos))

entries = sorted(entries)

with open("surface_lemma_lexicon.csv", "w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["surface", "lemma", "upos"])
    writer.writerows(entries)

print(f"Wrote {len(entries)} unique mappings to surface_lemma_lexicon.csv")