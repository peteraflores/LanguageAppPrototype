import csv
from collections import defaultdict
from datetime import datetime

IN_PATH = "needs_review_instances.csv"
OUT_PATH = "needs_review_summary.csv"

groups = {}

with open(IN_PATH, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        key = (
            row["surface_norm"],
            row["stanza_lemma"],
            row["stanza_upos"],
            row["udpipe_lemma"],
            row["udpipe_upos"],
            row["decision"],
        )
        if key not in groups:
            groups[key] = {
                "count": 0,
                "example": row["sentence"],
                "surface_orig": row["surface_orig"],
            }
        groups[key]["count"] += 1

rows = []
for key, meta in groups.items():
    surface_norm, sl, su, ul, uu, decision = key
    rows.append([
        meta["count"],
        meta["surface_orig"],
        surface_norm,
        sl, su,
        ul, uu,
        decision,
        meta["example"],
        "", "", ""  # approved_lemma, approved_upos, action
    ])

rows.sort(key=lambda x: (-int(x[0]), x[2], x[7]))

with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
    # ----- README HEADER BLOCK -----
    f.write("# ==============================================================\n")
    f.write("# NEEDS REVIEW SUMMARY\n")
    f.write("#\n")
    f.write("# How to use this file:\n")
    f.write("# 1. Review high-count rows first.\n")
    f.write("# 2. If neither tool is correct, manually enter the correct lemma.\n")
    f.write("# 3. Fill in:\n")
    f.write("#       approved_lemma\n")
    f.write("#       approved_upos\n")
    f.write("#       action = PROMOTE\n")
    f.write("# 4. Leave action blank to ignore.\n")
    f.write("# 5. Run promote_approved.py to add approved entries\n")
    f.write("#    into surface_lemma_lexicon.csv\n")
    f.write("#\n")
    f.write("# Notes:\n")
    f.write("# - surface_norm is what will be stored in the lexicon.\n")
    f.write("# - Counts represent number of occurrences seen.\n")
    f.write("# - You may override both tools if they are wrong.\n")
    f.write("# ==============================================================\n\n")

    w = csv.writer(f)
    w.writerow([
        "count",
        "surface_orig_example",
        "surface_norm",
        "stanza_lemma", "stanza_upos",
        "udpipe_lemma", "udpipe_upos",
        "decision",
        "example_sentence",
        "approved_lemma",
        "approved_upos",
        "action"
    ])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_PATH}")