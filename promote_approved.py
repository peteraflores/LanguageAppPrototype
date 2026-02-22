import csv
import os

SUMMARY = "needs_review_summary.csv"
INSTANCES = "needs_review_instances.csv"
LEXICON = "surface_lemma_lexicon.csv"

def read_non_comment_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            yield line

def load_existing_lexicon(path: str) -> set[tuple[str, str, str]]:
    existing = set()
    if not os.path.exists(path):
        return existing
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            existing.add((
                row["surface"].strip(),
                row["lemma"].strip(),
                row["upos"].strip().upper(),
            ))
    return existing

def append_new_lexicon_entries(path: str, triples: list[tuple[str, str, str]]):
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["surface", "lemma", "upos"])
        w.writerows(triples)

def atomic_write(path: str, lines_iterable):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        for line in lines_iterable:
            f.write(line)
    os.replace(tmp, path)

# ----------------------------
# 1) Read summary and collect promotions
# ----------------------------

summary_lines = list(open(SUMMARY, "r", encoding="utf-8").readlines())

# Keep README/comment lines to preserve them on rewrite
summary_comment_lines = [ln for ln in summary_lines if ln.startswith("#") or not ln.strip()]
summary_data_lines = [ln for ln in summary_lines if not (ln.startswith("#") or not ln.strip())]

summary_reader = csv.DictReader(summary_data_lines)

promote_triples = set()    # (surface_norm, approved_lemma, approved_upos)
promoted_signatures = set()  # signature of the summary row to remove instances too

kept_summary_rows = []  # rows we keep (non-promoted)

for row in summary_reader:
    action = (row.get("action") or "").strip().upper()
    if action == "PROMOTE":
        s = row["surface_norm"].strip()
        l = row["approved_lemma"].strip()
        u = row["approved_upos"].strip().upper()
        if s and l and u:
            promote_triples.add((s, l, u))

        # signature to delete from instances + summary
        promoted_signatures.add((
            row["surface_norm"].strip(),
            row["stanza_lemma"].strip(),
            row["stanza_upos"].strip().upper(),
            row["udpipe_lemma"].strip(),
            row["udpipe_upos"].strip().upper(),
            row["decision"].strip(),
        ))
        continue

    kept_summary_rows.append(row)

# ----------------------------
# 2) Append only new entries to lexicon (dedupe even if user marked PROMOTE)
# ----------------------------

existing = load_existing_lexicon(LEXICON)
to_add = sorted([t for t in promote_triples if t not in existing])

if to_add:
    append_new_lexicon_entries(LEXICON, to_add)

# ----------------------------
# 3) Rewrite needs_review_summary.csv WITHOUT promoted rows (preserve README)
# ----------------------------

fieldnames = [
    "count",
    "surface_orig_example",
    "surface_norm",
    "stanza_lemma", "stanza_upos",
    "udpipe_lemma", "udpipe_upos",
    "decision",
    "example_sentence",
    "approved_lemma",
    "approved_upos",
    "action",
]

def summary_out_lines():
    # preserve the README/comments exactly as they were
    for ln in summary_comment_lines:
        yield ln

    # ensure exactly one blank line between README and header if you want:
    if summary_comment_lines and not summary_comment_lines[-1].endswith("\n"):
        yield "\n"

    # write CSV header + rows
    from io import StringIO
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    yield buf.getvalue()
    buf.close()

    for r in kept_summary_rows:
        buf = StringIO()
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        # DictWriter writes header unless we avoid it; so write only row:
        w.writerow({k: r.get(k, "") for k in fieldnames})
        yield buf.getvalue()
        buf.close()

atomic_write(SUMMARY, summary_out_lines())

# ----------------------------
# 4) Rewrite needs_review_instances.csv removing instances matching promoted signatures
# ----------------------------

if os.path.exists(INSTANCES) and promoted_signatures:
    with open(INSTANCES, "r", encoding="utf-8") as f:
        inst_reader = csv.DictReader(f)
        inst_rows_kept = []
        inst_fieldnames = inst_reader.fieldnames

        for row in inst_reader:
            sig = (
                row["surface_norm"].strip(),
                row["stanza_lemma"].strip(),
                row["stanza_upos"].strip().upper(),
                row["udpipe_lemma"].strip(),
                row["udpipe_upos"].strip().upper(),
                row["decision"].strip(),
            )
            if sig in promoted_signatures:
                continue
            inst_rows_kept.append(row)

    tmp = INSTANCES + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=inst_fieldnames)
        w.writeheader()
        w.writerows(inst_rows_kept)
    os.replace(tmp, INSTANCES)

print(f"Approved promotions requested: {len(promote_triples)}")
print(f"Actually appended to lexicon (new only): {len(to_add)}")
print(f"Removed {len(promoted_signatures)} promoted rows from summary and matching instances.")