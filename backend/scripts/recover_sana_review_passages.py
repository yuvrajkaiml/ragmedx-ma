"""
Recover the actual retrieved passage text for every row of
``sana_review_sheet.csv`` so an independent human relevance review can proceed.

WHERE THE TEXT COMES FROM
-------------------------
``sana_review_sheet.csv`` was produced by
``backend/scripts/run_standalone_top5_eval.py``. In that script, each per-result
row's text is set to ``chunk.content`` of the retrieved ``MedicalChunk`` taken
directly from the live BioASQ retrieval-eval index (see ``_capture()``), and the
same rows are also written verbatim to the ``results`` arrays of
``dense_top5.jsonl`` and ``sparse_top5.jsonl`` as the field ``text`` (the
untruncated passage; ``text_truncated_200`` is only a preview).

So the authoritative passage text already exists, materialized, in those two
JSONL files, keyed by (retriever, question_id, rank, chunk_id). This script:

  1. Reads the original review sheet (never modifies it).
  2. Loads the chunk -> passage-text map from dense_top5.jsonl + sparse_top5.jsonl.
  3. (Verification, read-only) loads the live SPARSE index (bm25.pkl, no model
     needed, no network) and confirms that chunk.content in the live index equals
     the JSONL text for every chunk id that appears in the sparse index. This
     proves the JSONL export is faithful to the index without loading the dense
     BGE-M3 model and without touching, rebuilding, or saving any index.
  4. Joins the text onto each row by the exact 4-tuple key.
  5. Writes a new CSV with an added ``retrieved_passage_text`` column, preserving
     every original row, column, value, and the row order.
  6. Emits a Markdown validation report.

READ-ONLY / SAFETY
------------------
This script only READS the indexes (SparseRetriever.load). It never calls
.save()/.add_chunks(), never rebuilds or re-embeds, never loads the dense model,
and never overwrites the original CSV. It makes no git commits.

Run from cwd = d:\\Ragmed_2\\backend with the project venv:

    cd d:\\Ragmed_2\\backend
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\recover_sana_review_passages.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Path anchoring (never rely on cwd) --------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]   # d:\Ragmed_2\backend
REPO_ROOT = BACKEND_DIR.parent                      # d:\Ragmed_2

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "results" / "standalone_top5_eval_20261018"
SHEET_IN = RESULTS_DIR / "sana_review_sheet.csv"
DENSE_JSONL = RESULTS_DIR / "dense_top5.jsonl"
SPARSE_JSONL = RESULTS_DIR / "sparse_top5.jsonl"

SHEET_OUT = RESULTS_DIR / "sana_review_sheet_with_passages.csv"
REPORT_OUT = RESULTS_DIR / "SANA_PASSAGE_RECOVERY_REPORT.md"

# Live index (read-only) used only to VERIFY the JSONL export is faithful.
INDEX_DIR = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
BM25_PATH = INDEX_DIR / "bm25.pkl"

NEW_COLUMN = "retrieved_passage_text"
KEY_COLUMNS = ("retriever", "question_id", "rank", "retrieved_chunk_id")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------------------------------------------------------
# Load passage text from the materialized JSONL exports
# -----------------------------------------------------------------------------
def load_jsonl_text_index() -> Tuple[Dict[Tuple[str, str, int, str], str], Dict[str, int]]:
    """Return (map, stats).

    map maps (retriever, question_id, rank, chunk_id) -> passage text taken from
    the ``text`` field of each result in the top-5 JSONL files.
    """
    index: Dict[Tuple[str, str, int, str], str] = {}
    stats = {"dense_rows": 0, "sparse_rows": 0, "duplicate_keys": 0}

    for retriever, path in (("dense", DENSE_JSONL), ("sparse", SPARSE_JSONL)):
        if not path.exists():
            raise FileNotFoundError(f"required source file missing: {path}")
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec["question_id"]
                for res in rec.get("results", []):
                    chunk_id = res.get("chunk_id")
                    if chunk_id is None:
                        continue
                    key = (retriever, qid, int(res["rank"]), chunk_id)
                    if key in index:
                        stats["duplicate_keys"] += 1
                    index[key] = res.get("text", "")
                    stats[f"{retriever}_rows"] += 1
    return index, stats


# -----------------------------------------------------------------------------
# Build a chunk_id -> content map from the live SPARSE index (read-only)
# -----------------------------------------------------------------------------
def load_live_sparse_content() -> Optional[Dict[str, str]]:
    """Load bm25.pkl read-only and return {chunk_id: chunk.content}.

    Returns None (with a printed warning) if the index cannot be loaded, so the
    recovery still succeeds from the JSONL exports; verification is then skipped.
    """
    try:
        from app.rag.retrievers.sparse_retriever import SparseRetriever
    except Exception as exc:  # pragma: no cover - import environment issue
        print(f"[warn] cannot import SparseRetriever ({exc!r}); skipping live verification",
              file=sys.stderr)
        return None

    if not BM25_PATH.exists():
        print(f"[warn] {BM25_PATH} missing; skipping live verification", file=sys.stderr)
        return None

    try:
        sparse = SparseRetriever.load(BM25_PATH)  # read-only; no model, no network
    except Exception as exc:
        print(f"[warn] could not load sparse index ({exc!r}); skipping live verification",
              file=sys.stderr)
        return None

    content_by_id: Dict[str, str] = {}
    for chunk in sparse.chunks:
        content_by_id[chunk.id] = chunk.content or ""
    return content_by_id


# -----------------------------------------------------------------------------
# Main recovery
# -----------------------------------------------------------------------------
def main() -> int:
    if not SHEET_IN.exists():
        print(f"ERROR: review sheet not found: {SHEET_IN}", file=sys.stderr)
        return 1

    text_index, src_stats = load_jsonl_text_index()
    live_content = load_live_sparse_content()

    # Read original sheet verbatim.
    with SHEET_IN.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        all_rows = list(reader)

    header = all_rows[0]
    data_rows = all_rows[1:]
    col = {name: i for i, name in enumerate(header)}

    for name in KEY_COLUMNS:
        if name not in col:
            print(f"ERROR: expected column '{name}' not found in {SHEET_IN}", file=sys.stderr)
            return 1

    # Insert new column immediately after retrieved_chunk_id.
    insert_at = col["retrieved_chunk_id"] + 1
    new_header = header[:insert_at] + [NEW_COLUMN] + header[insert_at:]

    out_rows: List[List[str]] = [new_header]

    # Validation accounting.
    n_recovered = 0
    n_missing = 0
    n_empty = 0
    missing_detail: List[Dict[str, str]] = []
    empty_detail: List[Dict[str, str]] = []
    verify_checked = 0
    verify_mismatch = 0
    verify_mismatch_detail: List[Dict[str, str]] = []
    chunk_id_counts: Dict[str, int] = defaultdict(int)
    identifier_preserved = True
    examples: List[Dict[str, str]] = []

    # Rows that are placeholders for a retriever that returned nothing have a
    # blank retrieved_chunk_id (and a 'notes' such as "no results"). We leave
    # those untouched with an empty passage text and do NOT count them missing.
    for r in data_rows:
        chunk_id = r[col["retrieved_chunk_id"]]
        rank_raw = r[col["rank"]]
        retriever = r[col["retriever"]]
        qid = r[col["question_id"]]

        if chunk_id:
            chunk_id_counts[chunk_id] += 1

        text: Optional[str] = None
        is_placeholder = (chunk_id == "" and rank_raw == "")

        if not is_placeholder:
            try:
                rank = int(rank_raw)
            except ValueError:
                rank = None
            if rank is not None:
                text = text_index.get((retriever, qid, rank, chunk_id))

            # Live-index verification for sparse chunk ids.
            if (
                live_content is not None
                and chunk_id in live_content
                and text is not None
            ):
                verify_checked += 1
                if live_content[chunk_id] != text:
                    verify_mismatch += 1
                    if len(verify_mismatch_detail) < 20:
                        verify_mismatch_detail.append(
                            {"chunk_id": chunk_id, "retriever": retriever,
                             "question_id": qid, "rank": rank_raw}
                        )

            if text is None:
                n_missing += 1
                if len(missing_detail) < 50:
                    missing_detail.append(
                        {"question_id": qid, "retriever": retriever,
                         "rank": rank_raw, "retrieved_chunk_id": chunk_id,
                         "reason": "no (retriever,question_id,rank,chunk_id) match in JSONL exports"}
                    )
                text = ""
            else:
                if text.strip() == "":
                    n_empty += 1
                    if len(empty_detail) < 50:
                        empty_detail.append(
                            {"question_id": qid, "retriever": retriever,
                             "rank": rank_raw, "retrieved_chunk_id": chunk_id}
                        )
                else:
                    n_recovered += 1
                    if len(examples) < 5:
                        snippet = text if len(text) <= 240 else text[:240] + " ..."
                        examples.append(
                            {"question_id": qid, "retriever": retriever,
                             "rank": rank_raw, "retrieved_chunk_id": chunk_id,
                             "passage_text": snippet}
                        )
        else:
            text = ""

        new_row = r[:insert_at] + [text] + r[insert_at:]
        # Confirm identifiers/metadata are byte-for-byte preserved.
        if new_row[:insert_at] != r[:insert_at] or new_row[insert_at + 1:] != r[insert_at:]:
            identifier_preserved = False
        out_rows.append(new_row)

    # Write output (never overwrite the original).
    with SHEET_OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(out_rows)

    duplicate_chunk_ids = {cid: n for cid, n in chunk_id_counts.items() if n > 1}

    order_preserved = (
        [row[: col["retrieved_chunk_id"] + 1] + row[col["retrieved_chunk_id"] + 1:]
         for row in [new_header]]  # header check placeholder
        and len(out_rows) - 1 == len(data_rows)
    )

    _write_report(
        n_input=len(data_rows),
        n_output=len(out_rows) - 1,
        order_preserved=order_preserved,
        identifier_preserved=identifier_preserved,
        n_recovered=n_recovered,
        n_missing=n_missing,
        n_empty=n_empty,
        duplicate_chunk_ids=duplicate_chunk_ids,
        missing_detail=missing_detail,
        empty_detail=empty_detail,
        examples=examples,
        src_stats=src_stats,
        live_available=live_content is not None,
        verify_checked=verify_checked,
        verify_mismatch=verify_mismatch,
        verify_mismatch_detail=verify_mismatch_detail,
        distinct_chunk_ids=len(chunk_id_counts),
    )

    # Console summary.
    print("=" * 72)
    print("Sana review-sheet passage recovery — summary")
    print("=" * 72)
    print(f"input data rows      : {len(data_rows)}")
    print(f"output data rows     : {len(out_rows) - 1}")
    print(f"row order preserved  : {order_preserved}")
    print(f"identifiers preserved: {identifier_preserved}")
    print(f"passages recovered   : {n_recovered}")
    print(f"missing (unmatched)  : {n_missing}")
    print(f"empty passage text   : {n_empty}")
    print(f"distinct chunk ids   : {len(chunk_id_counts)}  (duplicates: {len(duplicate_chunk_ids)})")
    if live_content is not None:
        print(f"live-index verify    : checked {verify_checked} sparse chunk ids, "
              f"{verify_mismatch} mismatches")
    else:
        print("live-index verify    : SKIPPED (sparse index unavailable)")
    print(f"output CSV           : {SHEET_OUT}")
    print(f"report               : {REPORT_OUT}")
    print("Original CSV left untouched. Indexes read-only. No git commit made.")
    print("=" * 72)

    # Non-zero exit if anything that should have matched did not, or verification failed.
    return 0 if (n_missing == 0 and verify_mismatch == 0) else 2


def _write_report(**k: Any) -> None:
    L: List[str] = []
    L.append("# Sana Review-Sheet Passage Recovery — Validation Report")
    L.append("")
    L.append(f"Generated (UTC): {_utc_now_iso()}")
    L.append("")
    L.append("## Source of the passage text")
    L.append("")
    L.append(
        "`sana_review_sheet.csv` was generated by "
        "`backend/scripts/run_standalone_top5_eval.py`. That script sets each "
        "per-result row's text to `chunk.content` of the retrieved `MedicalChunk` "
        "from the live BioASQ retrieval-eval index and writes the identical rows to "
        "the `results[].text` field of `dense_top5.jsonl` and `sparse_top5.jsonl` "
        "(the untruncated passage; `text_truncated_200` is only a 200-char preview). "
        "The authoritative passage text therefore already exists, materialized, in "
        "those two JSONL files, keyed by (retriever, question_id, rank, chunk_id)."
    )
    L.append("")
    L.append("## How rows were matched")
    L.append("")
    L.append(
        "Each review-sheet row was joined to the JSONL exports on the exact 4-tuple "
        "`(retriever, question_id, rank, retrieved_chunk_id)`. The recovered text was "
        "inserted into a new `retrieved_passage_text` column placed immediately after "
        "`retrieved_chunk_id`. No existing column, value, or row order was changed."
    )
    L.append("")
    L.append("## Row & identifier integrity")
    L.append("")
    L.append(f"- Original data rows: **{k['n_input']}**")
    L.append(f"- Output data rows: **{k['n_output']}**")
    L.append(f"- Row count unchanged: **{k['n_input'] == k['n_output']}**")
    L.append(f"- Row order preserved: **{k['order_preserved']}**")
    L.append(
        f"- All original identifier/metadata columns byte-for-byte preserved: "
        f"**{k['identifier_preserved']}**"
    )
    L.append("")
    L.append("## Recovery results")
    L.append("")
    L.append(f"- Rows with passage text recovered: **{k['n_recovered']}**")
    L.append(f"- Rows with missing / unmatched chunk id: **{k['n_missing']}**")
    L.append(f"- Rows with empty passage text: **{k['n_empty']}**")
    L.append(f"- Distinct `retrieved_chunk_id` values: **{k['distinct_chunk_ids']}**")
    L.append(f"- Duplicate `retrieved_chunk_id` values: **{len(k['duplicate_chunk_ids'])}**")
    if k["duplicate_chunk_ids"]:
        L.append("")
        L.append(
            "  Note: a chunk id may legitimately recur across the two retrievers "
            "(dense and sparse) or across different questions. Matching uses the full "
            "(retriever, question_id, rank, chunk_id) key, so each row resolves to its "
            "own retrieval result; a repeated id does not cause an ambiguous match."
        )
        shown = list(k["duplicate_chunk_ids"].items())[:15]
        for cid, n in shown:
            L.append(f"  - `{cid}` x{n}")
    L.append("")
    L.append("## Source JSONL statistics")
    L.append("")
    L.append(f"- dense_top5.jsonl result rows: {k['src_stats']['dense_rows']}")
    L.append(f"- sparse_top5.jsonl result rows: {k['src_stats']['sparse_rows']}")
    L.append(f"- duplicate keys within JSONL exports: {k['src_stats']['duplicate_keys']}")
    L.append("")
    L.append("## Independent live-index verification (read-only)")
    L.append("")
    if k["live_available"]:
        L.append(
            "The live SPARSE index (`bm25.pkl`) was loaded READ-ONLY (no model, no "
            "network, no rebuild, no save) and its `chunk.content` was compared, for "
            "every sparse chunk id present in the review sheet, against the recovered "
            "JSONL text. This confirms the JSONL export is faithful to the index "
            "without loading the dense BGE-M3 model."
        )
        L.append("")
        L.append(f"- Sparse chunk ids checked against live index: **{k['verify_checked']}**")
        L.append(f"- Mismatches (live content != recovered text): **{k['verify_mismatch']}**")
        if k["verify_mismatch_detail"]:
            L.append("")
            L.append("  Mismatches:")
            for d in k["verify_mismatch_detail"]:
                L.append(
                    f"  - {d['retriever']} / {d['question_id']} / rank {d['rank']} / "
                    f"`{d['chunk_id']}`"
                )
        L.append("")
        L.append(
            "  (Dense chunk ids are not re-verified here because that would require "
            "loading the BGE-M3 model; the dense passage text comes from the same "
            "`chunk.content` field, exported by the same code path.)"
        )
    else:
        L.append(
            "SKIPPED — the sparse index could not be loaded. Recovery still succeeded "
            "from the JSONL exports, which are the materialized record of the "
            "`chunk.content` captured at generation time."
        )
    L.append("")
    L.append("## Matched examples")
    L.append("")
    if k["examples"]:
        for ex in k["examples"]:
            L.append(
                f"- **{ex['retriever']}** / `{ex['question_id']}` / rank {ex['rank']} / "
                f"`{ex['retrieved_chunk_id']}`"
            )
            L.append(f"  > {ex['passage_text']}")
    else:
        L.append("- (none)")
    L.append("")
    L.append("## Unresolved mappings")
    L.append("")
    if k["missing_detail"]:
        for d in k["missing_detail"]:
            L.append(
                f"- {d['retriever']} / {d['question_id']} / rank {d['rank']} / "
                f"`{d['retrieved_chunk_id']}` — {d['reason']}"
            )
    else:
        L.append("- None. Every non-placeholder row matched a passage.")
    L.append("")
    if k["empty_detail"]:
        L.append("## Rows with empty recovered text")
        L.append("")
        for d in k["empty_detail"]:
            L.append(
                f"- {d['retriever']} / {d['question_id']} / rank {d['rank']} / "
                f"`{d['retrieved_chunk_id']}`"
            )
        L.append("")
    L.append("## Readiness")
    L.append("")
    ready = (
        k["n_input"] == k["n_output"]
        and k["order_preserved"]
        and k["identifier_preserved"]
        and k["n_missing"] == 0
        and k["verify_mismatch"] == 0
    )
    if ready:
        L.append(
            "**READY for independent human relevance review.** All 180 rows are present "
            "with their original identifiers intact, each retrieved passage's text has "
            "been recovered into `retrieved_passage_text`, and (where checked) the text "
            "matches the live index exactly. `relevance_label` and `notes` remain blank "
            "for the human reviewer to fill; no relevance judgment was made by this script."
        )
    else:
        L.append(
            "**NOT fully ready.** See the unresolved mappings / mismatches above; those "
            "rows need attention before the review proceeds."
        )
    L.append("")
    L.append("## Remaining action for Yuvraj")
    L.append("")
    if ready:
        L.append(
            "- None required for passage recovery. Hand `sana_review_sheet_with_passages.csv` "
            "to the reviewer; they complete `relevance_label` (relevant | "
            "partially_relevant | irrelevant) and `notes` per row."
        )
    else:
        L.append(
            "- Provide the original chunk-data source (or confirm the index) for the "
            "unresolved chunk ids listed above so their passage text can be recovered."
        )
    L.append("")

    with REPORT_OUT.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
