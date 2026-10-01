"""
build_eval_split.py — Leakage-safe held-out evaluation split for RAGMedX-MA.

Splits backend/data/datasets/bioasq/training13b.json into:
  - 250 held-out eval questions  (eval_questions.json)
  - train snippet corpus, with cross-question PMID exclusion applied
    (train_corpus.json, ready for build_index.py --corpus-file)

Leakage prevention
------------------
For each eval question, BioASQ's curators listed every document they
considered relevant in the question["documents"] field (full PubMed URLs,
same format as snippet["document"]).  Excluding only the eval questions'
own snippets is insufficient: a TRAIN question may also cite one of those
same documents, putting the gold-answer passage in the index under a
different question ID.  The correct exclusion set is therefore:

    excluded_pmid_urls = ⋃  question["documents"]
                         for every question in the eval set

Then we remove from the TRAIN snippets every snippet whose snippet["document"]
URL appears in that set — regardless of which train question it came from.

Output files
------------
  backend/data/datasets/bioasq_eval_split/
    train_corpus.json   — list of snippet dicts (text, document, beginSection,
                          endSection, offsetInBeginSection, offsetInEndSection,
                          question_id, source_file) ready for ingestion
    eval_questions.json — list of full question objects (body, ideal_answer,
                          documents, snippets, type, id, concepts)

Usage
-----
    python -m scripts.build_eval_split            # default source file
    python -m scripts.build_eval_split --src path/to/training13b.json
    python -m scripts.build_eval_split --eval-size 500 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_DEFAULT_SRC = (
    _BACKEND_DIR / "data" / "datasets" / "bioasq" / "training13b.json"
)
_DEFAULT_OUT = _BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split"
_DEFAULT_EVAL_SIZE = 250
_DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_questions(src: Path) -> list[dict]:
    with open(src, "r", encoding="utf-8") as fh:
        return json.load(fh)["questions"]


def _collect_excluded_pmid_urls(eval_questions: list[dict]) -> frozenset[str]:
    """Return every PubMed URL referenced in any eval question's documents list."""
    excluded: set[str] = set()
    for q in eval_questions:
        for url in q.get("documents", []):
            url = url.strip()
            if url:
                excluded.add(url)
    return frozenset(excluded)


def _build_train_corpus(
    train_questions: list[dict],
    excluded_urls: frozenset[str],
    src_filename: str,
) -> tuple[list[dict], int, int]:
    """
    Collect all snippets from train questions, then remove any whose
    snippet["document"] URL appears in excluded_urls.

    Returns
    -------
    (kept_snippets, total_before_exclusion, cross_question_extra_removed)
    """
    all_snippets: list[dict] = []
    for q in train_questions:
        q_id = q.get("id", "")
        for snippet in q.get("snippets", []):
            text = snippet.get("text", "").strip()
            if not text:
                continue
            all_snippets.append({
                "text":                 text,
                "document":             snippet.get("document", ""),
                "beginSection":         snippet.get("beginSection", "abstract"),
                "endSection":           snippet.get("endSection", "abstract"),
                "offsetInBeginSection": snippet.get("offsetInBeginSection", 0),
                "offsetInEndSection":   snippet.get("offsetInEndSection", 0),
                # Provenance metadata for traceability
                "question_id":          q_id,
                "source_file":          src_filename,
            })

    total_before = len(all_snippets)

    # Snippets that belong to train questions but cite an excluded document
    # (cross-question contamination — this is the extra leakage the task
    # description specifically calls out).
    kept: list[dict] = []
    removed_cross = 0
    for s in all_snippets:
        doc_url = s.get("document", "").strip()
        if doc_url in excluded_urls:
            removed_cross += 1
        else:
            kept.append(s)

    # The eval questions' own snippets aren't in all_snippets (they come from
    # eval_questions, which were split out before this function is called), so
    # removed_cross counts *only* the cross-question extras that would have
    # leaked from train questions.
    return kept, total_before, removed_cross


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build leakage-safe eval split from BioASQ training data."
    )
    ap.add_argument(
        "--src",
        type=Path,
        default=_DEFAULT_SRC,
        help="Path to training13b.json (default: %(default)s)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory (default: %(default)s)",
    )
    ap.add_argument(
        "--eval-size",
        type=int,
        default=_DEFAULT_EVAL_SIZE,
        help="Number of questions to hold out for eval (default: %(default)s)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help="Random seed for reproducible shuffle (default: %(default)s)",
    )
    args = ap.parse_args()

    src: Path = args.src
    out_dir: Path = args.out_dir
    eval_size: int = args.eval_size
    seed: int = args.seed

    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        sys.exit(1)

    print("=" * 64)
    print("RAGMedX-MA — BioASQ leakage-safe eval split")
    print(f"  src       : {src}")
    print(f"  out_dir   : {out_dir}")
    print(f"  eval_size : {eval_size}")
    print(f"  seed      : {seed}")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # 1. Load and shuffle
    # ------------------------------------------------------------------ #
    print("\nStep 1/4 — loading and shuffling questions …")
    questions = _load_questions(src)
    total_questions = len(questions)
    print(f"  {total_questions} questions loaded from {src.name}")

    rng = random.Random(seed)
    shuffled = list(questions)       # copy, don't mutate the original list
    rng.shuffle(shuffled)

    # ------------------------------------------------------------------ #
    # 2. Split
    # ------------------------------------------------------------------ #
    print(f"\nStep 2/4 — splitting off {eval_size} eval questions …")
    eval_questions  = shuffled[:eval_size]
    train_questions = shuffled[eval_size:]
    print(f"  eval  : {len(eval_questions)} questions")
    print(f"  train : {len(train_questions)} questions")

    # ------------------------------------------------------------------ #
    # 3. Build excluded PMID URL set from eval documents field
    # ------------------------------------------------------------------ #
    print("\nStep 3/4 — collecting excluded document URLs from eval set …")
    excluded_urls = _collect_excluded_pmid_urls(eval_questions)
    print(f"  {len(excluded_urls)} unique document URLs excluded "
          f"(from eval questions' 'documents' field)")

    # ------------------------------------------------------------------ #
    # 4. Filter train snippets
    # ------------------------------------------------------------------ #
    print("\nStep 4/4 — building train corpus with cross-question exclusion …")
    kept_snippets, total_before, cross_removed = _build_train_corpus(
        train_questions, excluded_urls, src.name
    )

    eval_own_snippets = sum(
        len(q.get("snippets", [])) for q in eval_questions
    )
    # Snippets removed by cross-question exclusion = snippets from train
    # questions that cite a document referenced by any eval question.
    # (eval questions' own snippets aren't in all_snippets at all, so they
    # don't appear in total_before.)
    print(f"\n  Train snippets (all, before exclusion) : {total_before:>7,}")
    print(f"  Eval questions' own snippets (not in   : {eval_own_snippets:>7,}")
    print(f"    train corpus, shown for reference)   :")
    print(f"  Removed by cross-question PMID exclusion: {cross_removed:>6,}")
    print(f"  Train snippets kept                    : {len(kept_snippets):>7,}")

    # ------------------------------------------------------------------ #
    # 5. Write output files
    # ------------------------------------------------------------------ #
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_corpus.json"
    eval_path  = out_dir / "eval_questions.json"

    # train_corpus.json: list of snippet dicts
    with open(train_path, "w", encoding="utf-8") as fh:
        json.dump(kept_snippets, fh, ensure_ascii=False, indent=2)

    # eval_questions.json: full question objects (body, ideal_answer, documents,
    # snippets, type, id, concepts) for use as ground truth in benchmarking
    eval_out = [
        {k: v for k, v in q.items()}   # full object, no fields stripped
        for q in eval_questions
    ]
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(eval_out, fh, ensure_ascii=False, indent=2)

    print(f"\nOutput written:")
    print(f"  {train_path}  "
          f"({train_path.stat().st_size / 1_048_576:.1f} MB, "
          f"{len(kept_snippets):,} snippets)")
    print(f"  {eval_path}  "
          f"({eval_path.stat().st_size / 1_048_576:.1f} MB, "
          f"{len(eval_out):,} questions)")

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Total questions in source         : {total_questions:>7,}")
    print(f"  Held-out eval questions           : {len(eval_questions):>7,}")
    print(f"  Train questions                   : {len(train_questions):>7,}")
    print(f"  Excluded document URLs (eval docs): {len(excluded_urls):>7,}")
    print(f"  Train snippets before exclusion   : {total_before:>7,}")
    print(f"  Removed (cross-question, PMID)    : {cross_removed:>7,}")
    print(f"  Train snippets after exclusion    : {len(kept_snippets):>7,}")
    pct = cross_removed / total_before * 100 if total_before else 0
    print(f"  Cross-question removal rate       : {pct:>6.1f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
