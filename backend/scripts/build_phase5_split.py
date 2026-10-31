"""Phase 5 — P5-2: build the 500+ question stratified/seeded eval split.

Pool
----
  * BioASQ eval split : 250 free-text questions with gold PMIDs/snippets
      (data/datasets/bioasq_eval_split/eval_questions.json)
  * MIRAGE PubMedQA   : 500 MCQ questions with gold PMID(s)
      (data/datasets/mirage/benchmark.json ["pubmedqa"])
  → combined pool = 750 >= 500

Each record is normalised to the common shape:
    {id, question, answer, type, source, pmid?}

Split
-----
``evaluation.split_builder.build_splits(dataset, dev_n=100, cal_n=150,
test_n=250, seed=42, stratify_by='source')`` → 100 dev / 150 calibration /
250 test (stratified by ``source``, seeded for reproducibility).

Leakage checks
--------------
  * ``check_near_duplicates`` across every split pair (threshold 0.9)
  * question/answer leakage vs the eval_split corpus chunk texts (threshold
    0.95). The eval_split corpus has ~58k chunks; computing the exact
    ``check_question_leakage`` nested-loop Jaccard 750×58k times is too slow,
    so this script computes the IDENTICAL token-Jaccard metric using a
    pre-tokenised corpus (same semantics, same threshold). This is documented
    in the manifest as the method used.
  * ``IndexLeakageGuard.assert_safe`` on the resolved eval_split path.

Measurability (honest)
----------------------
Retrieval-quality metrics (Recall@K / MRR / nDCG) are measurable ONLY for
questions whose gold PMIDs are present in the eval_split index. The eval_split
index was built from the BioASQ corpus; PubMedQA abstracts are NOT in it, so
PubMedQA retrieval-quality = NOT MEASURED (answer-quality only). This is
recorded in the manifest.

Output (under backend/results/phase5_<timestamp>/)
--------------------------------------------------
  dataset_manifest.json
  splits/dev.json
  splits/calibration.json
  splits/test.json

CPU only — no GPU, no model load. Use the CPU interpreter:
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe backend\\scripts\\build_phase5_split.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from evaluation.split_builder import (  # noqa: E402
    IndexLeakageGuard,
    build_splits,
    check_near_duplicates,
)

_REPO_ROOT = _BACKEND_DIR.parent
_BIOASQ_PATH = _BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
_PUBMEDQA_PATH = _BACKEND_DIR / "data" / "datasets" / "mirage" / "benchmark.json"
_EVAL_SPLIT_DIR = (_BACKEND_DIR / "data" / "faiss_index" / "eval_split").resolve()
_RESULTS_ROOT = _BACKEND_DIR / "results"


# ---------------------------------------------------------------------------
# Dataset normalisation
# ---------------------------------------------------------------------------

def _extract_pmid(url_or_val: Any) -> Optional[str]:
    s = str(url_or_val)
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else None


def load_bioasq_pool(path: Path) -> List[Dict[str, Any]]:
    """Normalise the 250 BioASQ eval questions to the common record shape."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = raw.get("questions", list(raw.values())[0])
    pool: List[Dict[str, Any]] = []
    for item in raw:
        answer = item.get("ideal_answer") or item.get("exact_answer") or ""
        if isinstance(answer, list):
            # ideal_answer is a list of strings; exact_answer may be nested lists.
            flat: List[str] = []
            for a in answer:
                if isinstance(a, list):
                    flat.extend(str(x) for x in a)
                else:
                    flat.append(str(a))
            answer = " ".join(flat)
        pmids = [p for p in (_extract_pmid(d) for d in item.get("documents", [])) if p]
        pool.append({
            "id": f"bioasq::{item.get('id', '')}",
            "question": item.get("body", ""),
            "answer": answer,
            "type": item.get("type", ""),
            "source": "bioasq",
            "pmid": pmids,
        })
    return pool


def load_pubmedqa_pool(path: Path) -> List[Dict[str, Any]]:
    """Normalise the 500 MIRAGE PubMedQA questions to the common record shape."""
    with open(path, encoding="utf-8") as fh:
        bench = json.load(fh)
    pq = bench["pubmedqa"]
    pool: List[Dict[str, Any]] = []
    for key, item in pq.items():
        options = item.get("options", {})
        answer_letter = item.get("answer", "")
        answer_text = options.get(answer_letter, answer_letter)
        pmids = [str(p) for p in item.get("PMID", []) if str(p)]
        pool.append({
            "id": f"pubmedqa::{key}",
            "question": item.get("question", ""),
            "answer": answer_text,
            "answer_letter": answer_letter,
            "options": options,
            "type": "yesno",
            "source": "pubmedqa",
            "pmid": pmids,
        })
    return pool


# ---------------------------------------------------------------------------
# Manifest hash (mirrors Phase 1 _compute_dataset_manifest_hash)
# ---------------------------------------------------------------------------

def compute_manifest_hash(records: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for item in records:
        h.update(str(item.get("question", "")).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Fast corpus-leakage check (same token-Jaccard semantics as
# split_builder.check_question_leakage, pre-tokenised for performance)
# ---------------------------------------------------------------------------

def _tokens(text: str) -> frozenset:
    return frozenset(re.findall(r"\w+", text.lower()))


def leakage_vs_corpus(
    records: List[Dict[str, Any]],
    corpus_token_sets: List[frozenset],
    field: str,
    threshold: float = 0.95,
) -> List[str]:
    """Return record IDs whose <field> token-Jaccard >= threshold vs any chunk.

    Identical metric to split_builder._token_jaccard / check_question_leakage,
    but the corpus is tokenised once up front.
    """
    leaking: List[str] = []
    for rec in records:
        q_tokens = _tokens(str(rec.get(field, "")))
        if not q_tokens:
            continue
        hit = False
        for c_tokens in corpus_token_sets:
            if not c_tokens:
                continue
            inter = len(q_tokens & c_tokens)
            if inter == 0:
                continue
            union = len(q_tokens | c_tokens)
            if union and (inter / union) >= threshold:
                hit = True
                break
        if hit:
            leaking.append(rec["id"])
    return leaking


def load_corpus_token_sets() -> Tuple[List[frozenset], Dict[str, Any]]:
    """Load eval_split chunk texts and return their token sets + metadata."""
    from app.rag.vectorstore.faiss_store import FAISSVectorStore

    store = FAISSVectorStore.load(_EVAL_SPLIT_DIR)
    token_sets = [_tokens(getattr(c, "content", "") or "") for c in store.chunks]
    meta = {"num_chunks": len(store.chunks)}
    return token_sets, meta


# ---------------------------------------------------------------------------
# Atomic JSON write
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 5 stratified eval split.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Phase 5 output dir (default: backend/results/phase5_<timestamp>/).")
    parser.add_argument("--skip-corpus-leakage", action="store_true",
                        help="Skip the (slow) eval_split corpus leakage scan.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_root or (_RESULTS_ROOT / f"phase5_{timestamp}")
    out_root = Path(out_root)
    splits_dir = out_root / "splits"

    print(f"[P5-2] Output dir: {out_root}")

    # 1. Guard the eval_split path (fail loud if it is the production index).
    guard = IndexLeakageGuard(repo_root=str(_REPO_ROOT))
    guard.assert_safe(str(_EVAL_SPLIT_DIR))
    print(f"[P5-2] IndexLeakageGuard.assert_safe PASS for {_EVAL_SPLIT_DIR}")

    # 2. Build the 750-question pool.
    bioasq = load_bioasq_pool(_BIOASQ_PATH)
    pubmedqa = load_pubmedqa_pool(_PUBMEDQA_PATH)
    pool = bioasq + pubmedqa
    print(f"[P5-2] Pool: BioASQ={len(bioasq)} + PubMedQA={len(pubmedqa)} = {len(pool)}")
    if len(pool) < 500:
        raise SystemExit(f"[P5-2] pool has only {len(pool)} questions (<500). Aborting.")

    # 3. Stratified seeded split.
    dev, cal, test = build_splits(
        pool, dev_n=100, cal_n=150, test_n=250, seed=42, stratify_by="source"
    )
    print(f"[P5-2] Split: dev={len(dev)} cal={len(cal)} test={len(test)}")

    def _source_counts(recs: List[Dict[str, Any]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in recs:
            out[r["source"]] = out.get(r["source"], 0) + 1
        return out

    # 4. Near-duplicate checks across split pairs (threshold 0.9).
    near_dup = {
        "dev_vs_cal": check_near_duplicates(dev, cal, threshold=0.9),
        "dev_vs_test": check_near_duplicates(dev, test, threshold=0.9),
        "cal_vs_test": check_near_duplicates(cal, test, threshold=0.9),
    }
    near_dup_counts = {k: len(v) for k, v in near_dup.items()}
    print(f"[P5-2] Near-duplicate counts (threshold 0.9): {near_dup_counts}")

    # 5. Corpus leakage (question + answer vs eval_split chunk texts, 0.95).
    corpus_meta: Dict[str, Any] = {}
    question_leak: List[str] = []
    answer_leak: List[str] = []
    corpus_leakage_method = "token_jaccard_prefused_threshold_0.95"
    if args.skip_corpus_leakage:
        corpus_leakage_method = "SKIPPED (--skip-corpus-leakage)"
        print("[P5-2] Corpus leakage scan SKIPPED by flag.")
    else:
        print("[P5-2] Loading eval_split corpus chunk texts for leakage scan …")
        t0 = time.perf_counter()
        corpus_token_sets, corpus_meta = load_corpus_token_sets()
        print(f"[P5-2] Loaded {corpus_meta['num_chunks']} chunks "
              f"in {time.perf_counter() - t0:.1f}s. Scanning …")
        t1 = time.perf_counter()
        question_leak = leakage_vs_corpus(pool, corpus_token_sets, "question", 0.95)
        answer_leak = leakage_vs_corpus(pool, corpus_token_sets, "answer", 0.95)
        print(f"[P5-2] Corpus leakage scan done in {time.perf_counter() - t1:.1f}s: "
              f"question_leak={len(question_leak)} answer_leak={len(answer_leak)}")

    # 6. Retrieval-quality measurability (honest).
    pubmedqa_with_pmid = sum(1 for r in pubmedqa if r.get("pmid"))
    bioasq_with_pmid = sum(1 for r in bioasq if r.get("pmid"))
    measurability = {
        "bioasq": {
            "retrieval_quality": "MEASURABLE (gold PMIDs; eval_split built from BioASQ corpus)",
            "questions_with_gold_pmid": bioasq_with_pmid,
            "note": "Recall@K/MRR/nDCG measurable only where the gold PMID is present in eval_split.",
        },
        "pubmedqa": {
            "retrieval_quality": "NOT MEASURED",
            "reason": "PubMedQA abstracts are not in the BioASQ-built eval_split index; "
                      "gold PMIDs are not retrievable from this index. Report answer-quality only.",
            "questions_with_gold_pmid_in_dataset": pubmedqa_with_pmid,
        },
    }

    # 7. Write splits.
    _atomic_write_json(splits_dir / "dev.json", dev)
    _atomic_write_json(splits_dir / "calibration.json", cal)
    _atomic_write_json(splits_dir / "test.json", test)
    print(f"[P5-2] Wrote splits to {splits_dir}")

    # 8. Write manifest.
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": 42,
        "stratify_by": "source",
        "evidence_tier": "SOFTWARE TEST",
        "pool": {
            "total": len(pool),
            "bioasq": len(bioasq),
            "pubmedqa": len(pubmedqa),
        },
        "record_shape": ["id", "question", "answer", "type", "source", "pmid?"],
        "splits": {
            "dev": {"n": len(dev), "source_counts": _source_counts(dev)},
            "calibration": {"n": len(cal), "source_counts": _source_counts(cal)},
            "test": {"n": len(test), "source_counts": _source_counts(test)},
        },
        "manifest_hash": {
            "pool": compute_manifest_hash(pool),
            "dev": compute_manifest_hash(dev),
            "calibration": compute_manifest_hash(cal),
            "test": compute_manifest_hash(test),
        },
        "leakage_checks": {
            "near_duplicates_threshold": 0.9,
            "near_duplicate_counts": near_dup_counts,
            "near_duplicate_pairs": near_dup,
            "corpus_leakage_threshold": 0.95,
            "corpus_leakage_method": corpus_leakage_method,
            "corpus_meta": corpus_meta,
            "question_leakage_ids": question_leak,
            "answer_leakage_ids": answer_leak,
            "index_leakage_guard": "PASS",
            "resolved_eval_split_dir": str(_EVAL_SPLIT_DIR),
        },
        "retrieval_quality_measurability": measurability,
    }
    _atomic_write_json(out_root / "dataset_manifest.json", manifest)
    print(f"[P5-2] Wrote manifest to {out_root / 'dataset_manifest.json'}")
    print(f"[P5-2] DONE. Phase 5 output root: {out_root}")
    # Emit the output root on the last line so callers can capture it.
    print(f"PHASE5_OUTPUT_ROOT={out_root}")


if __name__ == "__main__":
    main()
