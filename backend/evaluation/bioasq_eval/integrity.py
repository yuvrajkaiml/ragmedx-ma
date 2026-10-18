"""
Integrity safeguards for the BioASQ retrieval-eval + calibration run.

Recomputes SHA-256 of the four protected files (production index.faiss/chunks.pkl
and eval_split index.faiss/chunks.pkl) and compares them to the frozen baseline in
``results/bioasq_eval_20261018/integrity_baseline.json``. Any mismatch is fatal and
marks the run INVALID. Also re-asserts eval_split membership (distinct-PMID set +
chunk count) is unchanged.

This module NEVER writes to any protected directory. It only reads the protected
files (to hash them) and writes its verdict under the deliverables root.

Run standalone from ``backend/``::

    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m evaluation.bioasq_eval.integrity --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

BACKEND_DIR = Path(__file__).resolve().parents[2]   # d:\Ragmed_2\backend
REPO_ROOT = Path(__file__).resolve().parents[3]     # d:\Ragmed_2

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
BASELINE_PATH = RESULTS_DIR / "integrity_baseline.json"
INTEGRITY_CHECK_PATH = RESULTS_DIR / "integrity_check.json"

# Protected directories: writes here are forbidden.
PRODUCTION_INDEX_DIR = BACKEND_DIR / "data" / "faiss_index"
EVAL_SPLIT_DIR = PRODUCTION_INDEX_DIR / "eval_split"
EVAL_SPLIT_DATASET_DIR = BACKEND_DIR / "data" / "datasets" / "bioasq_eval_split"

# Expected eval_split membership (verified during design; re-asserted at run time).
EXPECTED_EVAL_SPLIT_CHUNKS = 58173


def compute_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file read in binary mode."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_baseline(baseline_path: str | Path = BASELINE_PATH) -> Dict[str, str]:
    """Load the frozen {absolute_path: sha256} baseline."""
    with open(baseline_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _recompute(baseline: Dict[str, str]) -> Dict[str, Dict[str, object]]:
    """Recompute SHA-256 of every baselined file; return per-file verdicts."""
    verdicts: Dict[str, Dict[str, object]] = {}
    for path, expected in baseline.items():
        p = Path(path)
        if not p.exists():
            verdicts[path] = {"exists": False, "expected": expected,
                              "actual": None, "match": False}
            continue
        actual = compute_sha256(p)
        verdicts[path] = {"exists": True, "expected": expected,
                          "actual": actual, "match": actual == expected}
    return verdicts


def _check(baseline: Dict[str, str]) -> Dict[str, object]:
    verdicts = _recompute(baseline)
    ok = all(v["match"] for v in verdicts.values())
    return {"ok": ok, "files": verdicts}


def check_pre_run(baseline_path: str | Path = BASELINE_PATH) -> Dict[str, object]:
    """Pre-run integrity check: all four protected files must match baseline."""
    return _check(load_baseline(baseline_path))


def check_post_run(baseline_path: str | Path = BASELINE_PATH) -> Dict[str, object]:
    """Post-run integrity check: identical rule to pre-run."""
    return _check(load_baseline(baseline_path))


def check_eval_split_membership() -> Dict[str, object]:
    """Re-assert eval_split distinct-PMID set + chunk count are unchanged.

    Opens the eval_split index read-only via FAISSVectorStore.load (never writes).
    """
    from app.rag.vectorstore.faiss_store import FAISSVectorStore

    store = FAISSVectorStore.load(EVAL_SPLIT_DIR)
    n_chunks = len(store.chunks)
    distinct_pmids = set()
    missing_pmid = 0
    for ch in store.chunks:
        pmid = (ch.metadata or {}).get("pmid")
        if pmid is None:
            missing_pmid += 1
            continue
        distinct_pmids.add(str(pmid))
    return {
        "n_chunks": n_chunks,
        "expected_chunks": EXPECTED_EVAL_SPLIT_CHUNKS,
        "chunks_unchanged": n_chunks == EXPECTED_EVAL_SPLIT_CHUNKS,
        "distinct_pmids": len(distinct_pmids),
        "chunks_missing_pmid": missing_pmid,
    }


def run_and_write(phase: str, baseline_path: str | Path = BASELINE_PATH,
                  check_membership: bool = False) -> Dict[str, object]:
    """Run the appropriate check, merge with any prior verdict, write the file.

    ``phase`` is one of ``"pre_run"`` / ``"post_run"``. Raises ``RuntimeError`` on
    any mismatch so the caller can mark the run INVALID.
    """
    verdict = _check(load_baseline(baseline_path))
    membership = check_eval_split_membership() if check_membership else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, object] = {}
    if INTEGRITY_CHECK_PATH.exists():
        try:
            with open(INTEGRITY_CHECK_PATH, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = {}

    existing[f"{phase}_ok"] = verdict["ok"]
    existing[f"{phase}_files"] = verdict["files"]
    if membership is not None:
        existing[f"{phase}_eval_split_membership"] = membership

    tmp = INTEGRITY_CHECK_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    import os
    os.replace(tmp, INTEGRITY_CHECK_PATH)

    membership_ok = membership is None or (
        membership["chunks_unchanged"] and membership["chunks_missing_pmid"] == 0
    )
    if not verdict["ok"] or not membership_ok:
        raise RuntimeError(
            f"INTEGRITY {phase} FAILED — protected file checksum or eval_split "
            f"membership changed. Run is INVALID. See {INTEGRITY_CHECK_PATH}."
        )
    return existing


def _main() -> int:
    parser = argparse.ArgumentParser(description="BioASQ integrity check")
    parser.add_argument("--check-only", action="store_true",
                        help="Recompute hashes and print verdict without raising.")
    parser.add_argument("--phase", choices=["pre_run", "post_run"], default="pre_run")
    parser.add_argument("--membership", action="store_true",
                        help="Also check eval_split membership.")
    args = parser.parse_args()

    verdict = _check(load_baseline())
    print(f"pre_run_ok: {str(verdict['ok']).lower()}")
    for path, v in verdict["files"].items():
        print(f"  {'OK ' if v['match'] else 'BAD'} {path}")

    if args.check_only:
        return 0 if verdict["ok"] else 1

    try:
        run_and_write(args.phase, check_membership=args.membership)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
