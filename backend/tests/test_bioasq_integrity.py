"""Protected-index + eval-split integrity tests (read-only; never writes protected dirs)."""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.bioasq_eval import integrity  # noqa: E402
from evaluation.split_builder import IndexLeakageError, IndexLeakageGuard  # noqa: E402

PROD_DIR = BACKEND_DIR / "data" / "faiss_index"
EVAL_SPLIT_DIR = PROD_DIR / "eval_split"
ISOLATED_INDEX = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"


def test_production_and_eval_split_checksums_match_baseline():
    verdict = integrity.check_pre_run()
    assert verdict["ok"], f"checksum mismatch: {verdict['files']}"
    for path, v in verdict["files"].items():
        assert v["exists"], f"missing protected file {path}"
        assert v["match"], f"checksum changed for {path}"


@pytest.mark.integration
def test_eval_split_membership_unchanged():
    membership = integrity.check_eval_split_membership()
    assert membership["chunks_unchanged"], membership
    assert membership["chunks_missing_pmid"] == 0, membership


def test_leakage_guard_rejects_production_root():
    guard = IndexLeakageGuard()
    with pytest.raises(IndexLeakageError):
        guard.assert_safe(str(PROD_DIR))


def test_leakage_guard_accepts_eval_split_and_isolated_index():
    guard = IndexLeakageGuard()
    guard.assert_safe(str(EVAL_SPLIT_DIR))      # should not raise
    guard.assert_safe(str(ISOLATED_INDEX))      # external path, should not raise
