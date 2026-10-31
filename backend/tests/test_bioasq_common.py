"""Common-helper unit tests: PMID normalization, article dedup, JSONL checkpoint resume."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.bioasq_eval.common import (  # noqa: E402
    JsonlCheckpoint, canonical_pmid, config_hash, dedup_articles, normalize_pmid,
)


class _Chunk:
    """Minimal stand-in for MedicalChunk (metadata + source)."""

    def __init__(self, pmid=None, source=None, document_id=None):
        self.metadata = {"pmid": pmid} if pmid is not None else {}
        self.source = source
        self.document_id = document_id


def test_normalize_pmid_variants():
    assert normalize_pmid("http://www.ncbi.nlm.nih.gov/pubmed/29040384") == "29040384"
    assert normalize_pmid("29040384") == "29040384"
    assert normalize_pmid(29040384) == "29040384"
    assert normalize_pmid("PMID:12345") == "12345"
    assert normalize_pmid("  987  ") == "987"


def test_normalize_pmid_unparsable():
    assert normalize_pmid("") is None
    assert normalize_pmid(None) is None
    assert normalize_pmid("no-digits-here") is None


def test_canonical_pmid_metadata_then_source_fallback():
    fb = {}
    assert canonical_pmid(_Chunk(pmid="111"), fb) == "111"
    assert fb.get("pmid_fallback_used", 0) == 0
    # No metadata pmid -> falls back to source parse and counts it.
    assert canonical_pmid(_Chunk(source="http://x/pubmed/222"), fb) == "222"
    assert fb["pmid_fallback_used"] == 1


def test_dedup_articles_preserves_rank_and_collapses_multichunk():
    chunks = [_Chunk(pmid="11"), _Chunk(pmid="11"), _Chunk(pmid="22"), _Chunk(pmid="11"),
              _Chunk(pmid="33")]
    assert dedup_articles(chunks) == ["11", "22", "33"]


def test_dedup_articles_skips_unrecoverable():
    chunks = [_Chunk(pmid=None, source=None), _Chunk(pmid="44")]
    assert dedup_articles(chunks) == ["44"]


def test_config_hash_stable_and_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_jsonl_checkpoint_resume_no_duplication(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    ck = JsonlCheckpoint(path, key_field="question_id")
    assert ck.append({"question_id": "q1", "v": 1}) is True
    assert ck.append({"question_id": "q2", "v": 2}) is True
    # Duplicate key -> no write.
    assert ck.append({"question_id": "q1", "v": 99}) is False

    # Reopen -> completed keys are remembered (resume).
    ck2 = JsonlCheckpoint(path, key_field="question_id")
    assert ck2.is_done("q1") and ck2.is_done("q2")
    assert ck2.append({"question_id": "q1", "v": 1}) is False
    records = ck2.read_all()
    assert len(records) == 2
    assert {r["question_id"] for r in records} == {"q1", "q2"}


def test_jsonl_checkpoint_same_config_rerun_consistent(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    ck = JsonlCheckpoint(path, key_field="question_id")
    for i in range(5):
        ck.append({"question_id": f"q{i}", "v": i})
    first = ck.read_all()
    # Simulate a rerun: all keys already done, nothing appended.
    ck2 = JsonlCheckpoint(path, key_field="question_id")
    for i in range(5):
        assert ck2.append({"question_id": f"q{i}", "v": i}) is False
    assert ck2.read_all() == first
