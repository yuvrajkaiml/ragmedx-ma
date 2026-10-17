"""FEAT-002 — Per-cell checkpoint/resume tests for the phase5 runner.

All tests run on CPU only: no GPU, no model, no real FAISS index. The matrix
orchestration (``run_matrix``) takes an injected async ``cell_runner`` so these
tests exercise the real checkpoint/resume/dedup/retry logic with a stub.

Covers (FEAT-002 step 8):
  (a) resume skips validated completed cells and runs only missing;
  (b) duplicate prevention (second append for same cell_key is refused);
  (c) corrupted/invalid record is NOT skipped and is re-run;
  (d) partial write (crash between append and state flush) recovers w/o dup;
  (e) failure recovery (failed cell retried up to bound, then recorded failed);
  (f) abstention recorded distinctly from error.
Plus: pilot-cell reuse seeding + dry-run plan counts (reused==128).

Evidence tier: SOFTWARE TEST (unit + fault-injection, no real pipeline).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import scripts.run_phase5_experiment as m  # noqa: E402

_PILOT_DIR = _BACKEND_DIR / "results" / "phase5_20261017_131123"
_MANIFEST = _BACKEND_DIR / "results" / "phase5_manifest" / "experiment_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rag_record(qid: str, config_id: str, *, answer: str = "an answer",
                abstain: bool = False, errors: Any = None) -> Dict[str, Any]:
    """A schema-complete per-cell record that passes _validate_record."""
    return {
        "question_id": qid,
        "resolved_index_path": r"D:\Ragmed_2\backend\data\faiss_index\eval_split",
        "split": "test",
        "config": config_id,
        "source": "pubmedqa",
        "gold_answer": "yes",
        "answer": _NOT_MEASURED_IF(errors, answer, abstain),
        "status": "error" if errors else ("abstained" if abstain else "verified"),
        "citations": [] if not errors else m._NOT_MEASURED,
        "claim_verdicts": {"value": "x"} if not errors else m._NOT_MEASURED,
        "confidence": 0.5 if not errors else m._NOT_MEASURED,
        "confidence_note": "x",
        "abstain_flag": bool(abstain),
        "errors": errors,
        "error_code": None,
        "per_stage_latency_ms": m._NOT_MEASURED,
        "model_call_count": 1,
        "retrieval_call_count": 1,
        "cache_hits": 0,
        "peak_vram_mb": m._NOT_MEASURED,
        "cold_latency_ms": 100,
        "warm_latency_ms": m._NOT_MEASURED,
    }


def _NOT_MEASURED_IF(errors, answer, abstain):
    if errors:
        return m._NOT_MEASURED
    if abstain:
        return ""
    return answer


def _make_question(qid: str) -> Dict[str, Any]:
    return {"id": qid, "question": f"Q {qid}", "answer": "yes", "source": "pubmedqa",
            "split": "test"}


def _run(coro):
    return asyncio.run(coro)


# A couple of valid question-ids drawn from the frozen set so _validate_record's
# allowed_ids check passes.
_QID_A = "pubmedqa::15943725"
_QID_B = "pubmedqa::21394762"
_CFG = "phase5-full-pipeline"


# ---------------------------------------------------------------------------
# (a) resume skips validated completed cells, runs only missing
# ---------------------------------------------------------------------------

class TestResumeSkipsCompleted:
    def test_resume_runs_only_missing(self, tmp_path):
        out_root = tmp_path / "run"
        allowed = {_QID_A, _QID_B}
        questions = [_make_question(_QID_A), _make_question(_QID_B)]
        configs = list(m._CONFIGS.keys())

        # Pre-seed QID_A across all configs as completed+valid.
        state = {"version": 1, "cells": {}}
        per_q = out_root / "per_question"
        per_q.mkdir(parents=True)
        for cfg in configs:
            rec = _rag_record(_QID_A, cfg) if cfg != "phase5-plain-llm-no-rag" else {
                **_rag_record(_QID_A, cfg), "resolved_index_path": "N/A (plain-llm-no-rag)"}
            m._append_jsonl(per_q / f"{cfg}.jsonl", rec)
            m._mark_cell(out_root, state, m._cell_key(_QID_A, cfg), "completed",
                         output_file=str(per_q / f"{cfg}.jsonl"), attempts=0)

        ran: List[str] = []

        async def stub(*, question_item, split_name, config_id, config_def):
            ran.append(m._cell_key(question_item["id"], config_id))
            rip = ("N/A (plain-llm-no-rag)" if config_id == "phase5-plain-llm-no-rag"
                   else r"...\eval_split")
            return {**_rag_record(question_item["id"], config_id), "resolved_index_path": rip}

        counts = _run(m.run_matrix(out_root=out_root, question_records=questions,
                                   allowed_ids=allowed, state=state, cell_runner=stub))

        # Only QID_B's 8 cells should have run; QID_A's 8 were reused.
        assert len(ran) == 8
        assert all(k.startswith(_QID_B) for k in ran)
        assert counts["reused_skipped"] == 8
        assert counts["completed"] == 8


# ---------------------------------------------------------------------------
# (b) duplicate prevention
# ---------------------------------------------------------------------------

class TestDuplicatePrevention:
    def test_second_append_refused(self, tmp_path):
        out_path = tmp_path / "cfg.jsonl"
        rec = _rag_record(_QID_A, _CFG)
        assert m._append_cell_guarded(out_path, rec) is True
        # Second write for the same question_id is refused.
        assert m._append_cell_guarded(out_path, rec) is False
        lines = m._read_jsonl(out_path)
        assert len(lines) == 1

    def test_matrix_does_not_double_write_preexisting_line(self, tmp_path):
        out_root = tmp_path / "run"
        allowed = {_QID_A}
        per_q = out_root / "per_question"
        per_q.mkdir(parents=True)
        # A line already exists on disk but checkpoint has NO completed mark
        # (crash between append and state flush).
        for cfg in m._CONFIGS:
            rip = ("N/A (plain-llm-no-rag)" if cfg == "phase5-plain-llm-no-rag"
                   else r"x\eval_split")
            m._append_jsonl(per_q / f"{cfg}.jsonl",
                            {**_rag_record(_QID_A, cfg), "resolved_index_path": rip})
        state = {"version": 1, "cells": {}}

        async def stub(*, question_item, split_name, config_id, config_def):
            rip = ("N/A (plain-llm-no-rag)" if config_id == "phase5-plain-llm-no-rag"
                   else r"x\eval_split")
            return {**_rag_record(question_item["id"], config_id), "resolved_index_path": rip}

        counts = _run(m.run_matrix(out_root=out_root,
                                   question_records=[_make_question(_QID_A)],
                                   allowed_ids=allowed, state=state, cell_runner=stub))
        # Each config file still has exactly 1 line (no duplicate), and the cell
        # was recognised as already-present and marked completed.
        for cfg in m._CONFIGS:
            assert len(m._read_jsonl(per_q / f"{cfg}.jsonl")) == 1
        assert counts["duplicate_skipped"] == 8


# ---------------------------------------------------------------------------
# (c) corrupted / invalid record is NOT skipped and is re-run
# ---------------------------------------------------------------------------

class TestCorruptRecordRerun:
    def test_invalid_completed_cell_is_rerun(self, tmp_path):
        out_root = tmp_path / "run"
        per_q = out_root / "per_question"
        per_q.mkdir(parents=True)
        allowed = {_QID_A}

        # Write a BROKEN record (missing required 'citations' key) but mark the
        # cell completed in the checkpoint — a dishonest/corrupt state.
        broken = _rag_record(_QID_A, _CFG)
        del broken["citations"]
        m._append_jsonl(per_q / f"{_CFG}.jsonl", broken)
        state = {"version": 1, "cells": {}}
        m._mark_cell(out_root, state, m._cell_key(_QID_A, _CFG), "completed",
                     output_file=str(per_q / f"{_CFG}.jsonl"), attempts=0)

        # plan_matrix must NOT count this as reused.
        plan = m.plan_matrix(out_root=out_root, question_ids=[_QID_A],
                             allowed_ids=allowed, state=state)
        assert m._cell_key(_QID_A, _CFG) in plan["missing"]

    def test_corrupt_checkpoint_file_recovers(self, tmp_path):
        out_root = tmp_path / "run"
        ckpt = m._checkpoint_path(out_root)
        ckpt.parent.mkdir(parents=True)
        ckpt.write_text("{not valid json{{", encoding="utf-8")
        state = m._load_checkpoint(out_root)
        assert state == {"version": 1, "cells": {}}


# ---------------------------------------------------------------------------
# (d) partial write (crash between append and state flush) recovers w/o dup
# ---------------------------------------------------------------------------

class TestPartialWriteRecovery:
    def test_appended_but_state_not_flushed(self, tmp_path):
        out_root = tmp_path / "run"
        per_q = out_root / "per_question"
        per_q.mkdir(parents=True)
        allowed = {_QID_A}
        # Record on disk, but checkpoint empty (state flush never happened).
        rip = r"x\eval_split"
        m._append_jsonl(per_q / f"{_CFG}.jsonl",
                        {**_rag_record(_QID_A, _CFG), "resolved_index_path": rip})
        state = {"version": 1, "cells": {}}

        async def stub(*, question_item, split_name, config_id, config_def):
            return {**_rag_record(question_item["id"], config_id),
                    "resolved_index_path": rip}

        # Run only the single config for QID_A by restricting _CONFIGS? Instead
        # run full matrix and assert the pre-written config has no duplicate.
        _run(m.run_matrix(out_root=out_root, question_records=[_make_question(_QID_A)],
                          allowed_ids=allowed, state=state, cell_runner=stub))
        assert len(m._read_jsonl(per_q / f"{_CFG}.jsonl")) == 1
        # The cell is now recorded completed in the checkpoint.
        assert state["cells"][m._cell_key(_QID_A, _CFG)]["state"] == "completed"


# ---------------------------------------------------------------------------
# (e) failure recovery: retried up to bound, then recorded failed (not dropped)
# ---------------------------------------------------------------------------

class TestFailureRecovery:
    def test_cell_retried_then_recorded_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda *_: None)  # no real backoff wait
        out_root = tmp_path / "run"
        allowed = {_QID_A}
        attempts: Dict[str, int] = {}

        async def always_fails(*, question_item, split_name, config_id, config_def):
            key = m._cell_key(question_item["id"], config_id)
            attempts[key] = attempts.get(key, 0) + 1
            raise RuntimeError("boom")

        state = {"version": 1, "cells": {}}
        counts = _run(m.run_matrix(out_root=out_root,
                                   question_records=[_make_question(_QID_A)],
                                   allowed_ids=allowed, state=state,
                                   cell_runner=always_fails))

        # Each cell tried exactly _MAX_CELL_ATTEMPTS times.
        for cfg in m._CONFIGS:
            assert attempts[m._cell_key(_QID_A, cfg)] == m._MAX_CELL_ATTEMPTS
        assert counts["failed"] == 8
        # A failed record is written (not dropped) with NOT MEASURED + error.
        per_q = out_root / "per_question"
        rec = m._read_jsonl(per_q / f"{_CFG}.jsonl")[0]
        assert rec["answer"] == m._NOT_MEASURED
        assert rec["errors"]
        assert rec["abstain_flag"] is False
        assert rec["error_code"] == "CELL_RETRIES_EXHAUSTED"
        assert state["cells"][m._cell_key(_QID_A, _CFG)]["state"] == "failed"

    def test_transient_failure_then_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda *_: None)  # no real backoff wait
        out_root = tmp_path / "run"
        allowed = {_QID_A}
        calls: Dict[str, int] = {}
        rip = r"x\eval_split"

        async def fail_once(*, question_item, split_name, config_id, config_def):
            key = m._cell_key(question_item["id"], config_id)
            calls[key] = calls.get(key, 0) + 1
            if calls[key] == 1:
                raise RuntimeError("transient")
            return {**_rag_record(question_item["id"], config_id),
                    "resolved_index_path": rip}

        state = {"version": 1, "cells": {}}
        counts = _run(m.run_matrix(out_root=out_root,
                                   question_records=[_make_question(_QID_A)],
                                   allowed_ids=allowed, state=state,
                                   cell_runner=fail_once))
        assert counts["completed"] == 8
        assert counts["failed"] == 0
        for cfg in m._CONFIGS:
            assert calls[m._cell_key(_QID_A, cfg)] == 2  # 1 fail + 1 success


# ---------------------------------------------------------------------------
# (f) abstention recorded distinctly from error
# ---------------------------------------------------------------------------

class TestAbstentionDistinctFromError:
    def test_abstention_is_completed_not_failed(self, tmp_path):
        out_root = tmp_path / "run"
        allowed = {_QID_A}
        rip = r"x\eval_split"

        async def abstains(*, question_item, split_name, config_id, config_def):
            return {**_rag_record(question_item["id"], config_id, abstain=True),
                    "resolved_index_path": rip}

        state = {"version": 1, "cells": {}}
        counts = _run(m.run_matrix(out_root=out_root,
                                   question_records=[_make_question(_QID_A)],
                                   allowed_ids=allowed, state=state,
                                   cell_runner=abstains))
        assert counts["completed"] == 8
        assert counts["abstained"] == 8
        assert counts["failed"] == 0
        rec = m._read_jsonl(out_root / "per_question" / f"{_CFG}.jsonl")[0]
        assert rec["abstain_flag"] is True
        assert rec["errors"] is None
        assert state["cells"][m._cell_key(_QID_A, _CFG)]["state"] == "completed"

    def test_validate_record_abstain_and_error_are_distinct_branches(self):
        allowed = {_QID_A}
        abstain = _rag_record(_QID_A, _CFG, abstain=True)
        err = _rag_record(_QID_A, _CFG, errors="gpu_oom")
        answered = _rag_record(_QID_A, _CFG)
        assert m._validate_record(abstain, question_id=_QID_A, config_id=_CFG,
                                  allowed_ids=allowed)
        assert m._validate_record(err, question_id=_QID_A, config_id=_CFG,
                                  allowed_ids=allowed)
        assert m._validate_record(answered, question_id=_QID_A, config_id=_CFG,
                                  allowed_ids=allowed)
        # A record that both abstains AND errors is self-inconsistent -> invalid.
        both = _rag_record(_QID_A, _CFG, abstain=True)
        both["errors"] = "boom"
        assert not m._validate_record(both, question_id=_QID_A, config_id=_CFG,
                                      allowed_ids=allowed)


# ---------------------------------------------------------------------------
# Validation guard details
# ---------------------------------------------------------------------------

class TestValidateRecordGuards:
    def test_wrong_question_id_rejected(self):
        allowed = {_QID_A}
        rec = _rag_record(_QID_B, _CFG)  # QID_B not in allowed
        assert not m._validate_record(rec, question_id=_QID_B, config_id=_CFG,
                                      allowed_ids=allowed)

    def test_non_eval_split_path_rejected(self):
        allowed = {_QID_A}
        rec = _rag_record(_QID_A, _CFG)
        rec["resolved_index_path"] = r"D:\Ragmed_2\backend\data\faiss_index"  # production!
        assert not m._validate_record(rec, question_id=_QID_A, config_id=_CFG,
                                      allowed_ids=allowed)

    def test_missing_required_key_rejected(self):
        allowed = {_QID_A}
        rec = _rag_record(_QID_A, _CFG)
        del rec["answer"]
        assert not m._validate_record(rec, question_id=_QID_A, config_id=_CFG,
                                      allowed_ids=allowed)


# ---------------------------------------------------------------------------
# Atomic checkpoint write
# ---------------------------------------------------------------------------

class TestAtomicCheckpoint:
    def test_save_uses_tmp_then_replace(self, tmp_path, monkeypatch):
        import os
        out_root = tmp_path / "run"
        replace_calls = []
        real_replace = os.replace

        def tracking(src, dst):
            replace_calls.append((Path(src), Path(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracking)
        m._save_checkpoint(out_root, {"version": 1, "cells": {}})
        assert len(replace_calls) == 1
        src, dst = replace_calls[0]
        assert src.parent == dst.parent
        assert dst == m._checkpoint_path(out_root)

    def test_roundtrip(self, tmp_path):
        out_root = tmp_path / "run"
        state = {"version": 1, "cells": {}}
        m._mark_cell(out_root, state, _QID_A + "::" + _CFG, "completed",
                     output_file="x", attempts=0)
        loaded = m._load_checkpoint(out_root)
        assert loaded["cells"][_QID_A + "::" + _CFG]["state"] == "completed"


# ---------------------------------------------------------------------------
# Pilot-cell reuse seeding + plan counts (acceptance criteria)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _PILOT_DIR.exists() or not _MANIFEST.exists(),
                    reason="pilot dir / manifest not present")
class TestPilotReuseSeeding:
    def _load_ids(self, key: str) -> List[str]:
        return json.loads(_MANIFEST.read_text(encoding="utf-8"))[key]

    def test_seed_reuses_128_cells_and_never_touches_pilot(self, tmp_path):
        out_root = tmp_path / "run"
        ids_50 = self._load_ids("question_ids_50")
        allowed = set(ids_50)
        state = {"version": 1, "cells": {}}

        # Snapshot pilot files' mtimes to assert they are untouched.
        pilot_files = list((_PILOT_DIR / "per_question").glob("*.jsonl"))
        before = {p: p.stat().st_mtime_ns for p in pilot_files}

        seeded = m._seed_pilot_cells(
            out_root=out_root, pilot_dir=_PILOT_DIR, question_ids=ids_50,
            pilot_reused_ids=m._PILOT_REUSED_IDS_16, allowed_ids=allowed, state=state,
        )
        assert seeded == 128

        plan = m.plan_matrix(out_root=out_root, question_ids=ids_50,
                             allowed_ids=allowed, state=state)
        assert plan["total"] == 400
        assert plan["n_reused"] == 128
        assert plan["n_missing"] == 272

        # Pilot dir is read-only input: no file was modified.
        after = {p: p.stat().st_mtime_ns for p in pilot_files}
        assert before == after

    def test_target_40_plan_counts(self, tmp_path):
        out_root = tmp_path / "run"
        ids_40 = self._load_ids("question_ids_40")
        allowed = set(ids_40)
        state = {"version": 1, "cells": {}}
        m._seed_pilot_cells(
            out_root=out_root, pilot_dir=_PILOT_DIR, question_ids=ids_40,
            pilot_reused_ids=m._PILOT_REUSED_IDS_16, allowed_ids=allowed, state=state,
        )
        plan = m.plan_matrix(out_root=out_root, question_ids=ids_40,
                             allowed_ids=allowed, state=state)
        assert plan["total"] == 320
        assert plan["n_reused"] == 128
        assert plan["n_missing"] == 192

    def test_seed_is_idempotent_no_duplicates(self, tmp_path):
        """Running seeding twice must not duplicate JSONL lines."""
        out_root = tmp_path / "run"
        ids_50 = self._load_ids("question_ids_50")
        allowed = set(ids_50)
        state = {"version": 1, "cells": {}}
        m._seed_pilot_cells(out_root=out_root, pilot_dir=_PILOT_DIR, question_ids=ids_50,
                            pilot_reused_ids=m._PILOT_REUSED_IDS_16, allowed_ids=allowed,
                            state=state)
        m._seed_pilot_cells(out_root=out_root, pilot_dir=_PILOT_DIR, question_ids=ids_50,
                            pilot_reused_ids=m._PILOT_REUSED_IDS_16, allowed_ids=allowed,
                            state=state)
        for cfg in m._CONFIGS:
            lines = m._read_jsonl(out_root / "per_question" / f"{cfg}.jsonl")
            qids = [r["question_id"] for r in lines]
            assert len(qids) == len(set(qids)) == 16
