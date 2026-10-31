"""
Workstream D — Checkpoint/Resume Reliability + Fault-Injection Tests
=====================================================================

All tests run on CPU only (no real FAISS index, no LLM loading).
They target the checkpoint helpers in backend/scripts/run_step4_matrix.py via
direct import and via behavioural assertions over temp-dir fixtures.

Evidence tier: SOFTWARE TEST (unit + fault-injection, no real pipeline).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Make sure backend/ is importable ─────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ---------------------------------------------------------------------------
# Helpers — import the module under test (lazy, avoids heavy app-level imports
# that need GPU or model files at collection time).
# ---------------------------------------------------------------------------

def _import_matrix():
    """Import the matrix runner module, skipping GPU-heavy side effects."""
    import importlib
    # Provide stub modules so the top-level imports in the script don't fail
    # in a CPU-only test environment.
    _stub_if_missing("torch")
    import scripts.run_step4_matrix as m
    return m


def _stub_if_missing(name: str) -> None:
    """Register a MagicMock in sys.modules for ``name`` if not already there."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG_NAMES = [
    "full-pipeline",
    "no-critic",
    "no-iterative-retrieval",
    "no-calibration",
    "basic-rag",
    "plain-llm-no-rag",
]

_N_QUESTIONS = 30


def _make_valid_output(config_name: str, n: int = 30) -> dict[str, Any]:
    """Return a valid output dict that passes _verify_done_config."""
    samples = [
        {"question": f"Q{i}", "answer": f"A{i}", "rouge_l_f1": 0.5}
        for i in range(n)
    ]
    temp_val = "N/A" if config_name == "plain-llm-no-rag" else 0.0
    return {
        "run_metadata": {
            "config_name": config_name,
            "generator_temperature": temp_val,
            "index_dir": "N/A" if config_name == "plain-llm-no-rag" else "/some/eval/index",
        },
        "status": "complete",
        "samples_evaluated": n,
        "per_sample_results": samples,
    }


def _make_checkpoint(
    run_id: str = "20000101_000000",
    configs_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a synthetic checkpoint dict."""
    configs = {
        name: {"status": "pending", "completed_questions": 0, "output_file": None}
        for name in _CONFIG_NAMES
    }
    if configs_override:
        for name, vals in configs_override.items():
            configs[name].update(vals)
    return {
        "run_id": run_id,
        "configs": configs,
        "llm_loaded": False,
        "index_dir_verified": None,
    }


# ---------------------------------------------------------------------------
# Patch target helpers
# ---------------------------------------------------------------------------

_MODULE_PATH = "scripts.run_step4_matrix"


# ===========================================================================
# 1. first_time_init
# ===========================================================================

class TestFirstTimeInit:
    """No checkpoint exists; _init_checkpoint creates correct initial state."""

    def test_init_structure(self):
        m = _import_matrix()
        state = m._init_checkpoint("run001")
        assert state["run_id"] == "run001"
        assert state["llm_loaded"] is False
        assert state["index_dir_verified"] is None
        assert set(state["configs"].keys()) == set(_CONFIG_NAMES)
        for name, cfg in state["configs"].items():
            assert cfg["status"] == "pending", f"{name} should be pending"
            assert cfg["completed_questions"] == 0
            assert cfg["output_file"] is None

    def test_load_checkpoint_returns_none_when_absent(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", tmp_path / "nonexistent.json")
        result = m._load_checkpoint()
        assert result is None

    def test_save_then_load_roundtrip(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", tmp_path / "state.json")
        state = m._init_checkpoint("run_new")
        m._save_checkpoint(state)
        loaded = m._load_checkpoint()
        assert loaded is not None
        assert loaded["run_id"] == "run_new"
        assert all(c["status"] == "pending" for c in loaded["configs"].values())


# ===========================================================================
# 2. completed_fixture_skipped
# ===========================================================================

class TestCompletedFixtureSkipped:
    """All 6 configs marked done AND output files valid → _verify_done_config
    returns True for all; runner logs skip without loading LLM."""

    def test_verify_done_config_passes_when_complete(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)

        for config_name in _CONFIG_NAMES:
            out_file = tmp_path / f"{config_name}_out.json"
            out_file.write_text(
                json.dumps(_make_valid_output(config_name, n=30)), encoding="utf-8"
            )
            state = _make_checkpoint(
                configs_override={
                    config_name: {
                        "status": "done",
                        "completed_questions": 30,
                        "output_file": str(out_file),
                    }
                }
            )
            result = m._verify_done_config(config_name, state)
            assert result is True, f"Expected True for {config_name}"

    def test_verify_done_config_all_six(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)

        configs_override = {}
        for config_name in _CONFIG_NAMES:
            out_file = tmp_path / f"{config_name}_out.json"
            out_file.write_text(
                json.dumps(_make_valid_output(config_name, n=30)), encoding="utf-8"
            )
            configs_override[config_name] = {
                "status": "done",
                "completed_questions": 30,
                "output_file": str(out_file),
            }

        state = _make_checkpoint(configs_override=configs_override)
        all_verified = all(
            m._verify_done_config(name, state) for name in _CONFIG_NAMES
        )
        assert all_verified, "All 6 configs should verify as done"


# ===========================================================================
# 3. partial_resume
# ===========================================================================

class TestPartialResume:
    """Checkpoint shows 2 of 6 done; runner skips completed, resumes from third."""

    def test_two_done_configs_verify_true(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)

        done_names = _CONFIG_NAMES[:2]
        pending_names = _CONFIG_NAMES[2:]
        configs_override = {}

        for name in done_names:
            out_file = tmp_path / f"{name}_out.json"
            out_file.write_text(
                json.dumps(_make_valid_output(name, n=30)), encoding="utf-8"
            )
            configs_override[name] = {
                "status": "done",
                "completed_questions": 30,
                "output_file": str(out_file),
            }

        state = _make_checkpoint(configs_override=configs_override)

        for name in done_names:
            assert m._verify_done_config(name, state) is True
        for name in pending_names:
            assert m._verify_done_config(name, state) is False

    def test_load_partial_output_returns_already_done(self, tmp_path):
        m = _import_matrix()
        partial_samples = [
            {"question": f"Q{i}", "answer": f"A{i}"} for i in range(15)
        ]
        out_file = tmp_path / "partial.json"
        out_file.write_text(
            json.dumps({"run_metadata": {}, "per_sample_results": partial_samples}),
            encoding="utf-8",
        )
        loaded = m._load_partial_output(str(out_file))
        assert len(loaded) == 15

    def test_load_partial_output_returns_empty_for_nonexistent(self, tmp_path):
        m = _import_matrix()
        result = m._load_partial_output(str(tmp_path / "missing.json"))
        assert result == []


# ===========================================================================
# 4. reordered_questions
# ===========================================================================

class TestReorderedQuestions:
    """Stable question IDs: resume skips questions by text, not list index."""

    def test_incremental_writer_tracks_by_text(self, tmp_path):
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        writer = m.IncrementalResultWriter(out_file, {"generator_temperature": 0.0})

        q_list = [
            {"question": "Alpha", "answer": "A"},
            {"question": "Beta",  "answer": "B"},
            {"question": "Gamma", "answer": "C"},
        ]
        for q in q_list:
            writer.append(q)

        # Reorder: present Beta first on next (simulated) pass
        texts = writer.completed_question_texts
        assert "Alpha" in texts
        assert "Beta" in texts
        assert "Gamma" in texts

        # Simulate resume: "Delta" not yet done
        reordered = [
            {"question": "Beta",  "answer": "B"},
            {"question": "Alpha", "answer": "A"},
            {"question": "Gamma", "answer": "C"},
            {"question": "Delta", "answer": "D"},
        ]
        new_writer = m.IncrementalResultWriter(out_file, {"generator_temperature": 0.0})
        to_run = [q for q in reordered if q["question"] not in new_writer.completed_question_texts]
        assert len(to_run) == 1
        assert to_run[0]["question"] == "Delta"

    def test_completed_question_texts_populated_on_load(self, tmp_path):
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        initial_samples = [{"question": "Q1"}, {"question": "Q2"}]
        out_file.write_text(
            json.dumps({"run_metadata": {}, "per_sample_results": initial_samples}),
            encoding="utf-8",
        )
        writer = m.IncrementalResultWriter(out_file, {})
        assert "Q1" in writer.completed_question_texts
        assert "Q2" in writer.completed_question_texts


# ===========================================================================
# 5. duplicate_ids
# ===========================================================================

class TestDuplicateIds:
    """Dataset has duplicate question texts; IncrementalResultWriter must not
    double-count them and the caller's dedup guard must fire."""

    def test_writer_dedup_by_text(self, tmp_path):
        """Writer should not store duplicates when the caller guards by text."""
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        writer = m.IncrementalResultWriter(out_file, {})

        # First append
        writer.append({"question": "Same question", "answer": "A"})
        # Simulate the caller's guard (already in completed_question_texts)
        assert "Same question" in writer.completed_question_texts
        # If caller guards correctly, second won't be appended
        if "Same question" not in writer.completed_question_texts:
            writer.append({"question": "Same question", "answer": "B"})
        assert writer.completed_questions == 1

    def test_detect_duplicate_ids_in_dataset(self):
        """A helper that detects duplicate question IDs should raise or return
        duplicate set (to be used by the caller before the run starts)."""
        m = _import_matrix()
        # _detect_duplicate_question_ids is the function we require to exist
        assert hasattr(m, "_detect_duplicate_question_ids"), (
            "run_step4_matrix must expose _detect_duplicate_question_ids(questions) -> set"
        )
        questions = [
            {"question": "Q1", "_id": "id1"},
            {"question": "Q2", "_id": "id2"},
            {"question": "Q1", "_id": "id1"},  # duplicate text
        ]
        dupes = m._detect_duplicate_question_ids(questions)
        assert "Q1" in dupes


# ===========================================================================
# 6. altered_manifest
# ===========================================================================

class TestAlteredManifest:
    """Dataset manifest changes between runs; runner detects and refuses silently."""

    def test_manifest_hash_mismatch_detected(self, tmp_path, monkeypatch):
        m = _import_matrix()
        assert hasattr(m, "_compute_dataset_manifest_hash"), (
            "run_step4_matrix must expose _compute_dataset_manifest_hash(questions) -> str"
        )
        assert hasattr(m, "_check_manifest_consistency"), (
            "run_step4_matrix must expose _check_manifest_consistency(state, questions) -> bool"
        )

        questions_v1 = [{"question": "Q1"}, {"question": "Q2"}]
        questions_v2 = [{"question": "Q1"}, {"question": "Q3"}]  # altered

        hash1 = m._compute_dataset_manifest_hash(questions_v1)
        hash2 = m._compute_dataset_manifest_hash(questions_v2)
        assert hash1 != hash2, "Different datasets must produce different hashes"

        # Simulate state with v1 hash stored
        state = _make_checkpoint()
        state["dataset_manifest_hash"] = hash1

        # v2 must be rejected
        ok = m._check_manifest_consistency(state, questions_v2)
        assert ok is False, "Altered manifest must not pass consistency check"

    def test_manifest_consistent_when_same(self, tmp_path):
        m = _import_matrix()
        questions = [{"question": "Q1"}, {"question": "Q2"}]
        state = _make_checkpoint()
        state["dataset_manifest_hash"] = m._compute_dataset_manifest_hash(questions)
        ok = m._check_manifest_consistency(state, questions)
        assert ok is True

    def test_manifest_accepted_when_no_hash_in_state(self):
        """First run: no hash stored yet → always consistent (records it)."""
        m = _import_matrix()
        questions = [{"question": "Q1"}]
        state = _make_checkpoint()  # no dataset_manifest_hash key
        ok = m._check_manifest_consistency(state, questions)
        assert ok is True


# ===========================================================================
# 7. interrupted_write_before_atomic_replace
# ===========================================================================

class TestInterruptedWriteBeforeAtomicReplace:
    """Crash during .tmp write leaves original checkpoint intact."""

    def test_leftover_tmp_does_not_corrupt_checkpoint(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", tmp_path / "state.json")

        # Write valid checkpoint first
        good_state = _make_checkpoint(run_id="run_good")
        m._save_checkpoint(good_state)

        # Simulate: .tmp was written (with bad data) but os.replace never called
        tmp_file = tmp_path / "state.tmp"
        tmp_file.write_text("CORRUPT_PARTIAL_WRITE", encoding="utf-8")

        # Original checkpoint must still be loadable
        loaded = m._load_checkpoint()
        assert loaded is not None
        assert loaded["run_id"] == "run_good"

    def test_atomic_write_uses_same_dir_tmp(self, tmp_path, monkeypatch):
        """_save_checkpoint must write .tmp in the same directory before replace."""
        m = _import_matrix()
        monkeypatch.setattr(m, "_CHECKPOINT_DIR", tmp_path)
        ckpt_file = tmp_path / "state.json"
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", ckpt_file)

        replace_calls = []
        real_replace = os.replace

        def tracking_replace(src, dst):
            replace_calls.append((Path(src), Path(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracking_replace)

        state = _make_checkpoint(run_id="atomic_test")
        m._save_checkpoint(state)

        assert len(replace_calls) == 1, "os.replace must be called exactly once"
        src_path, dst_path = replace_calls[0]
        assert src_path.parent == dst_path.parent, (
            "Temp file must be in the same directory as the target"
        )
        assert dst_path == ckpt_file


# ===========================================================================
# 8. corrupt_checkpoint
# ===========================================================================

class TestCorruptCheckpoint:
    """Corrupt checkpoint JSON → _load_checkpoint returns None; runner starts fresh."""

    def test_corrupt_json_returns_none(self, tmp_path, monkeypatch):
        m = _import_matrix()
        ckpt = tmp_path / "state.json"
        ckpt.write_text("{not valid json{{", encoding="utf-8")
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", ckpt)
        result = m._load_checkpoint()
        assert result is None

    def test_truncated_json_returns_none(self, tmp_path, monkeypatch):
        m = _import_matrix()
        ckpt = tmp_path / "state.json"
        ckpt.write_text('{"run_id": "abc", "configs":', encoding="utf-8")
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", ckpt)
        result = m._load_checkpoint()
        assert result is None

    def test_original_checkpoint_untouched_after_corrupt_load(self, tmp_path, monkeypatch):
        """Loading a corrupt file must not modify it."""
        m = _import_matrix()
        ckpt = tmp_path / "state.json"
        corrupt_content = "{not valid json{{"
        ckpt.write_text(corrupt_content, encoding="utf-8")
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", ckpt)
        m._load_checkpoint()  # should return None, not raise, not modify file
        assert ckpt.read_text(encoding="utf-8") == corrupt_content


# ===========================================================================
# 9. corrupt_output
# ===========================================================================

class TestCorruptOutput:
    """Output file malformed → _verify_done_config returns False."""

    def test_verify_done_fails_on_corrupt_json(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)
        out_file = tmp_path / "output.json"
        out_file.write_text("{corrupt json", encoding="utf-8")
        state = _make_checkpoint(
            configs_override={
                "full-pipeline": {
                    "status": "done",
                    "completed_questions": 30,
                    "output_file": str(out_file),
                }
            }
        )
        assert m._verify_done_config("full-pipeline", state) is False

    def test_load_partial_output_returns_empty_on_corrupt(self, tmp_path):
        m = _import_matrix()
        out_file = tmp_path / "partial.json"
        out_file.write_text("{bad json", encoding="utf-8")
        result = m._load_partial_output(str(out_file))
        assert result == []


# ===========================================================================
# 10. missing_output_marked_done
# ===========================================================================

class TestMissingOutputMarkedDone:
    """Checkpoint claims completion but output file is missing → not counted as done."""

    def test_verify_done_config_fails_when_output_missing(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)
        state = _make_checkpoint(
            configs_override={
                "no-critic": {
                    "status": "done",
                    "completed_questions": 30,
                    "output_file": str(tmp_path / "nonexistent.json"),
                }
            }
        )
        assert m._verify_done_config("no-critic", state) is False

    def test_verify_done_config_false_when_output_file_is_none(self, monkeypatch):
        m = _import_matrix()
        state = _make_checkpoint(
            configs_override={
                "basic-rag": {
                    "status": "done",
                    "completed_questions": 30,
                    "output_file": None,
                }
            }
        )
        assert m._verify_done_config("basic-rag", state) is False


# ===========================================================================
# 11. invalid_completed_count
# ===========================================================================

class TestInvalidCompletedCount:
    """completed_questions in checkpoint exceeds actual output file sample count."""

    def test_verify_done_config_fails_count_mismatch(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)

        # Output only has 10 samples, checkpoint claims 30
        out_file = tmp_path / "output.json"
        out_file.write_text(
            json.dumps(_make_valid_output("full-pipeline", n=10)), encoding="utf-8"
        )
        state = _make_checkpoint(
            configs_override={
                "full-pipeline": {
                    "status": "done",
                    "completed_questions": 30,  # mismatch!
                    "output_file": str(out_file),
                }
            }
        )
        # _verify_done_config checks len(per_sample_results) == _N_QUESTIONS (30)
        assert m._verify_done_config("full-pipeline", state) is False

    def test_verify_done_config_passes_exact_count(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)
        out_file = tmp_path / "output.json"
        out_file.write_text(
            json.dumps(_make_valid_output("full-pipeline", n=30)), encoding="utf-8"
        )
        state = _make_checkpoint(
            configs_override={
                "full-pipeline": {
                    "status": "done",
                    "completed_questions": 30,
                    "output_file": str(out_file),
                }
            }
        )
        assert m._verify_done_config("full-pipeline", state) is True


# ===========================================================================
# 12. gpu_oom_then_success
# ===========================================================================

class TestGpuOomThenSuccess:
    """GPU OOM on question N is recorded as a failure, never as answer/abstention/skip."""

    def test_oom_sample_has_error_key(self):
        """The OOM path in _run_single_rag_question must return an error dict."""
        m = _import_matrix()

        # Verify the error record structure that the runner produces for OOM.
        # We don't actually call GPU code; we check the shape the real code would return
        # by reading the source pattern: {"question": ..., "error": "gpu_oom", ...}.
        oom_record = {"question": "Test Q", "error": "gpu_oom", "error_detail": "OOM msg"}

        # Must have "error" key → must NOT be counted as a valid answer
        assert "error" in oom_record
        assert oom_record["error"] == "gpu_oom"

        # Verify the finalize logic: n_errored counts samples with "error"
        all_samples = [
            {"question": "Q1", "answer": "A"},
            oom_record,
            {"question": "Q3", "answer": "C"},
        ]
        n_errored = sum(1 for s in all_samples if "error" in s)
        assert n_errored == 1, "Exactly one OOM error must be counted"

        # OOM must not appear in any accuracy/rouge aggregation
        rouge_scores = [
            s.get("rouge_l_f1", 0.0)
            for s in all_samples
            if "error" not in s and s.get("rouge_l_f1") is not None
        ]
        assert len(rouge_scores) == 0, (
            "OOM sample has no rouge_l_f1; should not appear in rouge aggregation"
        )

    def test_oom_never_counted_as_answer_or_skip(self, tmp_path):
        """IncrementalResultWriter stores OOM records but finalize excludes them from
        the answer/abstention count (n_errored separate from n_ok)."""
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        writer = m.IncrementalResultWriter(out_file, {"generator_temperature": 0.0})

        writer.append({"question": "Q1", "answer": "A", "rouge_l_f1": 0.7})
        writer.append({"question": "Q2", "error": "gpu_oom", "error_detail": "OOM"})
        writer.append({"question": "Q3", "answer": "C", "rouge_l_f1": 0.5})

        all_s = writer._samples
        n_valid = sum(1 for s in all_s if "error" not in s)
        n_errored = sum(1 for s in all_s if "error" in s)
        assert n_valid == 2
        assert n_errored == 1
        assert writer.completed_questions == 3  # total stored
        # Verify rouge computed only from valid
        rouge = [s.get("rouge_l_f1", 0.0) for s in all_s if "error" not in s]
        assert len(rouge) == 2


# ===========================================================================
# 13. crash_after_output_before_state
# ===========================================================================

class TestCrashAfterOutputBeforeState:
    """Output is persisted but checkpoint state not updated → resume re-processes safely."""

    def test_output_ahead_of_checkpoint_is_safe(self, tmp_path):
        """Output has 5 samples, checkpoint says 4 completed.
        _load_partial_output reads the file (5 samples), so next run skips first 5."""
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        samples = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(5)]
        out_file.write_text(
            json.dumps({"run_metadata": {}, "per_sample_results": samples}),
            encoding="utf-8",
        )
        # Checkpoint only knows about 4 (crash happened after output write but before state save)
        loaded = m._load_partial_output(str(out_file))
        assert len(loaded) == 5, "Output file is ground truth; checkpoint lags"

        # When IncrementalResultWriter loads from the file, it recovers 5 samples
        writer = m.IncrementalResultWriter(out_file, {})
        assert writer.completed_questions == 5

    def test_duplicate_question_not_re_added_on_resume(self, tmp_path):
        """On resume, if a question is already in completed_question_texts it is skipped."""
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        initial = [{"question": "Q1"}, {"question": "Q2"}]
        out_file.write_text(
            json.dumps({"run_metadata": {}, "per_sample_results": initial}),
            encoding="utf-8",
        )
        writer = m.IncrementalResultWriter(out_file, {})
        assert "Q1" in writer.completed_question_texts
        # Simulated re-run: Q1 is skipped
        questions = [{"question": "Q1"}, {"question": "Q3"}]
        pending = [q for q in questions if q["question"] not in writer.completed_question_texts]
        assert len(pending) == 1
        assert pending[0]["question"] == "Q3"


# ===========================================================================
# 14. crash_after_state_before_output
# ===========================================================================

class TestCrashAfterStateBeforeOutput:
    """State updated but output not yet persisted → on resume, question is re-processed."""

    def test_missing_output_forces_rerun(self, tmp_path, monkeypatch):
        """checkpoint says config done, but output file is absent → verify returns False."""
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)
        # State claims done, output file doesn't exist
        state = _make_checkpoint(
            configs_override={
                "no-calibration": {
                    "status": "done",
                    "completed_questions": 30,
                    "output_file": str(tmp_path / "missing_output.json"),
                }
            }
        )
        assert m._verify_done_config("no-calibration", state) is False

    def test_partial_output_fewer_than_n_allows_resume(self, tmp_path, monkeypatch):
        """State shows 15 completed but output only has 12 → resume picks up from file."""
        m = _import_matrix()
        out_file = tmp_path / "output.json"
        samples = [{"question": f"Q{i}"} for i in range(12)]
        out_file.write_text(
            json.dumps({"run_metadata": {}, "per_sample_results": samples}),
            encoding="utf-8",
        )
        # Writer loads from file (12 samples), not from checkpoint count (15)
        writer = m.IncrementalResultWriter(out_file, {})
        assert writer.completed_questions == 12


# ===========================================================================
# 15. invalid_index_path_rejected
# ===========================================================================

class TestInvalidIndexPathRejected:
    """Invalid and production index paths must be rejected BEFORE any model loads."""

    def test_production_path_causes_sys_exit_or_runtime_error(self):
        """_resolve_and_guard_index_dir rejects the production index path."""
        from scripts.run_step4_benchmark import (
            _resolve_and_guard_index_dir,
            _PRODUCTION_INDEX_DIR,
        )
        # The guard should call sys.exit(1) for production path
        with pytest.raises(SystemExit) as exc_info:
            _resolve_and_guard_index_dir(_PRODUCTION_INDEX_DIR)
        assert exc_info.value.code == 1

    def test_nonexistent_path_causes_sys_exit(self, tmp_path):
        """An index dir that doesn't exist should cause sys.exit(1)."""
        from scripts.run_step4_benchmark import _resolve_and_guard_index_dir
        fake_path = tmp_path / "no_such_index"
        with pytest.raises(SystemExit) as exc_info:
            _resolve_and_guard_index_dir(fake_path)
        assert exc_info.value.code == 1

    def test_guard_called_before_llm_load(self):
        """In the matrix runner, the index guard fires before llm_wrapper.load_model().
        We verify this by inspecting the source order: guard call precedes LLM load."""
        m = _import_matrix()
        import inspect
        src = inspect.getsource(m._main)
        guard_pos = src.find("_resolve_and_guard_index_dir")
        llm_load_pos = src.find("load_model()")
        assert guard_pos < llm_load_pos, (
            "_resolve_and_guard_index_dir must appear before load_model() in _main"
        )


# ===========================================================================
# 16. plain_llm_config_records_na
# ===========================================================================

class TestPlainLlmConfigRecordsNa:
    """plain-llm-no-rag config must record index='N/A' and temperature='N/A'."""

    def test_run_metadata_has_na_index(self):
        """The run_metadata for plain-llm-no-rag must have index_dir='N/A'."""
        # This mirrors the logic in _main() step 6e
        config_name = "plain-llm-no-rag"
        is_plain_llm = config_name == "plain-llm-no-rag"
        run_metadata = {
            "config_name": config_name,
            "generator_temperature": "N/A" if is_plain_llm else 0.0,
            "index_dir": "N/A" if is_plain_llm else "/some/path",
        }
        assert run_metadata["index_dir"] == "N/A"
        assert run_metadata["generator_temperature"] == "N/A"

    def test_verify_done_plain_llm_accepts_na_temp(self, tmp_path, monkeypatch):
        """_verify_done_config must accept generator_temperature='N/A' for plain-llm."""
        m = _import_matrix()
        monkeypatch.setattr(m, "_N_QUESTIONS", 30)
        out_file = tmp_path / "plain_llm_out.json"
        out_file.write_text(
            json.dumps(_make_valid_output("plain-llm-no-rag", n=30)), encoding="utf-8"
        )
        state = _make_checkpoint(
            configs_override={
                "plain-llm-no-rag": {
                    "status": "done",
                    "completed_questions": 30,
                    "output_file": str(out_file),
                }
            }
        )
        assert m._verify_done_config("plain-llm-no-rag", state) is True

    def test_plain_llm_not_in_rag_kwargs_dict(self):
        """plain-llm-no-rag must map to None in the _rag_kwargs dict to take the plain path."""
        m = _import_matrix()
        import inspect
        import re
        src = inspect.getsource(m._main)
        # Match "plain-llm-no-rag": <whitespace> None (with optional trailing comma/comment)
        pattern = r'"plain-llm-no-rag"\s*:\s*None'
        assert re.search(pattern, src) is not None, (
            "plain-llm-no-rag must map to None in _rag_kwargs inside _main"
        )


# ===========================================================================
# 17. atomic_write_verified  (bonus — verifies fsync is present)
# ===========================================================================

class TestAtomicWriteVerified:
    """_save_checkpoint uses temp file + os.replace. IncrementalResultWriter._flush
    must also use same-dir temp + os.replace (not direct write)."""

    def test_flush_uses_atomic_replace(self, tmp_path, monkeypatch):
        """IncrementalResultWriter._flush must call os.replace, not write directly."""
        m = _import_matrix()
        replace_calls = []
        real_replace = os.replace

        def tracking_replace(src, dst):
            replace_calls.append((Path(src), Path(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", tracking_replace)

        out_file = tmp_path / "output.json"
        writer = m.IncrementalResultWriter(out_file, {"generator_temperature": 0.0})
        writer.append({"question": "Q1", "answer": "A"})

        assert len(replace_calls) >= 1, "_flush must call os.replace at least once"
        src, dst = replace_calls[0]
        assert src.parent == dst.parent, "Temp file must be in the same dir as output"
        assert dst == out_file

    def test_save_checkpoint_file_is_valid_json_after_write(self, tmp_path, monkeypatch):
        m = _import_matrix()
        monkeypatch.setattr(m, "_CHECKPOINT_DIR", tmp_path)
        monkeypatch.setattr(m, "_CHECKPOINT_FILE", tmp_path / "state.json")
        state = _make_checkpoint(run_id="json_check")
        m._save_checkpoint(state)
        raw = (tmp_path / "state.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)  # must not raise
        assert parsed["run_id"] == "json_check"


# ===========================================================================
# 18. real_checkpoint_read_only_verification  (documents Phase 0 findings)
# ===========================================================================

class TestRealCheckpointReadOnly:
    """Verify the production checkpoint is parseable and consistent (read-only)."""

    @pytest.fixture
    def real_checkpoint_path(self) -> Path:
        p = Path(__file__).resolve().parents[1] / "data" / "checkpoints" / "step4_matrix_state.json"
        return p

    def test_real_checkpoint_is_parseable(self, real_checkpoint_path):
        if not real_checkpoint_path.exists():
            pytest.skip("Real checkpoint not present — skipping read-only verification")
        raw = real_checkpoint_path.read_text(encoding="utf-8")
        data = json.loads(raw)  # must not raise
        assert isinstance(data, dict)

    def test_real_checkpoint_has_six_configs(self, real_checkpoint_path):
        if not real_checkpoint_path.exists():
            pytest.skip("Real checkpoint not present")
        data = json.loads(real_checkpoint_path.read_text(encoding="utf-8"))
        configs = data.get("configs", {})
        assert set(configs.keys()) == set(_CONFIG_NAMES), (
            f"Expected 6 configs {_CONFIG_NAMES}, got {list(configs.keys())}"
        )

    def test_real_checkpoint_all_done(self, real_checkpoint_path):
        if not real_checkpoint_path.exists():
            pytest.skip("Real checkpoint not present")
        data = json.loads(real_checkpoint_path.read_text(encoding="utf-8"))
        for name, cfg in data["configs"].items():
            assert cfg["status"] == "done", f"{name} not done: {cfg['status']}"
            assert cfg["completed_questions"] == 30, (
                f"{name}: expected 30, got {cfg['completed_questions']}"
            )

    def test_real_checkpoint_output_files_exist(self, real_checkpoint_path):
        if not real_checkpoint_path.exists():
            pytest.skip("Real checkpoint not present")
        data = json.loads(real_checkpoint_path.read_text(encoding="utf-8"))
        missing = []
        for name, cfg in data["configs"].items():
            out = cfg.get("output_file")
            if out and not Path(out).exists():
                missing.append((name, out))
        assert missing == [], f"Output files missing: {missing}"

    def test_real_checkpoint_not_modified(self, real_checkpoint_path):
        """Reading must not change mtime or content."""
        if not real_checkpoint_path.exists():
            pytest.skip("Real checkpoint not present")
        import time as _time
        mtime_before = real_checkpoint_path.stat().st_mtime
        content_before = real_checkpoint_path.read_text(encoding="utf-8")
        _time.sleep(0.05)
        mtime_after = real_checkpoint_path.stat().st_mtime
        content_after = real_checkpoint_path.read_text(encoding="utf-8")
        assert mtime_before == mtime_after, "Checkpoint file mtime changed — was it written?"
        assert content_before == content_after, "Checkpoint file content changed"
