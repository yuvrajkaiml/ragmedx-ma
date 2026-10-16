"""Step 4 config-matrix runner — with checkpoint/resume support.

Loads Qwen3-8B ONCE, builds the RAGPipeline ONCE from the eval-split index,
then runs all 6 ablation configs sequentially, writing one output JSON per config.

Checkpoint/Resume
-----------------
State is persisted to backend/data/checkpoints/step4_matrix_state.json after
every completed question (atomic write — write .tmp then os.replace).  On any
restart the runner reads that file and skips configs already verified as done,
resuming mid-config from the last completed question.

Usage
-----
    cd d:\\Ragmed_2\\backend
    $env:RAGMEDX_FAISS_INDEX_DIR = "data/faiss_index/eval_split"
    $env:RAGMEDX_LLM_MODE = "real"
    d:\\Ragmed_2\\.venv-cu128-pilot\\Scripts\\python.exe -m scripts.run_step4_matrix

Runtime estimate (printed at startup)
--------------------------------------
  full-pipeline            : 30 × 120s = 3600s  (~60 min)
  no-critic                : 30 ×  70s = 2100s  (~35 min)
  no-iterative-retrieval   : 30 ×  70s = 2100s  (~35 min)
  no-calibration           : 30 × 120s = 3600s  (~60 min)
  basic-rag                : 30 ×  50s = 1500s  (~25 min)
  plain-llm-no-rag         : 30 ×  15s =  450s  ( ~8 min)
  Total                    :                     (~223 min / ~3.7 h)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Ensure backend/ is on sys.path so `app.*` and `evaluation.*` imports work ──
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import BASE_DIR, get_settings  # noqa: E402
from app.utils.logger import logger            # noqa: E402

# Re-use the constants and guard helper from the single-run benchmark script.
from scripts.run_step4_benchmark import (  # noqa: E402
    _CANONICAL_EVAL_SPLIT_DIR,
    _DEFAULT_DATASET_PATH,
    _resolve_and_guard_index_dir,
)

_DEFAULT_OUTPUT_DIR  = BASE_DIR / "data" / "results"
_CHECKPOINT_DIR      = BASE_DIR / "data" / "checkpoints"
_CHECKPOINT_FILE     = _CHECKPOINT_DIR / "step4_matrix_state.json"
_STDOUT_LOG          = BASE_DIR / "matrix_stdout.log"

# Ordered list of config names — order matters for sequential execution.
_CONFIG_NAMES = [
    "full-pipeline",
    "no-critic",
    "no-iterative-retrieval",
    "no-calibration",
    "basic-rag",
    "plain-llm-no-rag",
]

_N_QUESTIONS = 30


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_log(msg: str) -> None:
    """Append a timestamped line to matrix_stdout.log (mirrors console logging)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | CHECKPOINT | {msg}"
    print(line, flush=True)
    try:
        with open(_STDOUT_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # log write failure must not abort the run


def _init_checkpoint(run_id: str) -> dict[str, Any]:
    """Create a fresh checkpoint state dict."""
    return {
        "run_id": run_id,
        "configs": {
            name: {
                "status": "pending",
                "completed_questions": 0,
                "output_file": None,
            }
            for name in _CONFIG_NAMES
        },
        "llm_loaded": False,
        "index_dir_verified": None,
    }


def _load_checkpoint() -> dict[str, Any] | None:
    """Return the persisted checkpoint state, or None if it doesn't exist."""
    if not _CHECKPOINT_FILE.exists():
        return None
    try:
        with open(_CHECKPOINT_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[Matrix] Failed to read checkpoint file ({exc}); starting fresh.")
        return None


def _save_checkpoint(state: dict[str, Any]) -> None:
    """Atomically write checkpoint state to disk (write .tmp then os.replace)."""
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CHECKPOINT_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, _CHECKPOINT_FILE)
    except OSError as exc:
        logger.error(f"[Matrix] Checkpoint write failed: {exc}")


def _verify_done_config(config_name: str, state: dict[str, Any]) -> bool:
    """Return True only if config is done AND its output file passes all integrity checks."""
    cfg_state = state["configs"][config_name]
    output_file = cfg_state.get("output_file")
    if not output_file:
        return False
    path = Path(output_file)
    if not path.exists():
        logger.warning(
            f"[Matrix] Config '{config_name}' marked done but output file missing: {path}"
        )
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[Matrix] Config '{config_name}' output file unreadable: {exc}")
        return False
    # Check expected structure
    meta = data.get("run_metadata", {})
    samples = data.get("per_sample_results", [])
    n_ok = len(samples) == _N_QUESTIONS
    temp_ok = (
        meta.get("generator_temperature") == 0.0
        or (config_name == "plain-llm-no-rag" and meta.get("generator_temperature") == "N/A")
    )
    meta_ok = bool(meta)
    if not (n_ok and temp_ok and meta_ok):
        logger.warning(
            f"[Matrix] Config '{config_name}' failed integrity check — "
            f"samples={len(samples)}/{_N_QUESTIONS}, "
            f"temp={meta.get('generator_temperature')!r}, "
            f"meta_present={meta_ok}. Will rerun."
        )
        return False
    return True


def _load_partial_output(output_file: str | None) -> list[dict[str, Any]]:
    """Load already-completed per_sample_results from a partial output file."""
    if not output_file:
        return []
    path = Path(output_file)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("per_sample_results", [])
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Dataset manifest / duplicate-ID helpers
# ---------------------------------------------------------------------------

def _detect_duplicate_question_ids(questions: list[dict[str, Any]]) -> set[str]:
    """Return the set of question texts (IDs) that appear more than once.

    Question identity is based on the 'question' field (text), which is the
    same key used by IncrementalResultWriter.completed_question_texts for
    resume deduplication.  Callers should raise or log loudly when this set
    is non-empty.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in questions:
        q_text = item.get("question", "")
        if q_text in seen:
            duplicates.add(q_text)
        else:
            seen.add(q_text)
    return duplicates


import hashlib as _hashlib  # noqa: E402 — placed here to keep helpers co-located


def _compute_dataset_manifest_hash(questions: list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 hex digest of the ordered question texts.

    The hash is deterministic for the same ordered list of questions and
    changes whenever a question is added, removed, or its text is altered.
    Used by _check_manifest_consistency to detect dataset drift between runs.
    """
    h = _hashlib.sha256()
    for item in questions:
        q_text = item.get("question", "")
        h.update(q_text.encode("utf-8"))
        h.update(b"\x00")  # null separator so ["ab", "c"] != ["a", "bc"]
    return h.hexdigest()


def _check_manifest_consistency(
    state: dict[str, Any],
    questions: list[dict[str, Any]],
) -> bool:
    """Return True if the dataset matches the hash stored in *state*.

    - If no hash is stored yet (first run), always returns True (first-run
      acceptance) and does NOT modify *state*; the caller is responsible for
      storing the hash after a successful first verification.
    - If a hash is stored and does NOT match, returns False and logs a warning.
    """
    stored_hash = state.get("dataset_manifest_hash")
    if stored_hash is None:
        # First run — no prior hash to compare against; accept unconditionally.
        return True
    current_hash = _compute_dataset_manifest_hash(questions)
    if current_hash != stored_hash:
        logger.warning(
            f"[Matrix] Dataset manifest mismatch: "
            f"stored_hash={stored_hash[:12]}… "
            f"current_hash={current_hash[:12]}… "
            "The dataset appears to have changed since the checkpoint was created. "
            "Refusing to resume silently."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Incremental output-file writer
# ---------------------------------------------------------------------------

class IncrementalResultWriter:
    """Maintains a partial result file, appending one question at a time."""

    def __init__(self, output_path: Path, run_metadata: dict[str, Any]) -> None:
        self.output_path = output_path
        self.run_metadata = run_metadata
        self._samples: list[dict[str, Any]] = []
        # Pre-populate from any existing partial file (resume scenario).
        if output_path.exists():
            try:
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                self._samples = existing.get("per_sample_results", [])
            except (json.JSONDecodeError, OSError):
                self._samples = []

    def append(self, sample: dict[str, Any]) -> None:
        self._samples.append(sample)
        self._flush()

    def _flush(self) -> None:
        """Atomically write the current accumulated state to disk."""
        payload = {
            "run_metadata": self.run_metadata,
            "per_sample_results": self._samples,
        }
        tmp = self.output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.output_path)

    def finalize(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Write the final payload (including metrics/status) and return it."""
        payload = {
            "run_metadata": self.run_metadata,
            **extra,
            "per_sample_results": self._samples,
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @property
    def completed_questions(self) -> int:
        return len(self._samples)

    @property
    def completed_question_texts(self) -> set[str]:
        return {s.get("question", "") for s in self._samples}


# ---------------------------------------------------------------------------
# BioASQ-aware BenchmarkRunner (same adapter as in run_step4_benchmark.py)
# ---------------------------------------------------------------------------

def _make_bioasq_runner(dataset_path: Path, llm_wrapper):
    """Return a BioASQBenchmarkRunner with BioASQ JSON normalisation."""
    from evaluation.mirage_runner import BenchmarkRunner

    class BioASQBenchmarkRunner(BenchmarkRunner):
        """BenchmarkRunner subclass that loads BioASQ JSON format.

        Normalises BioASQ field names ('body', 'ideal_answer') to the
        'question' / 'answer' keys that run_rag_evaluation expects.
        """

        def load_dataset(self):
            if not self.dataset_path or not self.dataset_path.exists():
                return []
            with open(self.dataset_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("questions", list(raw.values())[0])
            normalised = []
            for item in raw:
                answer = item.get("ideal_answer") or item.get("exact_answer") or ""
                if isinstance(answer, list):
                    answer = " ".join(str(a) for a in answer)
                normalised.append({
                    "question": item.get("body", item.get("question", "")),
                    "answer":   answer,
                    "_id":      item.get("id", ""),
                    "_type":    item.get("type", ""),
                })
            logger.info(
                f"[BioASQBenchmarkRunner] Loaded {len(normalised)} questions "
                f"from {self.dataset_path.name}"
            )
            return normalised

    return BioASQBenchmarkRunner(dataset_path=dataset_path, llm_wrapper=llm_wrapper)


# ---------------------------------------------------------------------------
# Per-question pipeline runners (called one question at a time for checkpointing)
# ---------------------------------------------------------------------------

async def _run_single_rag_question(
    question_item: dict[str, Any],
    runner,
    index_dir: Path,
    pipeline,
    **kwargs,
) -> dict[str, Any]:
    """Run the RAG evaluation pipeline for a single question.

    Returns the per_sample dict.  On GPU OOM, returns an error record instead
    of raising — so one bad question never kills the whole config run.
    """
    import torch

    q_text = question_item.get("question", "")
    try:
        result = await runner.run_rag_evaluation(
            questions_override=[question_item],
            max_samples=1,
            index_dir=index_dir,
            generator_temperature=0.0,
            retrieval_provider=pipeline.retrieve_evidence,
            **kwargs,
        )
        samples = result.get("per_sample_results", [])
        return samples[0] if samples else {"question": q_text, "error": "no_result"}
    except torch.cuda.OutOfMemoryError as oom:
        logger.error(f"[Matrix] GPU OOM on question '{q_text[:60]}': {oom}")
        torch.cuda.empty_cache()
        return {"question": q_text, "error": "gpu_oom", "error_detail": str(oom)}
    except Exception as exc:
        logger.error(f"[Matrix] Unexpected error on question '{q_text[:60]}': {exc}")
        return {"question": q_text, "error": "exception", "error_detail": str(exc)}


def _run_single_plain_llm_question(
    question_item: dict[str, Any],
    llm_wrapper,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run the plain-LLM (no-RAG) path for a single question."""
    import torch
    from evaluation.plain_llm_runner import run_plain_llm_batch

    q_text = question_item.get("question", "")
    try:
        result = run_plain_llm_batch(
            questions=[question_item],
            llm_wrapper=llm_wrapper,
            temperature=temperature,
        )
        samples = result.get("per_sample_results", [])
        return samples[0] if samples else {"question": q_text, "error": "no_result"}
    except torch.cuda.OutOfMemoryError as oom:
        logger.error(f"[Matrix] GPU OOM (plain-llm) on question '{q_text[:60]}': {oom}")
        import torch as _torch
        _torch.cuda.empty_cache()
        return {"question": q_text, "error": "gpu_oom", "error_detail": str(oom)}
    except Exception as exc:
        logger.error(f"[Matrix] Unexpected error (plain-llm) on question '{q_text[:60]}': {exc}")
        return {"question": q_text, "error": "exception", "error_detail": str(exc)}


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    settings = get_settings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Print runtime estimate ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4 config-matrix runner — runtime estimate")
    print("=" * 65)
    print("  full-pipeline          : 30 × 120s = 3600s  (~60 min)")
    print("  no-critic              : 30 ×  70s = 2100s  (~35 min)")
    print("  no-iterative-retrieval : 30 ×  70s = 2100s  (~35 min)")
    print("  no-calibration         : 30 × 120s = 3600s  (~60 min)")
    print("  basic-rag              : 30 ×  50s = 1500s  (~25 min)")
    print("  plain-llm-no-rag       : 30 ×  15s =  450s  ( ~8 min)")
    print("  ─────────────────────────────────────────────────────")
    print("  Total                  :              ~223 min (~3.7 h)")
    print("=" * 65 + "\n")

    # ── 0. Safety guard: verify eval-split index BEFORE touching anything ─
    _checkpoint_log("Pre-flight: resolving and verifying FAISS index dir …")
    eval_split_dir = _resolve_and_guard_index_dir(None)
    if eval_split_dir != _CANONICAL_EVAL_SPLIT_DIR:
        raise RuntimeError(
            f"[Matrix] SAFETY GUARD: index_dir ({eval_split_dir}) != "
            f"_CANONICAL_EVAL_SPLIT_DIR ({_CANONICAL_EVAL_SPLIT_DIR}). "
            "Aborting — will not resume into wrong index."
        )
    _checkpoint_log(f"Index dir verified: {eval_split_dir}")

    # ── 1. Load or init checkpoint ────────────────────────────────────────
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_checkpoint()
    if state is None:
        run_id = timestamp
        state = _init_checkpoint(run_id)
        _save_checkpoint(state)
        _checkpoint_log(f"New checkpoint created: run_id={run_id}")
    else:
        run_id = state.get("run_id", timestamp)
        _checkpoint_log(
            f"Existing checkpoint loaded: run_id={run_id}, "
            f"configs={json.dumps({k: v['status'] for k, v in state['configs'].items()})}"
        )

    # ── 2. Load Qwen3-8B ONCE ─────────────────────────────────────────────
    from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper, ModelLoadError

    _checkpoint_log(
        f"About to load LLM '{settings.llm_model_name}' "
        f"(mode=LOCAL, allow_fallback=False). Checkpoint state: "
        f"{json.dumps({k: v['status'] for k, v in state['configs'].items()})}"
    )
    logger.info(
        f"[Matrix] Eager-loading LLM '{settings.llm_model_name}' "
        f"(mode=LOCAL, allow_fallback=False) …"
    )
    llm_wrapper = LocalLLMWrapper(
        model_name_or_path=settings.llm_model_name,
        mode=LLMRuntimeMode.LOCAL,
        allow_fallback=False,
    )
    t_llm_start = time.perf_counter()
    llm_wrapper.load_model()  # raises ModelLoadError on failure — intentional
    model_load_ms = int((time.perf_counter() - t_llm_start) * 1000)

    # Hard assertion: must be LOCAL after load.
    assert llm_wrapper.active_runtime_mode == LLMRuntimeMode.LOCAL, (
        f"Expected LOCAL runtime mode after load_model(), got "
        f"'{llm_wrapper.active_runtime_mode.value}'.  Aborting."
    )
    state["llm_loaded"] = True
    state["index_dir_verified"] = str(eval_split_dir)
    _save_checkpoint(state)
    logger.info(
        f"[Matrix] LLM loaded in {model_load_ms} ms — "
        f"active_runtime_mode={llm_wrapper.active_runtime_mode.value}"
    )

    # ── 3. Build RAGPipeline ONCE ─────────────────────────────────────────
    from app.rag.pipeline import RAGPipeline

    logger.info(f"[Matrix] Building RAGPipeline from: {eval_split_dir}")
    pipeline = RAGPipeline.build(
        index_dir=eval_split_dir,
        top_k_retrieval=10,
        top_k_final=3,
    )
    logger.info("[Matrix] RAGPipeline ready.")

    # ── 4. Load dataset and extract first 30 questions ────────────────────
    dataset_path = _DEFAULT_DATASET_PATH
    runner_for_load = _make_bioasq_runner(dataset_path, llm_wrapper)
    all_questions = runner_for_load.load_dataset()
    first_30_questions = all_questions[:_N_QUESTIONS]
    logger.info(f"[Matrix] Dataset loaded — using first {len(first_30_questions)} questions.")

    _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 5. Config definitions ─────────────────────────────────────────────
    # Maps config_name → kwargs for _run_rag (None = plain-llm path).
    _rag_kwargs: dict[str, dict | None] = {
        "full-pipeline":           {},
        "no-critic":               {"disable_critic": True},
        "no-iterative-retrieval":  {"max_retrieval_iterations": 1},
        "no-calibration":          {"disable_calibration": True},
        "basic-rag":               {"max_retrieval_iterations": 1,
                                    "disable_critic": True,
                                    "disable_calibration": True},
        "plain-llm-no-rag":        None,  # special path
    }

    # ── 6. Run all configs sequentially ───────────────────────────────────
    for config_name in _CONFIG_NAMES:
        cfg_state = state["configs"][config_name]

        # ── 6a. Skip verified-done configs ────────────────────────────────
        if cfg_state["status"] == "done":
            if _verify_done_config(config_name, state):
                _checkpoint_log(
                    f"Skipping config '{config_name}' — already done and verified "
                    f"({_N_QUESTIONS}/{_N_QUESTIONS} questions, "
                    f"output={cfg_state['output_file']})"
                )
                continue
            else:
                _checkpoint_log(
                    f"Config '{config_name}' marked done but failed integrity check — "
                    "treating as failed, will rerun from scratch."
                )
                cfg_state["status"] = "failed"
                cfg_state["completed_questions"] = 0
                _save_checkpoint(state)

        print(f"\n{'─' * 60}")
        print(f"[Matrix] Starting config: {config_name}")
        t_cfg_start = time.perf_counter()

        # ── 6b. Determine output path (stable per run_id, not per restart) ─
        output_path = _DEFAULT_OUTPUT_DIR / f"step4_{config_name}_n30_{run_id}.json"

        # ── 6c. Resume: load already-completed samples ────────────────────
        already_done_samples = _load_partial_output(str(output_path))
        already_done_texts = {s.get("question", "") for s in already_done_samples}
        n_already = len(already_done_samples)

        if cfg_state["status"] in ("running", "failed") and n_already > 0:
            _checkpoint_log(
                f"Resuming config '{config_name}' from question "
                f"{n_already + 1}/{_N_QUESTIONS} "
                f"({n_already} already complete per checkpoint)"
            )
        else:
            _checkpoint_log(
                f"Starting config '{config_name}' from question 1/{_N_QUESTIONS}. "
                f"Checkpoint state: {cfg_state['status']}"
            )

        # ── 6d. Mark as running ───────────────────────────────────────────
        cfg_state["status"] = "running"
        cfg_state["output_file"] = str(output_path)
        _save_checkpoint(state)

        # Safety guard for RAG configs (not applicable to plain-llm)
        is_plain_llm = (config_name == "plain-llm-no-rag")
        if not is_plain_llm:
            if eval_split_dir != _CANONICAL_EVAL_SPLIT_DIR:
                raise RuntimeError(
                    f"[Matrix] SAFETY GUARD FAILED for config '{config_name}': "
                    f"index_dir ({eval_split_dir}) != _CANONICAL_EVAL_SPLIT_DIR "
                    f"({_CANONICAL_EVAL_SPLIT_DIR}).  Aborting."
                )
        else:
            print("[Matrix] plain-llm-no-rag: no index guard (no retrieval)")

        # ── 6e. Build run_metadata for this config ────────────────────────
        run_metadata = {
            "config_name":           config_name,
            "model_load_ms":         model_load_ms,
            "generator_temperature": "N/A" if is_plain_llm else 0.0,
            "index_dir":             "N/A" if is_plain_llm else str(eval_split_dir),
            "n_questions":           _N_QUESTIONS,
            "completed_at":          None,  # filled in at finalize
            "llm_model":             settings.llm_model_name,
            "llm_mode":              llm_wrapper.active_runtime_mode.value,
        }

        # ── 6f. IncrementalResultWriter (pre-seeded with already-done samples) ─
        writer = IncrementalResultWriter(output_path, run_metadata)
        # Pre-seed from recovered samples so writer.completed_question_texts is correct.
        for s in already_done_samples:
            if s.get("question", "") not in writer.completed_question_texts:
                writer._samples.append(s)

        # ── 6g. Run each question ─────────────────────────────────────────
        runner = _make_bioasq_runner(dataset_path, llm_wrapper)
        rag_kw = _rag_kwargs[config_name]

        for qi, question_item in enumerate(first_30_questions, start=1):
            q_text = question_item.get("question", "")

            # Skip already-completed questions (match by text, not index)
            if q_text in writer.completed_question_texts:
                continue

            _checkpoint_log(
                f"Config '{config_name}' Q{qi}/{_N_QUESTIONS}: about to run. "
                f"checkpoint_state: completed={writer.completed_questions}"
            )

            if is_plain_llm:
                sample = _run_single_plain_llm_question(question_item, llm_wrapper)
            else:
                sample = await _run_single_rag_question(
                    question_item, runner, eval_split_dir, pipeline, **rag_kw
                )

            # Append immediately (atomic flush inside append)
            writer.append(sample)

            # Update checkpoint counter
            cfg_state["completed_questions"] = writer.completed_questions
            _save_checkpoint(state)

            status_str = f"error={sample.get('error')}" if "error" in sample else "ok"
            logger.info(
                f"[Matrix] Config '{config_name}' Q{qi}/{_N_QUESTIONS} "
                f"done — {status_str}, total_done={writer.completed_questions}"
            )

        # ── 6h. Finalize config ───────────────────────────────────────────
        elapsed = int(time.perf_counter() - t_cfg_start)
        run_metadata["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Compute metrics from all samples (mirrors original matrix runner)
        all_samples = writer._samples
        n_errored = sum(1 for s in all_samples if "error" in s)
        rouge_scores = [
            s.get("rouge_l_f1", s.get("rouge_l", 0.0))
            for s in all_samples
            if "error" not in s and s.get("rouge_l_f1") is not None
        ]
        rouge_l_mean = round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0.0

        extra = {
            "status": "complete",
            "samples_evaluated": len(all_samples),
            "metrics": {"rouge_l_f1": rouge_l_mean},
            "timing": {"total_elapsed_s": elapsed},
        }
        writer.finalize(extra)

        n_q = len(all_samples)
        print(
            f"CONFIG {config_name} DONE: {n_q} questions "
            f"({n_errored} errored), {elapsed}s, rouge_l_f1={rouge_l_mean}"
        )
        logger.info(f"[Matrix] Output written to: {output_path}")

        # ── 6i. Mark config done in checkpoint ────────────────────────────
        cfg_state["status"] = "done"
        cfg_state["completed_questions"] = n_q
        _save_checkpoint(state)
        _checkpoint_log(
            f"Config '{config_name}' marked done. "
            f"output={output_path}, errored_samples={n_errored}"
        )

    # ── 7. Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4 config-matrix run COMPLETE.")
    print(f"  Outputs written to: {_DEFAULT_OUTPUT_DIR}")
    print(f"  Checkpoint file:    {_CHECKPOINT_FILE}")
    print("=" * 65 + "\n")
    _checkpoint_log("All configs complete. Matrix run finished.")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
