"""Phase 5 — P5-4/P5-6: 8-config experiment runner (GPU).

Runs the Phase 5 split (built by build_phase5_split.py) through 8 configs,
producing rich per-sample records with latency, VRAM, and provenance.

GPU interpreter ONLY:
    d:\\Ragmed_2\\.venv-cu128-pilot\\Scripts\\python.exe backend\\scripts\\run_phase5_experiment.py [opts]

The 8 configs
-------------
  phase5-improved-v2            all new flags ON (claim verification + cache +
                                iterative retrieval + critic + calibration),
                                pipeline_version="v2", new experiment ID.
  phase5-improved-no-verification
                                improved minus claim verification
                                (cache ON, enable_claim_verification=False).
  phase5-full-pipeline          historical baseline, new flags OFF.
  phase5-no-critic              historical baseline (disable_critic).
  phase5-no-iterative-retrieval historical baseline (max_retrieval_iterations=1).
  phase5-no-calibration         historical baseline (disable_calibration).
  phase5-basic-rag              historical baseline (single pass, no critic/calib).
  phase5-plain-llm-no-rag       historical baseline (plain LLM, no retrieval).

Historical baselines keep all new flags OFF so they stay reproducible.

Per-sample fields (absent → "NOT MEASURED" + reason)
----------------------------------------------------
  question_id, resolved_index_path, split, answer, citations, claim_verdicts,
  confidence, abstain_flag, errors, per_stage_latency_ms, model_call_count,
  retrieval_call_count, cache_hits, peak_vram_mb.

Safety
------
  * Hard-asserts llm_mode == LOCAL (no mock fallback).
  * IndexLeakageGuard + _resolve_and_guard_index_dir assert eval_split before
    every RAG run.
  * Never touches the production index.

Tiering (P5-6)
--------------
  --max-questions N  : run only N dev + N test questions per config
  --smoke            : 3 dev questions per config (quick P5-4 verification)
  --full             : the full dev+test sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import BASE_DIR, get_settings  # noqa: E402
from app.utils.logger import logger  # noqa: E402
from evaluation.split_builder import IndexLeakageGuard  # noqa: E402
from scripts.run_step4_benchmark import (  # noqa: E402
    _CANONICAL_EVAL_SPLIT_DIR,
    _resolve_and_guard_index_dir,
)

_RESULTS_ROOT = _BACKEND_DIR / "results"
_NOT_MEASURED = "NOT MEASURED"

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------
# Each config maps to orchestrator/pipeline kwargs. "new_flags" indicates the
# Phase 5 additions (record_node_timing, pipeline_version, cache_enabled).
# Historical baselines keep new flags OFF.

_CONFIGS: Dict[str, Dict[str, Any]] = {
    "phase5-improved-v2": {
        "kind": "rag",
        "orchestrator": {
            "enable_claim_verification": True,
            "record_node_timing": True,
            "pipeline_version": "v2",
        },
        "max_retrieval_iterations": 3,
        "cache_enabled": True,
        "experiment_id": "phase5-improved-v2",
    },
    "phase5-improved-no-verification": {
        "kind": "rag",
        "orchestrator": {
            "enable_claim_verification": False,
            "record_node_timing": True,
            "pipeline_version": "v2",
        },
        "max_retrieval_iterations": 3,
        "cache_enabled": True,
        "experiment_id": "phase5-improved-no-verification",
    },
    "phase5-full-pipeline": {
        "kind": "rag",
        "orchestrator": {},
        "max_retrieval_iterations": 3,
        "cache_enabled": False,
        "experiment_id": "phase5-full-pipeline",
    },
    "phase5-no-critic": {
        "kind": "rag",
        "orchestrator": {"disable_critic": True},
        "max_retrieval_iterations": 3,
        "cache_enabled": False,
        "experiment_id": "phase5-no-critic",
    },
    "phase5-no-iterative-retrieval": {
        "kind": "rag",
        "orchestrator": {},
        "max_retrieval_iterations": 1,
        "cache_enabled": False,
        "experiment_id": "phase5-no-iterative-retrieval",
    },
    "phase5-no-calibration": {
        "kind": "rag",
        "orchestrator": {"disable_calibration": True},
        "max_retrieval_iterations": 3,
        "cache_enabled": False,
        "experiment_id": "phase5-no-calibration",
    },
    "phase5-basic-rag": {
        "kind": "rag",
        "orchestrator": {"disable_critic": True, "disable_calibration": True},
        "max_retrieval_iterations": 1,
        "cache_enabled": False,
        "experiment_id": "phase5-basic-rag",
    },
    "phase5-plain-llm-no-rag": {
        "kind": "plain",
        "orchestrator": {},
        "max_retrieval_iterations": 1,
        "cache_enabled": False,
        "experiment_id": "phase5-plain-llm-no-rag",
    },
}

_GENERATOR_TEMPERATURE = 0.0  # deterministic, benchmark-only override

# The 16 validated pilot question IDs (frozen; see context.json / FEAT-001 manifest).
# Used to seed the reusable pilot cells into a new run dir without re-running them.
_PILOT_REUSED_IDS_16 = [
    "pubmedqa::15943725",
    "bioasq::602825a61cb411341a0000fa",
    "bioasq::53372ba3d6d3ac6a34000058",
    "bioasq::54d7a1047035c08008000002",
    "pubmedqa::21394762",
    "pubmedqa::16735905",
    "pubmedqa::26215326",
    "pubmedqa::23321509",
    "pubmedqa::24449622",
    "pubmedqa::22251324",
    "pubmedqa::18719011",
    "pubmedqa::10548670",
    "pubmedqa::20605051",
    "bioasq::6060996394d57fd879000045",
    "pubmedqa::12098035",
    "pubmedqa::23621776",
]


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------

def _reset_peak_vram() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:
        pass
    return False


def _peak_vram_mb() -> Any:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass
    return _NOT_MEASURED


# ---------------------------------------------------------------------------
# Per-stage latency extraction from agent_trace
# ---------------------------------------------------------------------------

_STAGE_NODES = [
    "query_understanding",
    "retrieval",
    "gap_analysis",
    "evidence_reranking",
    "generator",
    "critic",
    "confidence",
    "abstention",
]


def _per_stage_latency(agent_trace: List[Dict[str, Any]], timing_recorded: bool) -> Dict[str, Any]:
    """Sum langgraph node durations per stage from the trace.

    When record_node_timing was OFF the langgraph durations are hardcoded 0;
    we report NOT MEASURED for those so a reader never mistakes 0 for 'instant'.
    """
    if not timing_recorded:
        return {stage: _NOT_MEASURED for stage in _STAGE_NODES} | {
            "_reason": "record_node_timing was OFF (historical baseline reproducibility)"
        }
    sums: Dict[str, float] = {stage: 0.0 for stage in _STAGE_NODES}
    for step in agent_trace:
        if step.get("agent_type") == "langgraph":
            node = (step.get("details") or {}).get("node")
            if node in sums:
                sums[node] += float(step.get("duration_ms", 0) or 0)
    return {k: round(v, 2) for k, v in sums.items()}


def _count_calls(agent_trace: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (model_call_count, retrieval_call_count) from the trace."""
    retrieval = 0
    model = 0
    for step in agent_trace:
        at = step.get("agent_type")
        details = step.get("details") or {}
        if at == "iterative_retrieval" and details.get("action") in ("native_retrieval",):
            retrieval += 1
        # Generator and critic represent model calls; query_understanding and
        # gap_analysis also call the LLM. Count non-langgraph agent steps that
        # are completed as model-involving calls where applicable.
        if at in ("generator", "critic", "query_understanding", "iterative_retrieval",
                  "evidence_reranking") and step.get("status") == "completed":
            if at != "iterative_retrieval":
                model += 1
    return model, retrieval


# ---------------------------------------------------------------------------
# Record extraction from final AgentState
# ---------------------------------------------------------------------------

def _extract_citations(evidence: List[Dict[str, Any]], answer_citations: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in evidence or []:
        if not isinstance(ev, dict):
            continue
        out.append({
            "citation_marker": ev.get("citation_marker"),
            "title": ev.get("title"),
            "document_id": ev.get("document_id"),
            "index_id": ev.get("index_id"),
            "pmid": ev.get("pmid") or ev.get("url"),
        })
    return out


def _build_record(
    *,
    question_item: Dict[str, Any],
    split_name: str,
    resolved_index_path: str,
    config_name: str,
    config_def: Dict[str, Any],
    final_state: Any,
    peak_vram: Any,
    cache_hits: Any,
    cold_latency_ms: int,
    warm_latency_ms: Any,
    error: Any = None,
) -> Dict[str, Any]:
    orch_kwargs = config_def.get("orchestrator", {})
    timing_recorded = bool(orch_kwargs.get("record_node_timing", False))
    critic_enabled = not orch_kwargs.get("disable_critic", False)
    calibration_enabled = not orch_kwargs.get("disable_calibration", False)
    verification_enabled = bool(orch_kwargs.get("enable_claim_verification", False))

    if error is not None or final_state is None:
        return {
            "question_id": question_item.get("id"),
            "resolved_index_path": resolved_index_path,
            "split": split_name,
            "config": config_name,
            "answer": _NOT_MEASURED,
            "citations": _NOT_MEASURED,
            "claim_verdicts": _NOT_MEASURED,
            "confidence": _NOT_MEASURED,
            "abstain_flag": _NOT_MEASURED,
            "errors": str(error) if error else "no_final_state",
            "per_stage_latency_ms": _NOT_MEASURED,
            "model_call_count": _NOT_MEASURED,
            "retrieval_call_count": _NOT_MEASURED,
            "cache_hits": cache_hits,
            "peak_vram_mb": peak_vram,
            "cold_latency_ms": cold_latency_ms,
            "warm_latency_ms": warm_latency_ms,
        }

    agent_trace = getattr(final_state, "agent_trace", []) or []
    model_calls, trace_retrieval_calls = _count_calls(agent_trace)
    # The runner performs one pre-retrieval pass before the orchestrator reuses
    # that evidence, so the real retrieval-call count is that pass plus any
    # additional native passes the orchestrator logged.
    retrieval_calls = 1 + trace_retrieval_calls

    # confidence
    calib = getattr(final_state, "confidence_calibration", None) or {}
    if calibration_enabled:
        confidence = calib.get("calibrated_confidence", _NOT_MEASURED)
        confidence_note = "calibrated"
    else:
        confidence = calib.get("calibrated_confidence", 1.0)
        confidence_note = "SYNTHETIC 1.0 (calibration disabled) — not a calibrated score"

    # claim verdicts
    if verification_enabled:
        er = getattr(final_state, "evidence_reranking", None) or {}
        cv = er.get("claim_verification")
        claim_verdicts = cv if cv is not None else getattr(final_state, "critic_verdict", None)
    elif critic_enabled:
        claim_verdicts = {
            "critic_verdict": getattr(final_state, "critic_verdict", None),
            "note": "LLM critic verdict (claim verification OFF)",
        }
    else:
        claim_verdicts = {"value": _NOT_MEASURED, "reason": "critic + claim verification disabled"}

    abstain_flag = getattr(final_state, "status", "") in ("abstained", "critic_rejected", "failed")
    ad = getattr(final_state, "abstention_decision", None) or {}
    if ad.get("should_abstain"):
        abstain_flag = True

    return {
        "question_id": question_item.get("id"),
        "resolved_index_path": resolved_index_path,
        "split": split_name,
        "config": config_name,
        "source": question_item.get("source"),
        "gold_answer": question_item.get("answer"),
        "answer": getattr(final_state, "answer", None) or "",
        "status": getattr(final_state, "status", None),
        "citations": _extract_citations(
            getattr(final_state, "evidence", []), getattr(final_state, "answer_citations", [])
        ),
        "claim_verdicts": claim_verdicts,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "abstain_flag": bool(abstain_flag),
        "errors": getattr(final_state, "error", None),
        "error_code": getattr(final_state, "error_code", None),
        "per_stage_latency_ms": _per_stage_latency(agent_trace, timing_recorded),
        "model_call_count": model_calls,
        "retrieval_call_count": retrieval_calls,
        "cache_hits": cache_hits,
        "peak_vram_mb": peak_vram,
        "cold_latency_ms": cold_latency_ms,
        "warm_latency_ms": warm_latency_ms,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_rag_question(
    *,
    question_item: Dict[str, Any],
    split_name: str,
    index_dir: Path,
    pipeline,
    config_name: str,
    config_def: Dict[str, Any],
    llm_wrapper,
    measure_warm: bool,
) -> Dict[str, Any]:
    from app.agents import run_agent_pipeline

    orch_kwargs = dict(config_def.get("orchestrator", {}))
    max_iters = config_def.get("max_retrieval_iterations", 3)
    question = question_item.get("question", "")

    _reset_peak_vram()
    cache_hits_before = getattr(pipeline, "cache_hits", 0)

    # Cold pass: pre-retrieve once (cache miss on first touch for cached configs).
    t_cold = time.perf_counter()
    try:
        raw_evidence, _ = await pipeline.retrieve_evidence(question)
        final_state = await _run_orchestrator(
            question=question,
            evidence=raw_evidence,
            llm_wrapper=llm_wrapper,
            retrieval_provider=pipeline.retrieve_evidence,
            orch_kwargs=orch_kwargs,
            max_iters=max_iters,
        )
        cold_ms = int((time.perf_counter() - t_cold) * 1000)
    except Exception as exc:
        logger.error("[P5] RAG error on %s/%s: %s", config_name, question[:50], exc)
        return _build_record(
            question_item=question_item, split_name=split_name,
            resolved_index_path=str(index_dir), config_name=config_name,
            config_def=config_def, final_state=None, peak_vram=_peak_vram_mb(),
            cache_hits=0, cold_latency_ms=int((time.perf_counter() - t_cold) * 1000),
            warm_latency_ms=_NOT_MEASURED, error=exc,
        )

    # Warm pass (cache-enabled configs only): a repeat retrieval should hit cache.
    warm_ms: Any = _NOT_MEASURED
    if measure_warm and getattr(pipeline, "cache_enabled", False):
        t_warm = time.perf_counter()
        try:
            await pipeline.retrieve_evidence(question)
            warm_ms = int((time.perf_counter() - t_warm) * 1000)
        except Exception:
            warm_ms = _NOT_MEASURED

    cache_hits = getattr(pipeline, "cache_hits", 0) - cache_hits_before
    peak_vram = _peak_vram_mb()

    return _build_record(
        question_item=question_item, split_name=split_name,
        resolved_index_path=str(index_dir), config_name=config_name,
        config_def=config_def, final_state=final_state, peak_vram=peak_vram,
        cache_hits=cache_hits, cold_latency_ms=cold_ms, warm_latency_ms=warm_ms,
    )


async def _run_orchestrator(*, question, evidence, llm_wrapper, retrieval_provider, orch_kwargs, max_iters):
    """Run the orchestrator directly and return the full final AgentState."""
    from app.agents.orchestrator import MultiAgentOrchestrator
    from app.agents.state import AgentState, create_deterministic_uuid

    orchestrator = MultiAgentOrchestrator(
        llm_wrapper=llm_wrapper,
        retrieval_provider=retrieval_provider,
        generator_temperature=_GENERATOR_TEMPERATURE,
        max_retrieval_iterations=max_iters,
        **orch_kwargs,
    )
    state = AgentState(
        query_id=create_deterministic_uuid(question or "query"),
        original_question=question,
        evidence=evidence or [],
    )
    return await orchestrator.run(state)


def _run_plain_question(
    *,
    question_item: Dict[str, Any],
    split_name: str,
    config_name: str,
    config_def: Dict[str, Any],
    llm_wrapper,
) -> Dict[str, Any]:
    from evaluation.plain_llm_runner import run_plain_llm

    question = question_item.get("question", "")
    _reset_peak_vram()
    t0 = time.perf_counter()
    try:
        result = run_plain_llm(question, llm_wrapper, temperature=_GENERATOR_TEMPERATURE)
        cold_ms = int((time.perf_counter() - t0) * 1000)
        answer = result.get("predicted_answer", "")
        err = None
    except Exception as exc:
        cold_ms = int((time.perf_counter() - t0) * 1000)
        answer = ""
        err = exc
    peak_vram = _peak_vram_mb()
    return {
        "question_id": question_item.get("id"),
        "resolved_index_path": "N/A (plain-llm-no-rag)",
        "split": split_name,
        "config": config_name,
        "source": question_item.get("source"),
        "gold_answer": question_item.get("answer"),
        "answer": answer,
        "status": "verified" if not err else "error",
        "citations": [],
        "claim_verdicts": {"value": _NOT_MEASURED, "reason": "no RAG / no critic in plain-llm"},
        "confidence": _NOT_MEASURED,
        "confidence_note": "no calibration in plain-llm path",
        "abstain_flag": False,
        "errors": str(err) if err else None,
        "per_stage_latency_ms": {_s: _NOT_MEASURED for _s in _STAGE_NODES} | {
            "_reason": "plain-llm-no-rag has no multi-agent stages"
        },
        "model_call_count": 1,
        "retrieval_call_count": 0,
        "cache_hits": 0,
        "peak_vram_mb": peak_vram,
        "cold_latency_ms": cold_ms,
        "warm_latency_ms": _NOT_MEASURED,
    }


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_split(out_root: Path, name: str) -> List[Dict[str, Any]]:
    path = out_root / "splits" / f"{name}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _stratified_sample(records, n, seed, stratify_by="source"):
    """Return a seeded, source-stratified sample of ~n records from `records`.

    Allocation is proportional to each stratum's share of the pool; a fixed
    random seed makes the selection reproducible. Order within strata is
    shuffled with the same seed.
    """
    import random as _random

    rng = _random.Random(seed)
    strata: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        strata.setdefault(r.get(stratify_by, "__unknown__"), []).append(r)

    total = len(records)
    picked: List[Dict[str, Any]] = []
    for key in sorted(strata.keys(), key=str):
        pool = strata[key][:]
        rng.shuffle(pool)
        take = max(1, round(n * len(strata[key]) / total)) if total else 0
        picked.extend(pool[:take])

    # Trim/pad to exactly n (trim from the end; pad from leftover unused records).
    if len(picked) > n:
        picked = picked[:n]
    elif len(picked) < n:
        chosen_ids = {id(x) for x in picked}
        leftovers = [r for r in records if id(r) not in chosen_ids]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])
    rng.shuffle(picked)
    return picked


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-cell checkpoint / resume (FEAT-002)
# ---------------------------------------------------------------------------
#
# Granularity is a single matrix cell (question_id, config_id). The on-disk
# layout mirrors the pilot run dir exactly: per_question/<config>.jsonl holds
# one JSON record per completed cell; a sidecar checkpoints/phase5_matrix_state.json
# maps cell_key -> {state, output_file, attempts, updated_at}. All state writes
# are atomic (write .tmp then os.replace), following run_step4_matrix.py.

_CHECKPOINT_FILENAME = "phase5_matrix_state.json"
_MAX_CELL_ATTEMPTS = 3  # 1 initial try + up to 2 retries (design §2 bounded retries)

# Required keys for a per-cell JSONL record to be accepted as reusable.
_REQUIRED_RECORD_KEYS = (
    "question_id",
    "resolved_index_path",
    "split",
    "config",
    "answer",
    "citations",
    "claim_verdicts",
    "confidence",
    "abstain_flag",
    "errors",
)


def _cell_key(question_id: str, config_id: str) -> str:
    """Stable, dedup-/resume-unit key for a single matrix cell."""
    return f"{question_id}::{config_id}"


def _checkpoint_path(out_root: Path) -> Path:
    return out_root / "checkpoints" / _CHECKPOINT_FILENAME


def _load_checkpoint(out_root: Path) -> Dict[str, Any]:
    """Return the persisted cell-state map, or a fresh skeleton if absent/corrupt.

    A corrupt/unreadable state file is treated as "no prior state" so the run
    simply re-validates the JSONL files and re-runs anything unproven — it never
    silently trusts a broken checkpoint.
    """
    path = _checkpoint_path(out_root)
    if not path.exists():
        return {"version": 1, "cells": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "cells" not in data:
            raise ValueError("missing 'cells'")
        return data
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("[P5] checkpoint unreadable (%s); starting from JSONL re-validation.", exc)
        return {"version": 1, "cells": {}}


def _save_checkpoint(out_root: Path, state: Dict[str, Any]) -> None:
    """Atomically persist the cell-state map (write .tmp then os.replace)."""
    path = _checkpoint_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.error("[P5] checkpoint write failed: %s", exc)


def _mark_cell(
    out_root: Path,
    state: Dict[str, Any],
    key: str,
    cell_state: str,
    *,
    output_file: Optional[str] = None,
    attempts: Optional[int] = None,
) -> None:
    """Update one cell's state and flush atomically."""
    entry = state["cells"].get(key, {})
    entry["state"] = cell_state
    if output_file is not None:
        entry["output_file"] = output_file
    if attempts is not None:
        entry["attempts"] = attempts
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["cells"][key] = entry
    _save_checkpoint(out_root, state)


def _validate_record(
    record: Any,
    *,
    question_id: str,
    config_id: str,
    allowed_ids: set,
) -> bool:
    """Return True iff *record* is a trustworthy completed cell for (qid, config).

    Checks (design §2 validation-on-resume):
      (a) parses to a dict with all required schema keys,
      (b) question_id matches and is in the frozen set; config matches an 8-config name,
      (c) resolved_index_path ends with 'eval_split' (or is the plain-llm sentinel),
      (d) exactly one of the answered / abstain / error branches holds.
    Any failure -> not reusable -> the cell is re-run.
    """
    if not isinstance(record, dict):
        return False
    for k in _REQUIRED_RECORD_KEYS:
        if k not in record:
            return False
    if record.get("question_id") != question_id or question_id not in allowed_ids:
        return False
    if record.get("config") != config_id or config_id not in _CONFIGS:
        return False

    resolved = str(record.get("resolved_index_path", ""))
    is_plain = config_id == "phase5-plain-llm-no-rag"
    if is_plain:
        if not resolved.startswith("N/A"):
            return False
    else:
        # Normalise trailing separators then require the eval_split leaf.
        norm = resolved.replace("\\", "/").rstrip("/")
        if not norm.endswith("eval_split"):
            return False

    # Exactly one of: answered / abstain / error.
    has_error = bool(record.get("errors"))
    abstained = record.get("abstain_flag") is True
    answer = record.get("answer")
    # An answer is present when it is a non-empty str or a non-empty list (the
    # plain-llm path stores list answers) and is not the NOT MEASURED sentinel.
    if isinstance(answer, str):
        has_answer = answer not in ("", _NOT_MEASURED)
    elif isinstance(answer, (list, tuple)):
        has_answer = len(answer) > 0
    else:
        has_answer = False
    answered = not has_error and not abstained and has_answer
    return (int(answered) + int(abstained) + int(has_error)) == 1


def _load_question_ids(path: Path) -> List[str]:
    """Load a frozen question-ID list (a JSON array of strings)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise SystemExit(f"[P5] {path} is not a JSON array of question-id strings.")
    return data


def _select_question_records(
    pool: List[Dict[str, Any]],
    question_ids: List[str],
) -> List[Dict[str, Any]]:
    """Return pool records matching *question_ids*, preserving the ID-list order.

    Raises if any requested ID is absent from the pool so a typo'd/altered ID
    list can never silently shrink the matrix.
    """
    by_id = {r.get("id"): r for r in pool}
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise SystemExit(
            f"[P5] {len(missing)} requested question id(s) not in the split pool: {missing[:5]}"
        )
    return [by_id[qid] for qid in question_ids]


def _existing_question_ids(path: Path) -> set:
    """Return the set of question_ids already present in a per_question JSONL file."""
    ids: set = set()
    for rec in _read_jsonl(path):
        qid = rec.get("question_id")
        if qid is not None:
            ids.add(qid)
    return ids


def _append_cell_guarded(out_path: Path, record: Dict[str, Any]) -> bool:
    """Append *record* only if its question_id is not already in the file.

    Returns True if written, False if a line for that question_id already
    existed (duplicate prevention). The file is <=50 lines/config so the
    re-scan is cheap and guards against a duplicate after a crash between the
    append and the state flush.
    """
    qid = record.get("question_id")
    if qid in _existing_question_ids(out_path):
        logger.warning(
            "[P5] refusing duplicate append for question_id=%s in %s", qid, out_path.name
        )
        return False
    _append_jsonl(out_path, record)
    return True


def _seed_pilot_cells(
    *,
    out_root: Path,
    pilot_dir: Path,
    question_ids: List[str],
    pilot_reused_ids: List[str],
    allowed_ids: set,
    state: Dict[str, Any],
) -> int:
    """Copy validated pilot JSONL lines for reusable cells into the new run dir.

    For each pilot id that is in the chosen question set, copy that id's line
    from the pilot's per_question/<config>.jsonl into the new run's file (deduped
    by question_id) for every config, re-validating each line before accepting,
    and mark the cell completed in the checkpoint. Returns the number of cells
    seeded.
    """
    chosen = set(question_ids)
    reusable_ids = [qid for qid in pilot_reused_ids if qid in chosen]
    per_q_src = pilot_dir / "per_question"
    per_q_dst = out_root / "per_question"
    per_q_dst.mkdir(parents=True, exist_ok=True)

    seeded = 0
    for config_id in _CONFIGS:
        src = per_q_src / f"{config_id}.jsonl"
        dst = per_q_dst / f"{config_id}.jsonl"
        src_by_id = {r.get("question_id"): r for r in _read_jsonl(src)}
        for qid in reusable_ids:
            rec = src_by_id.get(qid)
            if rec is None:
                logger.warning(
                    "[P5][seed] pilot line missing for %s/%s; cell will be run fresh.",
                    config_id, qid,
                )
                continue
            if not _validate_record(
                rec, question_id=qid, config_id=config_id, allowed_ids=allowed_ids
            ):
                logger.warning(
                    "[P5][seed] pilot line for %s/%s failed re-validation; cell will be run fresh.",
                    config_id, qid,
                )
                continue
            if _append_cell_guarded(dst, rec):
                _mark_cell(
                    out_root, state, _cell_key(qid, config_id), "completed",
                    output_file=str(dst), attempts=0,
                )
                seeded += 1
    logger.info("[P5][seed] seeded %d reused pilot cells from %s", seeded, pilot_dir.name)
    return seeded


def plan_matrix(
    *,
    out_root: Path,
    question_ids: List[str],
    allowed_ids: set,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Partition the full matrix into reused (completed+validated) vs missing cells.

    A cell counts as reused only if the checkpoint marks it completed AND its
    JSONL record re-validates. Everything else (missing / running / failed /
    invalid) is scheduled to run. Pure bookkeeping — no model work.
    """
    per_q_dir = out_root / "per_question"
    # Cache per-config id->record maps so we re-scan each file once.
    rec_cache: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for config_id in _CONFIGS:
        rec_cache[config_id] = {
            r.get("question_id"): r for r in _read_jsonl(per_q_dir / f"{config_id}.jsonl")
        }

    reused: List[str] = []
    missing: List[str] = []
    for qid in question_ids:
        for config_id in _CONFIGS:
            key = _cell_key(qid, config_id)
            entry = state["cells"].get(key, {})
            rec = rec_cache[config_id].get(qid)
            if (
                entry.get("state") == "completed"
                and rec is not None
                and _validate_record(
                    rec, question_id=qid, config_id=config_id, allowed_ids=allowed_ids
                )
            ):
                reused.append(key)
            else:
                missing.append(key)
    return {
        "total": len(question_ids) * len(_CONFIGS),
        "reused": reused,
        "missing": missing,
        "n_reused": len(reused),
        "n_missing": len(missing),
    }


async def run_matrix(
    *,
    out_root: Path,
    question_records: List[Dict[str, Any]],
    allowed_ids: set,
    state: Dict[str, Any],
    cell_runner,
    budget_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Run only the missing cells of the question-OUTER / config-INNER matrix.

    *cell_runner* is an async callable ``(question_item, split_name, config_id,
    config_def) -> record`` so this orchestration is CPU-testable with a stub
    (no GPU/model required). Each cell:
      1. is skipped if the checkpoint marks it completed AND its record validates;
      2. is marked ``running`` (atomic) before execution;
      3. retries exceptions up to ``_MAX_CELL_ATTEMPTS`` with a short backoff,
         then records a ``failed`` cell with answer="NOT MEASURED" (never dropped);
      4. appends its record guarded against duplicates, then marks ``completed``
         (abstentions stay distinct from errors).
    Returns run counters.
    """
    per_q_dir = out_root / "per_question"
    per_q_dir.mkdir(parents=True, exist_ok=True)
    config_order = list(_CONFIGS.keys())

    counts = {"completed": 0, "failed": 0, "abstained": 0, "reused_skipped": 0, "duplicate_skipped": 0}
    run_start = time.perf_counter()
    per_question_wall: List[float] = []
    budget_stopped = False
    questions_completed = 0

    for qi, q in enumerate(question_records, start=1):
        qid = q.get("id")
        split_name = q.get("split", "test")

        if budget_s is not None and per_question_wall:
            elapsed = time.perf_counter() - run_start
            mean_q = sum(per_question_wall) / len(per_question_wall)
            if elapsed + mean_q > budget_s:
                logger.warning(
                    "[P5][budget] Stopping before question %d: elapsed %.0fs + mean/q %.0fs "
                    "> budget %.0fs.", qi, elapsed, mean_q, budget_s,
                )
                budget_stopped = True
                break

        q_start = time.perf_counter()
        for config_id in config_order:
            cdef = _CONFIGS[config_id]
            out_path = per_q_dir / f"{config_id}.jsonl"
            key = _cell_key(qid, config_id)
            entry = state["cells"].get(key, {})

            # 1. Skip only validated-completed cells.
            if entry.get("state") == "completed":
                rec = {r.get("question_id"): r for r in _read_jsonl(out_path)}.get(qid)
                if rec is not None and _validate_record(
                    rec, question_id=qid, config_id=config_id, allowed_ids=allowed_ids
                ):
                    counts["reused_skipped"] += 1
                    continue
                logger.warning(
                    "[P5] cell %s marked completed but failed validation; re-running.", key
                )

            attempts_done = int(entry.get("attempts", 0))
            record: Optional[Dict[str, Any]] = None
            last_exc: Optional[Exception] = None

            # 2 + 3. Mark running, run with bounded retries.
            while attempts_done < _MAX_CELL_ATTEMPTS:
                attempts_done += 1
                _mark_cell(out_root, state, key, "running", output_file=str(out_path),
                           attempts=attempts_done)
                try:
                    record = await cell_runner(
                        question_item=q, split_name=split_name,
                        config_id=config_id, config_def=cdef,
                    )
                    last_exc = None
                    break
                except Exception as exc:  # transient GPU/IO hiccup — retry
                    last_exc = exc
                    logger.error(
                        "[P5] cell %s attempt %d/%d raised: %s",
                        key, attempts_done, _MAX_CELL_ATTEMPTS, exc,
                    )
                    if attempts_done < _MAX_CELL_ATTEMPTS:
                        time.sleep(min(2.0 * attempts_done, 10.0))

            if record is None:
                # 3. Exhausted retries -> failed cell, not dropped.
                record = _failed_record(
                    question_item=q, split_name=split_name, config_id=config_id,
                    error=last_exc,
                )

            # 4. Guarded append + state transition.
            written = _append_cell_guarded(out_path, record)
            if not written:
                counts["duplicate_skipped"] += 1
                # A line already exists for this cell; trust it, mark completed.
                _mark_cell(out_root, state, key, "completed", output_file=str(out_path),
                           attempts=attempts_done)
                continue

            if record.get("errors"):
                _mark_cell(out_root, state, key, "failed", output_file=str(out_path),
                           attempts=attempts_done)
                counts["failed"] += 1
            else:
                _mark_cell(out_root, state, key, "completed", output_file=str(out_path),
                           attempts=attempts_done)
                counts["completed"] += 1
                if record.get("abstain_flag") is True:
                    counts["abstained"] += 1

        per_question_wall.append(time.perf_counter() - q_start)
        questions_completed += 1

    counts["questions_completed"] = questions_completed
    counts["budget_stopped_early"] = budget_stopped
    counts["run_seconds"] = round(time.perf_counter() - run_start, 1)
    return counts


def _failed_record(
    *,
    question_item: Dict[str, Any],
    split_name: str,
    config_id: str,
    error: Any,
) -> Dict[str, Any]:
    """Build a NOT MEASURED failure record for a cell that exhausted retries.

    Distinct from an abstention: errors are set, abstain_flag is False.
    """
    is_plain = config_id == "phase5-plain-llm-no-rag"
    return {
        "question_id": question_item.get("id"),
        "resolved_index_path": "N/A (plain-llm-no-rag)" if is_plain else _NOT_MEASURED,
        "split": split_name,
        "config": config_id,
        "source": question_item.get("source"),
        "gold_answer": question_item.get("answer"),
        "answer": _NOT_MEASURED,
        "status": "error",
        "citations": _NOT_MEASURED,
        "claim_verdicts": _NOT_MEASURED,
        "confidence": _NOT_MEASURED,
        "confidence_note": "execution error after exhausting retries",
        "abstain_flag": False,
        "errors": str(error) if error else "unknown_error",
        "error_code": "CELL_RETRIES_EXHAUSTED",
        "per_stage_latency_ms": _NOT_MEASURED,
        "model_call_count": _NOT_MEASURED,
        "retrieval_call_count": _NOT_MEASURED,
        "cache_hits": _NOT_MEASURED,
        "peak_vram_mb": _NOT_MEASURED,
        "cold_latency_ms": _NOT_MEASURED,
        "warm_latency_ms": _NOT_MEASURED,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_gpu_cell_runner(*, index_dir, base_pipeline, cached_pipeline, llm):
    """Return an async cell_runner bound to the loaded model + pipelines.

    Preserves the exact per-cell behaviour of the legacy loop: cached vs base
    pipeline by config, _run_plain_question for the plain-llm config, and
    _run_rag_question (temperature 0.0, retrieval params, prompts) otherwise.
    """

    async def _cell_runner(*, question_item, split_name, config_id, config_def):
        if config_def["kind"] == "plain":
            return _run_plain_question(
                question_item=question_item, split_name=split_name,
                config_name=config_id, config_def=config_def, llm_wrapper=llm,
            )
        pipeline = cached_pipeline if config_def.get("cache_enabled") else base_pipeline
        return await _run_rag_question(
            question_item=question_item, split_name=split_name, index_dir=index_dir,
            pipeline=pipeline, config_name=config_id, config_def=config_def,
            llm_wrapper=llm, measure_warm=True,
        )

    return _cell_runner


async def _run_target_matrix(args: argparse.Namespace) -> None:
    """FEAT-002 target mode: 50/40 x 8 matrix with per-cell checkpoint + pilot reuse.

    Writes to a NEW backend/results/phase5_<ts>/ dir, seeds the validated pilot
    cells, and runs only the missing cells. Never touches the pilot dir.
    """
    settings = get_settings()

    # Resolve + guard eval_split path (unchanged guard).
    index_dir = _resolve_and_guard_index_dir(None)
    guard = IndexLeakageGuard()
    guard.assert_safe(str(index_dir))
    if index_dir != _CANONICAL_EVAL_SPLIT_DIR:
        raise RuntimeError(
            f"[P5] index_dir ({index_dir}) != canonical eval_split ({_CANONICAL_EVAL_SPLIT_DIR})"
        )
    logger.info("[P5] eval_split guard PASS: %s", index_dir)

    # Resolve the frozen pool (pilot splits/test.json) and the chosen ID list.
    pilot_dir = Path(args.pilot_dir) if args.pilot_dir else (_RESULTS_ROOT / "phase5_20261017_131123")
    pool = _load_split(pilot_dir, "test")

    if args.question_ids:
        ids_path = Path(args.question_ids)
        question_ids = _load_question_ids(ids_path)
    else:
        # Pull from the frozen manifest (FEAT-001) by target.
        manifest_path = _RESULTS_ROOT / "phase5_manifest" / "experiment_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"[P5] No --question-ids given and manifest missing: {manifest_path}. "
                "Run build_phase5_manifest.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = "question_ids_40" if args.target == 40 else "question_ids_50"
        question_ids = manifest[key]
    allowed_ids = set(question_ids)
    question_records = _select_question_records(pool, question_ids)
    for r in question_records:
        r.setdefault("split", "test")

    # Pilot reuse ids: prefer the manifest's list, fall back to frozen context.
    pilot_reused_ids = _PILOT_REUSED_IDS_16

    # NEW timestamped output dir (never the pilot dir).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else (_RESULTS_ROOT / f"phase5_{ts}")
    if out_root.resolve() == pilot_dir.resolve():
        raise SystemExit("[P5] REFUSING to write into the read-only pilot dir.")
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("[P5] target=%d questions -> new run dir %s", len(question_ids), out_root)

    # Copy the frozen split pool into the new run dir so aggregation + reproducibility
    # have the same layout as the pilot (never writes back to the pilot dir).
    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_root / "splits" / "test.json", pool)

    # Write config defs.
    configs_dir = out_root / "configs"
    for name, cdef in _CONFIGS.items():
        _atomic_write_json(configs_dir / f"{name}.json", {"config_name": name, **cdef})

    # Load (or init) checkpoint, then seed pilot cells once.
    state = _load_checkpoint(out_root)
    already_seeded = any(e.get("attempts") == 0 for e in state["cells"].values())
    if not already_seeded:
        _seed_pilot_cells(
            out_root=out_root, pilot_dir=pilot_dir, question_ids=question_ids,
            pilot_reused_ids=pilot_reused_ids, allowed_ids=allowed_ids, state=state,
        )

    plan = plan_matrix(
        out_root=out_root, question_ids=question_ids, allowed_ids=allowed_ids, state=state,
    )
    logger.info("[P5] matrix plan: total=%d reused=%d missing=%d",
                plan["total"], plan["n_reused"], plan["n_missing"])

    if args.dry_run:
        print(
            f"PHASE5_DRYRUN target={args.target} total={plan['total']} "
            f"reused={plan['n_reused']} missing={plan['n_missing']} root={out_root}"
        )
        return

    # --- Real GPU run from here (load model once, build pipelines once) ----
    from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper

    llm = LocalLLMWrapper(
        model_name_or_path=settings.llm_model_name,
        mode=LLMRuntimeMode.LOCAL,
        allow_fallback=False,
    )
    t_llm = time.perf_counter()
    llm.load_model()
    model_load_ms = int((time.perf_counter() - t_llm) * 1000)
    if llm.active_runtime_mode != LLMRuntimeMode.LOCAL:
        raise RuntimeError(
            f"[P5] llm_mode is '{llm.active_runtime_mode}', not LOCAL. Aborting (no mock fallback)."
        )

    from app.rag.pipeline import RAGPipeline

    base_pipeline = RAGPipeline.build(index_dir=index_dir, top_k_retrieval=10, top_k_final=3)
    cached_pipeline = RAGPipeline.build(
        index_dir=index_dir, top_k_retrieval=10, top_k_final=3,
        cache_enabled=True, cache_dir=str(out_root / "query_cache"),
    )
    cell_runner = _make_gpu_cell_runner(
        index_dir=index_dir, base_pipeline=base_pipeline,
        cached_pipeline=cached_pipeline, llm=llm,
    )

    budget_s = args.budget_hours * 3600.0 if args.budget_hours else None
    counts = await run_matrix(
        out_root=out_root, question_records=question_records, allowed_ids=allowed_ids,
        state=state, cell_runner=cell_runner, budget_s=budget_s,
    )

    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolved_eval_split_path": str(index_dir),
        "canonical_eval_split": str(_CANONICAL_EVAL_SPLIT_DIR),
        "llm_model": settings.llm_model_name,
        "llm_mode": llm.active_runtime_mode.value,
        "generator_temperature": _GENERATOR_TEMPERATURE,
        "model_load_ms": model_load_ms,
        "configs": list(_CONFIGS.keys()),
        "mode": "target_matrix",
        "target": args.target,
        "n_questions": len(question_ids),
        "matrix": {"plan": {"total": plan["total"], "reused": plan["n_reused"],
                            "missing": plan["n_missing"]}, "run_counts": counts},
        "subset": {"dev": 0, "test": len(question_ids), "full": False},
    }
    try:
        import torch
        environment["torch_version"] = torch.__version__
        environment["cuda_version"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            environment["peak_vram_mb_run"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 2
            )
    except Exception:
        pass
    _atomic_write_json(out_root / "environment_run.json", environment)
    aggregate_outputs(out_root, environment)
    logger.info("[P5] target matrix complete.")
    print(f"PHASE5_RUN_COMPLETE root={out_root} counts={counts}")


async def _amain(args: argparse.Namespace) -> None:
    settings = get_settings()

    # Resolve + guard eval_split path.
    index_dir = _resolve_and_guard_index_dir(None)
    guard = IndexLeakageGuard()
    guard.assert_safe(str(index_dir))
    if index_dir != _CANONICAL_EVAL_SPLIT_DIR:
        raise RuntimeError(
            f"[P5] index_dir ({index_dir}) != canonical eval_split ({_CANONICAL_EVAL_SPLIT_DIR})"
        )
    logger.info("[P5] eval_split guard PASS: %s", index_dir)

    # Resolve output root (must contain splits built by build_phase5_split.py).
    out_root = Path(args.output_root) if args.output_root else None
    if out_root is None:
        # Pick the most recent phase5_* dir with a splits/ subdir.
        candidates = sorted(_RESULTS_ROOT.glob("phase5_*"), reverse=True)
        for c in candidates:
            if (c / "splits" / "test.json").exists():
                out_root = c
                break
    if out_root is None or not (out_root / "splits" / "test.json").exists():
        raise SystemExit("[P5] No Phase 5 split found. Run build_phase5_split.py first.")
    logger.info("[P5] Using Phase 5 output root: %s", out_root)

    # Load LLM once in strict LOCAL mode (no mock fallback).
    from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper

    llm = LocalLLMWrapper(
        model_name_or_path=settings.llm_model_name,
        mode=LLMRuntimeMode.LOCAL,
        allow_fallback=False,
    )
    t_llm = time.perf_counter()
    llm.load_model()
    model_load_ms = int((time.perf_counter() - t_llm) * 1000)
    if llm.active_runtime_mode != LLMRuntimeMode.LOCAL:
        raise RuntimeError(
            f"[P5] llm_mode is '{llm.active_runtime_mode}', not LOCAL. Aborting (no mock fallback allowed)."
        )
    logger.info("[P5] LLM loaded in %d ms, mode=%s", model_load_ms, llm.active_runtime_mode.value)

    # Determine subset sizes.
    if args.full:
        dev_n = test_n = None
    elif args.smoke:
        dev_n, test_n = args.smoke_n, 0
    else:
        dev_n = test_n = args.max_questions

    dev = _load_split(out_root, "dev")
    test = _load_split(out_root, "test")

    if args.pilot_n is not None:
        # Budget-bounded pilot: a stratified (by source), seeded sample drawn
        # from the TEST split only. dev is not used in pilot mode.
        pilot = _stratified_sample(test, n=args.pilot_n, seed=args.pilot_seed, stratify_by="source")
        questions = [("test", q) for q in pilot]
        dev, test = [], pilot
        logger.info("[P5][pilot] %d test questions (seed=%d, stratified by source) over %d configs",
                    len(pilot), args.pilot_seed, len(_CONFIGS))
    else:
        if dev_n is not None:
            dev = dev[:dev_n]
        if test_n is not None:
            test = test[:test_n]
        questions = [("dev", q) for q in dev] + [("test", q) for q in test]
        logger.info("[P5] Running %d questions/config (dev=%d test=%d) over %d configs",
                    len(questions), len(dev), len(test), len(_CONFIGS))

    # Build base pipeline (cache OFF) and a cached pipeline (cache ON) once.
    from app.rag.pipeline import RAGPipeline

    base_pipeline = RAGPipeline.build(index_dir=index_dir, top_k_retrieval=10, top_k_final=3)
    cache_dir = out_root / "query_cache"
    cached_pipeline = RAGPipeline.build(
        index_dir=index_dir, top_k_retrieval=10, top_k_final=3,
        cache_enabled=True, cache_dir=str(cache_dir),
    )

    # Write config defs + environment.
    configs_dir = out_root / "configs"
    for name, cdef in _CONFIGS.items():
        _atomic_write_json(configs_dir / f"{name}.json", {"config_name": name, **cdef})

    per_q_dir = out_root / "per_question"
    config_order = list(_CONFIGS.keys())

    # Fresh output files for this invocation.
    for config_name in config_order:
        out_path = per_q_dir / f"{config_name}.jsonl"
        if out_path.exists():
            out_path.unlink()

    # Run question-OUTER so a wall-clock budget stop leaves every config with
    # the same completed question set (comparable). Each question is run through
    # all 8 configs before advancing.
    counts: Dict[str, int] = {c: 0 for c in config_order}
    budget_s = args.budget_hours * 3600.0 if args.budget_hours else None
    run_start = time.perf_counter()
    per_question_wall: List[float] = []
    budget_stopped = False
    questions_completed = 0

    for qi, (split_name, q) in enumerate(questions, start=1):
        # Budget guard: before starting a new question, stop if the elapsed time
        # plus the observed mean per-question wall-clock would exceed the budget.
        if budget_s is not None and per_question_wall:
            elapsed = time.perf_counter() - run_start
            mean_q = sum(per_question_wall) / len(per_question_wall)
            if elapsed + mean_q > budget_s:
                logger.warning(
                    "[P5][budget] Stopping before question %d: elapsed %.0fs + mean/q %.0fs "
                    "> budget %.0fs. Completed %d questions x %d configs.",
                    qi, elapsed, mean_q, budget_s, questions_completed, len(config_order),
                )
                budget_stopped = True
                break

        q_start = time.perf_counter()
        for config_name in config_order:
            cdef = _CONFIGS[config_name]
            out_path = per_q_dir / f"{config_name}.jsonl"
            pipeline = cached_pipeline if cdef.get("cache_enabled") else base_pipeline
            if cdef["kind"] == "plain":
                record = _run_plain_question(
                    question_item=q, split_name=split_name, config_name=config_name,
                    config_def=cdef, llm_wrapper=llm,
                )
            else:
                record = await _run_rag_question(
                    question_item=q, split_name=split_name, index_dir=index_dir,
                    pipeline=pipeline, config_name=config_name, config_def=cdef,
                    llm_wrapper=llm, measure_warm=True,
                )
            _append_jsonl(out_path, record)
            counts[config_name] += 1
        q_wall = time.perf_counter() - q_start
        per_question_wall.append(q_wall)
        questions_completed += 1
        logger.info("[P5] question %d/%d done across %d configs in %.0fs (elapsed %.0fs)",
                    qi, len(questions), len(config_order), q_wall,
                    time.perf_counter() - run_start)

    run_wall_s = round(time.perf_counter() - run_start, 1)

    environment = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "resolved_eval_split_path": str(index_dir),
        "canonical_eval_split": str(_CANONICAL_EVAL_SPLIT_DIR),
        "llm_model": settings.llm_model_name,
        "llm_mode": llm.active_runtime_mode.value,
        "generator_temperature": _GENERATOR_TEMPERATURE,
        "model_load_ms": model_load_ms,
        "configs": list(_CONFIGS.keys()),
        "subset": {"dev": len(dev), "test": len(test), "full": bool(args.full)},
        "records_per_config": counts,
        "mode": ("pilot" if args.pilot_n is not None else ("full" if args.full else "subset")),
        "pilot": None if args.pilot_n is None else {
            "requested_n": args.pilot_n,
            "seed": args.pilot_seed,
            "stratify_by": "source",
            "questions_completed": questions_completed,
            "budget_hours": args.budget_hours,
            "budget_stopped_early": budget_stopped,
        },
        "wall_clock": {
            "run_seconds": run_wall_s,
            "run_hours": round(run_wall_s / 3600.0, 3),
            "questions_completed": questions_completed,
            "mean_seconds_per_question_all_configs": (
                round(sum(per_question_wall) / len(per_question_wall), 1)
                if per_question_wall else None
            ),
        },
    }
    try:
        import torch
        environment["torch_version"] = torch.__version__
        environment["cuda_version"] = torch.version.cuda
        environment["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            environment["peak_vram_mb_run"] = round(
                torch.cuda.max_memory_allocated() / (1024 * 1024), 2
            )
    except Exception:
        pass
    _atomic_write_json(out_root / "environment_run.json", environment)
    logger.info("[P5] Wrote environment_run.json.")

    # P5-6: aggregate metrics, statistics, latency, failures, report.
    aggregate_outputs(out_root, environment)
    logger.info("[P5] Aggregation complete. DONE.")
    print(f"PHASE5_RUN_COMPLETE root={out_root} per_config={counts}")


# ---------------------------------------------------------------------------
# P5-6 aggregation (reads per_question/*.jsonl, writes the full output layout)
# ---------------------------------------------------------------------------

_BASELINE_CONFIG = "phase5-full-pipeline"


def _is_correct(answer: Any, gold: Any) -> Optional[bool]:
    """Token-containment correctness vs gold (BENCHMARK GROUND TRUTH tier).

    Returns None when the answer is NOT MEASURED / missing so it is excluded
    from the denominator rather than silently counted as wrong.
    """
    if answer in (None, "", _NOT_MEASURED):
        return False
    if gold in (None, "", _NOT_MEASURED):
        return None
    import re as _re
    a = set(_re.findall(r"\w+", str(answer).lower()))
    g = set(_re.findall(r"\w+", str(gold).lower()))
    if not g:
        return None
    # Correct if the gold key tokens are substantially present in the answer.
    overlap = len(a & g) / len(g)
    return overlap >= 0.6


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def aggregate_outputs(out_root: Path, environment: Dict[str, Any]) -> None:
    from evaluation.statistics import holm_correction, mcnemar_test, wilson_ci

    per_q_dir = out_root / "per_question"
    config_names = [c.stem for c in sorted(per_q_dir.glob("*.jsonl"))]

    # Load records + per-question correctness keyed by question_id.
    records_by_config: Dict[str, List[Dict[str, Any]]] = {}
    correct_by_config: Dict[str, Dict[Any, Optional[bool]]] = {}
    for cfg in config_names:
        recs = _read_jsonl(per_q_dir / f"{cfg}.jsonl")
        records_by_config[cfg] = recs
        cmap: Dict[Any, Optional[bool]] = {}
        for r in recs:
            cmap[r.get("question_id")] = _is_correct(r.get("answer"), r.get("gold_answer"))
        correct_by_config[cfg] = cmap

    # Per-config metric summary with Wilson CIs.
    metric_summary: Dict[str, Any] = {
        "evidence_tier": {
            "answer_correctness": "BENCHMARK GROUND TRUTH (token-containment vs gold, threshold 0.6)",
            "latency": "SOFTWARE TEST (wall-clock on this GPU run)",
            "abstention_rate": "SOFTWARE TEST",
        },
        "baseline_for_paired_tests": _BASELINE_CONFIG,
        "configs": {},
    }
    for cfg in config_names:
        cmap = correct_by_config[cfg]
        measured = [v for v in cmap.values() if v is not None]
        n = len(measured)
        k = sum(1 for v in measured if v)
        ci = wilson_ci(k, n) if n > 0 else [None, None]
        recs = records_by_config[cfg]
        abstained = sum(1 for r in recs if r.get("abstain_flag") is True)
        metric_summary["configs"][cfg] = {
            "n_measured": n,
            "n_total": len(recs),
            "accuracy_correct": k,
            "accuracy": round(k / n, 4) if n else _NOT_MEASURED,
            "accuracy_wilson_ci_95": list(ci),
            "abstention_rate": round(abstained / len(recs), 4) if recs else _NOT_MEASURED,
        }

    # Paired McNemar vs baseline + Holm correction.
    paired: Dict[str, Any] = {}
    p_values: List[float] = []
    p_labels: List[str] = []
    base_map = correct_by_config.get(_BASELINE_CONFIG, {})
    for cfg in config_names:
        if cfg == _BASELINE_CONFIG:
            continue
        cmap = correct_by_config[cfg]
        n00 = n01 = n10 = n11 = 0
        shared = 0
        for qid, base_val in base_map.items():
            cur_val = cmap.get(qid)
            if base_val is None or cur_val is None:
                continue
            shared += 1
            if base_val and cur_val:
                n11 += 1
            elif base_val and not cur_val:
                n10 += 1
            elif not base_val and cur_val:
                n01 += 1
            else:
                n00 += 1
        if shared == 0:
            paired[cfg] = {"note": "no shared measured questions", "vs": _BASELINE_CONFIG}
            continue
        stat, pval, method = mcnemar_test(((n00, n01), (n10, n11)))
        paired[cfg] = {
            "vs": _BASELINE_CONFIG,
            "table": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "shared_measured": shared,
            "mcnemar_statistic": stat,
            "p_value_raw": pval,
            "method": method,
        }
        p_values.append(pval)
        p_labels.append(cfg)

    if p_values:
        rejected, adjusted = holm_correction(p_values)
        for label, rej, adj in zip(p_labels, rejected, adjusted):
            paired[label]["p_value_holm_adjusted"] = adj
            paired[label]["significant_holm_0.05"] = bool(rej)
    metric_summary["paired_vs_baseline_mcnemar_holm"] = paired
    metric_summary["significance_caveat"] = (
        "No superiority claim is made without a Holm-adjusted significant paired test. "
        "Small pilot n yields wide CIs; treat deltas as directional only."
    )
    metric_summary["metric_caveat"] = (
        "Answer-correctness here is token-containment vs gold (threshold 0.6). It scores an "
        "abstention as incorrect, so configs with high abstention rates (the RAG baselines on "
        "this BioASQ/PubMedQA mix) are penalised relative to always-answering plain-LLM. This is "
        "a measurement property, NOT evidence that abstention is wrong — abstention can be the "
        "safe behaviour. Report abstention_rate alongside accuracy; do not read accuracy alone."
    )
    _atomic_write_json(out_root / "metric_summary.json", metric_summary)

    # Latency breakdown (per-stage, cold vs warm) + PRD targets.
    latency: Dict[str, Any] = {
        "evidence_tier": "SOFTWARE TEST (wall-clock on this run)",
        "prd_targets_ms": {
            "single_pass": "< 5000",
            "iterative_max_3_rounds": "< 15000",
            "retrieval": "< 500",
            "rerank": "< 2000",
            "note": "cold vs warm reported separately below",
        },
        "configs": {},
    }
    for cfg in config_names:
        recs = records_by_config[cfg]
        colds = [r.get("cold_latency_ms") for r in recs if isinstance(r.get("cold_latency_ms"), int)]
        warms = [r.get("warm_latency_ms") for r in recs if isinstance(r.get("warm_latency_ms"), int)]
        # Per-stage sums (only where measured as numeric).
        stage_acc: Dict[str, List[float]] = {s: [] for s in _STAGE_NODES}
        for r in recs:
            psl = r.get("per_stage_latency_ms")
            if isinstance(psl, dict):
                for s in _STAGE_NODES:
                    v = psl.get(s)
                    if isinstance(v, (int, float)):
                        stage_acc[s].append(float(v))
        stage_mean = {
            s: (round(sum(v) / len(v), 2) if v else _NOT_MEASURED) for s, v in stage_acc.items()
        }
        latency["configs"][cfg] = {
            "cold_latency_ms": {
                "n": len(colds),
                "mean": round(sum(colds) / len(colds), 2) if colds else _NOT_MEASURED,
                "min": min(colds) if colds else _NOT_MEASURED,
                "max": max(colds) if colds else _NOT_MEASURED,
            },
            "warm_latency_ms": {
                "n": len(warms),
                "mean": round(sum(warms) / len(warms), 2) if warms else _NOT_MEASURED,
            },
            "per_stage_mean_ms": stage_mean,
        }
    _atomic_write_json(out_root / "latency_breakdown.json", latency)

    # Failure cases.
    failures: List[Dict[str, Any]] = []
    for cfg in config_names:
        for r in records_by_config[cfg]:
            if r.get("errors") or r.get("status") in ("error", "failed"):
                failures.append({
                    "config": cfg,
                    "question_id": r.get("question_id"),
                    "status": r.get("status"),
                    "errors": r.get("errors"),
                    "error_code": r.get("error_code"),
                })
    _atomic_write_json(out_root / "failure_cases.json",
                       {"n_failures": len(failures), "cases": failures})

    # Summary + report.
    _mode = environment.get("mode", "subset")
    _tier = {
        "full": "FULL",
        "pilot": "BUDGET_BOUNDED_PILOT",
        "subset": "REDUCED_VALIDATION_SUBSET",
    }.get(_mode, "REDUCED_VALIDATION_SUBSET")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": _tier,
        "mode": _mode,
        "pilot": environment.get("pilot"),
        "wall_clock": environment.get("wall_clock"),
        "peak_vram_mb_run": environment.get("peak_vram_mb_run"),
        "subset": environment.get("subset"),
        "configs": config_names,
        "accuracy": {c: metric_summary["configs"][c]["accuracy"] for c in config_names},
        "n_failures": len(failures),
        "notes": "Every metric carries an evidence tier; NOT MEASURED used where data is absent. "
                 "No superiority claim without a Holm-significant paired test.",
    }
    _atomic_write_json(out_root / "summary.json", summary)
    _write_report_md(out_root, environment, metric_summary, latency, summary)


def _write_report_md(out_root, environment, metric_summary, latency, summary) -> None:
    lines: List[str] = []
    lines.append("# Phase 5 — 8-Config Experiment Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append(f"Tier: **{summary['tier']}**  "
                 f"(dev={summary['subset'].get('dev')}, test={summary['subset'].get('test')})")
    wc = summary.get("wall_clock") or {}
    if wc:
        lines.append("")
        lines.append(f"Wall-clock: {wc.get('run_hours')} h "
                     f"({wc.get('questions_completed')} questions × "
                     f"{len(summary.get('configs', []))} configs; "
                     f"mean {wc.get('mean_seconds_per_question_all_configs')} s/question-all-configs). "
                     f"Peak VRAM (run): {summary.get('peak_vram_mb_run')} MB.")
    pilot = summary.get("pilot") or {}
    if pilot:
        lines.append(f"Pilot: n={pilot.get('questions_completed')} test questions "
                     f"(seed={pilot.get('seed')}, stratified by source, "
                     f"budget={pilot.get('budget_hours')} h, "
                     f"stopped_early={pilot.get('budget_stopped_early')}).")
    lines.append("")
    lines.append(f"- Resolved eval_split: `{environment.get('resolved_eval_split_path')}`")
    lines.append(f"- LLM: `{environment.get('llm_model')}` mode=`{environment.get('llm_mode')}` "
                 f"temp={environment.get('generator_temperature')}")
    lines.append(f"- torch {environment.get('torch_version')} / CUDA {environment.get('cuda_version')}")
    lines.append("")
    lines.append("## Answer correctness (BENCHMARK GROUND TRUTH, token-containment ≥0.6)")
    lines.append("")
    lines.append("| Config | n | correct | accuracy | 95% Wilson CI | abstention |")
    lines.append("|---|---|---|---|---|---|")
    for cfg, m in metric_summary["configs"].items():
        ci = m["accuracy_wilson_ci_95"]
        ci_str = f"[{ci[0]}, {ci[1]}]" if ci and ci[0] is not None else "n/a"
        lines.append(f"| {cfg} | {m['n_measured']} | {m['accuracy_correct']} | "
                     f"{m['accuracy']} | {ci_str} | {m['abstention_rate']} |")
    lines.append("")
    lines.append(f"## Paired McNemar vs `{_BASELINE_CONFIG}` (Holm-corrected)")
    lines.append("")
    lines.append("| Config | shared n | p (raw) | p (Holm) | significant? | method |")
    lines.append("|---|---|---|---|---|---|")
    for cfg, p in metric_summary["paired_vs_baseline_mcnemar_holm"].items():
        if "p_value_raw" not in p:
            lines.append(f"| {cfg} | — | — | — | — | {p.get('note','')} |")
            continue
        lines.append(f"| {cfg} | {p['shared_measured']} | {p['p_value_raw']} | "
                     f"{p.get('p_value_holm_adjusted','—')} | "
                     f"{p.get('significant_holm_0.05','—')} | {p['method']} |")
    lines.append("")
    lines.append("## Latency (SOFTWARE TEST) vs PRD targets")
    lines.append("")
    lines.append("PRD: single-pass <5s; iterative ≤3 rounds <15s; retrieval <500ms; rerank <2s "
                 "(cold vs warm separate).")
    lines.append("")
    lines.append("| Config | cold mean (ms) | warm mean (ms) |")
    lines.append("|---|---|---|")
    for cfg, L in latency["configs"].items():
        lines.append(f"| {cfg} | {L['cold_latency_ms']['mean']} | {L['warm_latency_ms']['mean']} |")
    lines.append("")
    lines.append("## Honesty / limitations")
    lines.append("")
    lines.append("- Every metric is tagged with an evidence tier; `NOT MEASURED` is used where data "
                 "is genuinely absent (e.g. per-stage latency on historical baselines where "
                 "record_node_timing is OFF; PubMedQA retrieval-quality not in the eval index).")
    lines.append(f"- {metric_summary['significance_caveat']}")
    if metric_summary.get("metric_caveat"):
        lines.append(f"- {metric_summary['metric_caveat']}")
    lines.append("")
    lines.append("## Remaining full 500×8 sweep — NOT RUN (exceeds 4 h GPU budget)")
    lines.append("")
    lines.append("The full 500-question × 8-config sweep is estimated at ~40+ hours of GPU "
                 "wall-clock and was deliberately NOT run (it exceeds the 4-hour budget). "
                 "To run it later:")
    lines.append("")
    lines.append("```")
    lines.append(r"d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe "
                 r"backend\scripts\run_phase5_experiment.py --full "
                 r"--output-root " + str(out_root))
    lines.append("```")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 8-config experiment runner (GPU).")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Phase 5 output dir containing splits/ (default: latest phase5_*).")
    parser.add_argument("--max-questions", type=int, default=25,
                        help="Dev and test questions per config for the reduced tier (default 25).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: --smoke-n dev questions per config (default 1).")
    parser.add_argument("--smoke-n", type=int, default=1,
                        help="Number of dev questions per config in smoke mode (default 1).")
    parser.add_argument("--pilot-n", type=int, default=None,
                        help="Budget-bounded pilot: seeded, source-stratified sample of N test "
                             "questions x all 8 configs. Overrides smoke/max-questions.")
    parser.add_argument("--pilot-seed", type=int, default=42,
                        help="Seed for the stratified pilot sample (default 42).")
    parser.add_argument("--budget-hours", type=float, default=None,
                        help="Wall-clock GPU budget in hours. The run stops before starting a "
                             "new question if the projected finish would exceed it.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full dev+test sweep (LONG, ~40h — exceeds the 4h budget).")
    # FEAT-002 target-matrix mode (per-cell checkpoint + pilot reuse).
    parser.add_argument("--target", type=int, default=None, choices=(50, 40),
                        help="FEAT-002 matrix mode: run TARGET unique questions x 8 configs with "
                             "per-cell checkpoint/resume and pilot-cell reuse into a NEW run dir.")
    parser.add_argument("--question-ids", type=str, default=None,
                        help="Path to a frozen question-id JSON list (overrides --target's manifest "
                             "lookup). Enables target-matrix mode.")
    parser.add_argument("--pilot-dir", type=str, default=None,
                        help="Read-only pilot run dir to reuse validated cells from "
                             "(default: results/phase5_20261017_131123).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Target-matrix mode only: compute reused/missing cell counts (seed + "
                             "plan) without loading the model or running any cell.")
    args = parser.parse_args()

    # Dispatch: target-matrix mode (FEAT-002) vs the legacy pilot/subset/full runner.
    if args.target is not None or args.question_ids is not None or args.dry_run:
        if args.target is None:
            args.target = 50  # default when only --question-ids / --dry-run given
        asyncio.run(_run_target_matrix(args))
    else:
        asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
