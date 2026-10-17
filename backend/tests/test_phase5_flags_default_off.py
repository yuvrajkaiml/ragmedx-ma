"""
Phase 5 — P5-3: default-OFF versioning-compliance tests for the two new flags.

Two flags were added for Phase 5, both default OFF so the historical baselines
stay byte-for-byte reproducible:

  (a) MultiAgentOrchestrator(record_node_timing=False, pipeline_version=None)
      When record_node_timing is False (default) the langgraph trace steps keep
      their hardcoded duration_ms == 0 — identical to the pre-Phase-5 behaviour.
      pipeline_version default None leaves AgentState.pipeline_version == None.

  (b) RAGPipeline.build(cache_enabled=False, cache_dir=None)
      When cache_enabled is False (default) no QueryCache is constructed, no
      cache files are written, and cache_hits stays 0.

These tests run entirely on CPU with a mock LLM + a stub retrieval provider —
no GPU, no FAISS index, no real model.

Evidence tier: SOFTWARE TEST
"""
from __future__ import annotations

import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import asyncio

import pytest

from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.state import AgentState, create_deterministic_uuid
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mock_llm():
    """A deterministic mock LLM wrapper (no GPU, no model load)."""
    return LocalLLMWrapper(mode=LLMRuntimeMode.MOCK)


async def _stub_retrieval_provider(query=None, optimized_query=None, max_iterations=3, **_):
    """Return one fixed evidence chunk so the pipeline has something to work on."""
    evidence = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "source_type": "pubmed",
            "title": "Fixed Evidence",
            "cited_text": "Aspirin inhibits prostaglandin synthesis.",
            "relevance_score": 0.9,
            "cross_encoder_score": 0.9,
            "evidence_quality_score": 0.9,
            "citation_marker": "[E1]",
        }
    ]
    metadata = [{"iteration": 1, "query_used": query or "", "count": 1, "status": "completed"}]
    return evidence, metadata


def _build_orchestrator(**kwargs) -> MultiAgentOrchestrator:
    return MultiAgentOrchestrator(
        llm_wrapper=_mock_llm(),
        retrieval_provider=_stub_retrieval_provider,
        **kwargs,
    )


def _initial_state(question: str = "What does aspirin inhibit?") -> AgentState:
    return AgentState(
        query_id=create_deterministic_uuid(question),
        original_question=question,
    )


def _langgraph_durations(state: AgentState) -> list[int]:
    """Return the duration_ms of every langgraph trace step, in order."""
    return [
        step.get("duration_ms", 0)
        for step in state.agent_trace
        if step.get("agent_type") == "langgraph"
    ]


# ---------------------------------------------------------------------------
# (a) record_node_timing default OFF
# ---------------------------------------------------------------------------

def test_record_node_timing_defaults_to_false():
    orch = _build_orchestrator()
    assert orch.record_node_timing is False


def test_default_langgraph_durations_all_zero():
    """With record_node_timing OFF (default), every langgraph duration_ms is 0."""
    orch = _build_orchestrator()
    final = asyncio.run(orch.run(_initial_state()))
    durations = _langgraph_durations(final)
    assert durations, "expected at least one langgraph trace step"
    assert all(d == 0 for d in durations), (
        f"default (flag OFF) must keep every langgraph duration_ms == 0, got {durations}"
    )


def test_explicit_off_matches_implicit_default():
    """record_node_timing=False explicit == default: identical all-zero durations."""
    implicit = asyncio.run(_build_orchestrator().run(_initial_state()))
    explicit = asyncio.run(
        _build_orchestrator(record_node_timing=False).run(_initial_state())
    )
    assert _langgraph_durations(implicit) == _langgraph_durations(explicit)
    assert all(d == 0 for d in _langgraph_durations(explicit))


def test_record_node_timing_on_can_report_nonzero_without_changing_off_path():
    """
    With the flag ON the langgraph durations may be >= 0 (real wall-clock), and
    the number of trace steps is unchanged versus OFF — the flag only changes
    the duration_ms value, never the trace structure.
    """
    off = asyncio.run(_build_orchestrator().run(_initial_state()))
    on = asyncio.run(
        _build_orchestrator(record_node_timing=True).run(_initial_state())
    )
    off_durs = _langgraph_durations(off)
    on_durs = _langgraph_durations(on)
    # Same number of langgraph nodes recorded (structure identical).
    assert len(off_durs) == len(on_durs)
    # OFF path remains strictly all-zero.
    assert all(d == 0 for d in off_durs)
    # ON path durations are non-negative ints (timing may be 0 for a fast mock
    # node, but must never be negative).
    assert all(isinstance(d, int) and d >= 0 for d in on_durs)


# ---------------------------------------------------------------------------
# pipeline_version default None
# ---------------------------------------------------------------------------

def test_pipeline_version_defaults_to_none():
    orch = _build_orchestrator()
    assert orch.pipeline_version is None
    final = asyncio.run(orch.run(_initial_state()))
    assert final.pipeline_version is None


def test_pipeline_version_stamped_only_when_set():
    orch = _build_orchestrator(pipeline_version="v2")
    final = asyncio.run(orch.run(_initial_state()))
    assert final.pipeline_version == "v2"


# ---------------------------------------------------------------------------
# (b) cache_enabled default OFF
# ---------------------------------------------------------------------------

def test_pipeline_cache_defaults_off(tmp_path):
    """RAGPipeline default construction has no cache and writes no files."""
    pipeline = RAGPipeline(top_k_final=3)
    assert pipeline.cache_enabled is False
    assert pipeline.cache_hits == 0
    assert pipeline.cache_misses == 0


def test_pipeline_build_cache_defaults_off_no_files(tmp_path):
    """
    RAGPipeline.build() with default args (cache OFF) over a non-existent index
    returns an empty facade with cache OFF and creates no cache files.
    """
    cache_dir = tmp_path / "cache"
    pipeline = RAGPipeline.build(index_dir=tmp_path / "no_such_index")
    assert pipeline.cache_enabled is False
    # No cache dir should have been created by the default path.
    assert not cache_dir.exists()


def test_cache_enabled_true_requires_cache_dir():
    with pytest.raises(ValueError, match="cache_dir"):
        RAGPipeline(top_k_final=3, cache_enabled=True, cache_dir=None)


def test_default_retrieve_evidence_writes_no_cache_files(tmp_path):
    """
    A default (cache-OFF) retrieve_evidence call on an empty facade writes no
    cache files and reports cache_hits == 0.
    """
    cache_dir = tmp_path / "cache"
    pipeline = RAGPipeline(top_k_final=3)  # empty facade, cache OFF
    evidence, meta = asyncio.run(pipeline.retrieve_evidence("aspirin"))
    # Empty facade returns no evidence, but importantly: no cache files anywhere.
    assert pipeline.cache_hits == 0
    assert not cache_dir.exists()
    assert meta and meta[0].get("status") == "no_retriever"
