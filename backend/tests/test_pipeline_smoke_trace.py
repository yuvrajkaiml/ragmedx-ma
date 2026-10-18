"""
RAGMedX-MA seven-agent LangGraph pipeline — SMOKE TEST + PER-STAGE TRACE suite.

This is a NEW, additive smoke-test module authored for a fast, reproducible
verification of the compiled MultiAgentOrchestrator graph. It does NOT edit
production code, prompts, generation settings, retrieval params, metric
definitions, or tau. It observes the pipeline from OUTSIDE by reading the
returned AgentState (``agent_trace`` + final-state fields).

Case labelling (also emitted into results.json):
  [MOCK-LLM + SYNTHETIC-EVIDENCE]  deterministic, fast; injected SequenceFakeLLM
                                   + in-memory synthetic evidence (no index touched).
  [REAL-LLM + SYNTHETIC-EVIDENCE]  one true end-to-end case using the local Qwen GPU
                                   model via LocalLLMWrapper(LOCAL) with the index-free
                                   synthetic retrieval stub. Gated behind the env var
                                   RAGMED_SMOKE_REAL_LLM=1 so the CPU run skips it and
                                   NEVER presents a mocked result as real-model output.

Safety notes:
  * No case reads/writes data/faiss_index/ or the eval split. Evidence is supplied
    in-memory via AgentState.evidence or the index-free temporary_retrieval_stub.
  * tau=0.65 is read from source and asserted, never written.
  * Instrumentation is read-only: include_trace=True only gates serialization; no
    behaviour-altering flags are set (all default OFF).

Run (fast, mock/synthetic — CPU venv):
    cd d:\\Ragmed_2\\backend
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m pytest tests/test_pipeline_smoke_trace.py -q

Run (single real-LLM case — GPU venv):
    cd d:\\Ragmed_2\\backend
    $env:RAGMED_SMOKE_REAL_LLM="1"
    d:\\Ragmed_2\\.venv-cu128-pilot\\Scripts\\python.exe -m pytest tests/test_pipeline_smoke_trace.py -q -k real_llm -s
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from app.agents.abstention_decision import (
    COPY_CONTRADICTORY_EVIDENCE,
    COPY_RETRIEVAL_FAILURE,
)
from app.agents.confidence_calibration import CONFIDENCE_THRESHOLD
from app.agents.orchestrator import (
    MAX_CRITIC_ROUNDS,
    MAX_RETRIEVAL_ITERATIONS,
    MultiAgentOrchestrator,
)
from app.agents.state import AgentState, AgentType, create_deterministic_uuid
from app.llm.temporary_retrieval_stub import retrieve_evidence as stub_retrieval_provider


# ==============================================================================
# Shared mock-LLM + results collection
# ==============================================================================
class SequenceFakeLLM:
    """Mock LLM returning responses in a predefined sequence.

    Reused verbatim pattern from tests/test_orchestrator.py so the smoke suite
    exercises the same injection boundary the project already relies on. Captures
    every prompt so we can verify (from outside) that this run's evidence actually
    reaches the generator/critic.
    """

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.prompts: List[str] = []

    def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}
        return "Default response [E1]", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}


# Canonical control-plane mock responses (JSON for QUA/MGA/ERA/Critic).
QUA_RESP = (
    '{"medical_specialty": "Cardiology", "intent": "treatment", '
    '"expanded_query": "essential hypertension first line treatment", '
    '"abbreviations": {}, "confidence": 0.95}'
)
GAP_SUFFICIENT = (
    '{"sufficient": true, "reasoning": "Retrieved evidence directly answers the question.", '
    '"missing_information": [], "sub_query": null, "confidence": 0.95}'
)
GAP_INSUFFICIENT = (
    '{"sufficient": false, "reasoning": "First pass missed dosing detail.", '
    '"missing_information": ["dosing"], "sub_query": "first-line antihypertensive dosing", '
    '"confidence": 0.6}'
)
RERANK_OK = '{"contradictory": false, "selected_markers": [], "reasoning": "Sources agree."}'
RERANK_CONTRADICTORY = '{"contradictory": true, "selected_markers": [], "reasoning": "Sources materially conflict."}'
CRITIC_PASS = (
    '{"passed": true, "factual_consistency": 0.95, "citation_support": 0.92, '
    '"hallucination_risk": 0.05, "issues": [], "revision_instructions": null, "confidence": 0.95}'
)
# Citation EXISTS (marker resolves) but does NOT support the claim: low citation_support.
CRITIC_CITATION_UNSUPPORTED = (
    '{"passed": false, "factual_consistency": 0.80, "citation_support": 0.30, '
    '"hallucination_risk": 0.20, "issues": ["Cited source does not support the stated claim."], '
    '"revision_instructions": "Cite a source that actually supports the claim.", "confidence": 0.70}'
)
CRITIC_FAIL_THEN_FIX_R1 = (
    '{"passed": false, "factual_consistency": 0.60, "citation_support": 0.55, '
    '"hallucination_risk": 0.30, "issues": ["Clarify drug class"], '
    '"revision_instructions": "Specify ACE inhibitors or thiazide diuretics.", "confidence": 0.80}'
)

GEN_OK = "First-line therapy for essential hypertension includes ACE inhibitors and thiazide diuretics [E1]."
GEN_REVISED = "First-line agents include ACE inhibitors and thiazide diuretics per guidelines [E1]."


# ---- Results accumulator (written to the task folder as machine-readable JSON) ----
TASK_DIR = Path(r"d:\Ragmed_2\.agents\tasks\ragmed2-pipeline-smoke-2026-10-18")
RESULTS: Dict[str, Any] = {
    "suite": "test_pipeline_smoke_trace.py",
    "tau": None,
    "max_retrieval_iterations_source": MAX_RETRIEVAL_ITERATIONS,
    "max_critic_rounds_source": MAX_CRITIC_ROUNDS,
    "cases": [],
    "summary": {},
}


def _snippet(text: Optional[str], limit: int = 120) -> Optional[str]:
    """Short, redacted excerpt — never full private documents."""
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def _graph_nodes(trace: List[Dict[str, Any]]) -> List[str]:
    return [s["details"].get("node") for s in trace if s.get("agent_type") == "langgraph"]


def _agent_steps(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Non-langgraph agent step entries, with validated schema fields only."""
    out = []
    for s in trace:
        if s.get("agent_type") == "langgraph":
            continue
        out.append({
            "agent_type": s.get("agent_type"),
            "status": s.get("status"),
            "started_at": s.get("started_at"),
            "completed_at": s.get("completed_at"),
            "duration_ms": s.get("duration_ms"),
            "details_keys": sorted((s.get("details") or {}).keys()),
        })
    return out


def _schema_valid(final: AgentState) -> bool:
    """Round-trip the returned state through its own pydantic model."""
    try:
        AgentState.model_validate(final.model_dump())
        return True
    except Exception:
        return False


def _evidence_ids(final: AgentState) -> List[Dict[str, Any]]:
    ids = []
    for item in final.evidence:
        ids.append({
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "citation_marker": item.get("citation_marker"),
            "retrieval_iteration": item.get("retrieval_iteration"),
            "cited_text_excerpt": _snippet(item.get("cited_text")),
        })
    return ids


def _record_case(
    *,
    test_id: str,
    label: str,
    description: str,
    llm: Optional[SequenceFakeLLM],
    final: AgentState,
    expected: Dict[str, Any],
    extra_checks: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the per-case trace record and append it to RESULTS."""
    trace = final.agent_trace
    nodes = _graph_nodes(trace)
    conf = final.confidence_calibration or {}
    calibrated = conf.get("calibrated_confidence")
    abst = final.abstention_decision or {}
    critic = final.critic_verdict or {}

    confidence_finite_in_range = (
        calibrated is not None
        and isinstance(calibrated, (int, float))
        and math.isfinite(calibrated)
        and 0.0 <= calibrated <= 1.0
    )

    rec: Dict[str, Any] = {
        "test_id": test_id,
        "label": label,
        "description": description,
        "query_id": str(final.query_id),
        "schema_valid": _schema_valid(final),
        "graph_execution_order": nodes,
        "loop_counts": {
            "retrieval_nodes": nodes.count("retrieval"),
            "generator_nodes": nodes.count("generator"),
            "critic_nodes": nodes.count("critic"),
            "retrieval_iteration_count": final.retrieval_iteration_count,
            "critic_round_count": final.critic_round_count,
        },
        "agent_steps": _agent_steps(trace),
        "evidence": _evidence_ids(final),
        "answer_excerpt": _snippet(final.answer),
        "answer_citations": final.answer_citations,
        "gap_analysis": final.gap_analysis,
        "evidence_reranking": final.evidence_reranking,
        "critic_verdict": {
            "passed": critic.get("passed"),
            "factual_consistency": critic.get("factual_consistency"),
            "citation_support": critic.get("citation_support"),
            "hallucination_risk": critic.get("hallucination_risk"),
            "evidence_verification": critic.get("evidence_verification"),
        },
        "confidence": {
            "calibrated_confidence": calibrated,
            "confidence_band": conf.get("confidence_band"),
            "requires_abstention": conf.get("requires_abstention"),
            "finite_and_in_0_1": confidence_finite_in_range,
        },
        "abstention": {
            "decision": abst.get("decision"),
            "should_abstain": abst.get("should_abstain"),
            "reasons": abst.get("reasons"),
        },
        "status": final.status,
        "flags": {
            "retrieval_failure": final.retrieval_failure,
            "model_failure": final.model_failure,
            "contradictory_evidence": final.contradictory_evidence,
            "force_abstention": final.force_abstention,
            "out_of_domain": final.out_of_domain,
        },
        "error": final.error,
        "error_code": final.error_code,
        "latency_ms": latency_ms if latency_ms is not None else final.processing_time_ms,
        "expected": expected,
        "extra_checks": extra_checks or {},
    }
    RESULTS["cases"].append(rec)
    return rec


# ==============================================================================
# Synthetic evidence builders (in-memory; never touch the index)
# ==============================================================================
def _evidence_item(seed: str, marker: str, text: str, *, scores: float = 0.9) -> Dict[str, Any]:
    return {
        "chunk_id": str(create_deterministic_uuid(f"chunk_{seed}")),
        "document_id": str(create_deterministic_uuid(f"doc_{seed}")),
        "source_type": "pubmed",
        "title": f"Synthetic PubMed record {seed}",
        "journal": "Journal of Synthetic Evidence",
        "publication_year": 2024,
        "doi": f"10.0000/synthetic.{seed}",
        "url": f"https://doi.org/10.0000/synthetic.{seed}",
        "cited_text": text,
        "relevance_score": scores,
        "cross_encoder_score": scores,
        "evidence_quality_score": scores,
        "citation_marker": marker,
    }


SUPPORTING_EVIDENCE = [
    _evidence_item(
        "support-1",
        "[E1]",
        "First-line pharmacotherapy for essential hypertension includes thiazide "
        "diuretics and ACE inhibitors.",
        scores=0.92,
    )
]
UNRELATED_EVIDENCE = [
    _evidence_item(
        "swap-unrelated-1",
        "[E1]",
        "Photosynthesis in C4 plants concentrates carbon dioxide around RuBisCO to "
        "reduce photorespiration.",
        scores=0.90,
    )
]


# ==============================================================================
# CASE 1 — Sufficient evidence -> PROCEED / verified
# ==============================================================================
async def test_case01_sufficient_evidence_proceeds():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Strong evidence, critic passes -> verified, PROCEED."""
    llm = SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS])
    orch = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=stub_retrieval_provider)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case01"),
        original_question="What is the first-line treatment for essential hypertension?",
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    rec = _record_case(
        test_id="case01_sufficient",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Sufficient evidence -> verified / PROCEED (full 8-node happy path).",
        llm=llm, final=final,
        expected={"answerable": True, "abstain": False, "status": "verified"},
        latency_ms=latency,
    )

    assert final.status == "verified"
    assert final.abstention_decision["should_abstain"] is False
    assert "[E1]" in (final.answer or "")
    assert nodes == [
        "query_understanding", "retrieval", "gap_analysis", "evidence_reranking",
        "generator", "critic", "confidence", "abstention",
    ]
    assert rec["confidence"]["finite_and_in_0_1"] is True
    assert final.confidence_calibration["calibrated_confidence"] >= CONFIDENCE_THRESHOLD


# ==============================================================================
# CASE 2 — Requires additional retrieval (gap -> second retrieval pass)
# ==============================================================================
async def test_case02_requires_additional_retrieval():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] First gap=insufficient triggers a 2nd retrieval pass, then sufficient."""
    # LLM order: QUA, gap(insufficient), gap(sufficient), rerank, gen, critic
    llm = SequenceFakeLLM([QUA_RESP, GAP_INSUFFICIENT, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS])
    orch = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=stub_retrieval_provider)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case02"),
        original_question="What is the first-line treatment for essential hypertension?",
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    rec = _record_case(
        test_id="case02_additional_retrieval",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Insufficient first gap -> second retrieval pass -> sufficient -> verified.",
        llm=llm, final=final,
        expected={"answerable": True, "abstain": False, "status": "verified", "retrieval_passes": ">=2"},
        extra_checks={"retrieval_node_count": nodes.count("retrieval"),
                      "gap_node_count": nodes.count("gap_analysis")},
        latency_ms=latency,
    )

    assert nodes.count("retrieval") >= 2, "gap->retrieval loop must fire a second retrieval pass"
    assert nodes.count("gap_analysis") >= 2
    assert final.retrieval_iteration_count >= 2
    assert final.retrieval_iteration_count <= MAX_RETRIEVAL_ITERATIONS
    assert final.status == "verified"
    assert rec["confidence"]["finite_and_in_0_1"] is True


# ==============================================================================
# CASE 3 — Unanswerable / insufficient evidence -> ABSTAIN
# ==============================================================================
async def test_case03_unanswerable_abstains():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Empty retrieval -> retrieval_failure -> ABSTAIN (generator skipped)."""
    async def empty_retriever(**kwargs):
        return [], [{"iteration": 1, "status": "empty", "count": 0}]

    llm = SequenceFakeLLM([QUA_RESP])
    orch = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=empty_retriever)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case03"),
        original_question="What is the cure for an entirely fictional disease with no literature?",
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    rec = _record_case(
        test_id="case03_unanswerable",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Insufficient/empty evidence -> ABSTAIN, generator never runs.",
        llm=llm, final=final,
        expected={"answerable": False, "abstain": True, "status": "abstained"},
        latency_ms=latency,
    )

    assert final.retrieval_failure is True
    assert final.status == "abstained"
    assert final.abstention_decision["should_abstain"] is True
    assert "generator" not in nodes
    assert COPY_RETRIEVAL_FAILURE in final.abstention_decision["reasons"]
    assert rec["confidence"]["finite_and_in_0_1"] is True


# ==============================================================================
# CASE 4 — Claim contradicted by supplied evidence -> ABSTAIN
# ==============================================================================
async def test_case04_contradicted_claim_abstains():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Rerank flags contradiction -> ABSTAIN before generation."""
    llm = SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_CONTRADICTORY])
    orch = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=stub_retrieval_provider)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case04"),
        original_question="What is the treatment for hypertension?",
        evidence=[
            _evidence_item("contra-a", "[E1]", "Drug A lowers blood pressure in adults with hypertension."),
            _evidence_item("contra-b", "[E2]", "Drug A raises blood pressure and is contraindicated in hypertension."),
        ],
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    _record_case(
        test_id="case04_contradiction",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Contradictory evidence -> ABSTAIN; generator skipped.",
        llm=llm, final=final,
        expected={"answerable": False, "abstain": True, "status": "abstained",
                  "contradiction_detected": True},
        latency_ms=latency,
    )

    assert final.contradictory_evidence is True
    assert final.status == "abstained"
    assert final.abstention_decision["should_abstain"] is True
    assert "generator" not in nodes
    assert COPY_CONTRADICTORY_EVIDENCE in final.abstention_decision["reasons"]


# ==============================================================================
# CASE 5 — Citation that does not support its claim -> ABSTAIN
# ==============================================================================
async def test_case05_citation_not_supporting_claim_abstains():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Marker resolves (exists) but critic judges it does not support the claim.

    Verifies citation EXISTENCE alone is insufficient: the deterministic marker
    check passes (no blocking_issues) yet low citation_support drives ABSTAIN.
    """
    # Critic never passes within 2 rounds -> cap -> force_abstention. Round1 and Round2 both unsupported.
    llm = SequenceFakeLLM([
        QUA_RESP, GAP_SUFFICIENT, RERANK_OK,
        GEN_OK, CRITIC_CITATION_UNSUPPORTED,
        GEN_OK, CRITIC_CITATION_UNSUPPORTED,
    ])
    orch = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=stub_retrieval_provider)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case05"),
        original_question="What is the first-line treatment for essential hypertension?",
        evidence=list(SUPPORTING_EVIDENCE),
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    critic = final.critic_verdict or {}
    verification = critic.get("evidence_verification", {})
    rec = _record_case(
        test_id="case05_citation_unsupported",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Citation marker exists but does not support claim -> low citation_support -> ABSTAIN.",
        llm=llm, final=final,
        expected={"answerable": False, "abstain": True, "status": "abstained",
                  "citation_support_status": "exists_but_unsupported"},
        extra_checks={
            "deterministic_blocking_issues": verification.get("blocking_issues"),
            "critic_citation_support": critic.get("citation_support"),
        },
        latency_ms=latency,
    )

    # Marker [E1] resolves -> no deterministic blocking issue (existence is satisfied)...
    assert verification.get("blocking_issues") == []
    # ...but the critic's citation_support is low -> the pipeline must ABSTAIN.
    assert float(critic.get("citation_support", 1.0)) < 0.50
    assert final.status == "abstained"
    assert final.abstention_decision["should_abstain"] is True
    assert rec["confidence"]["finite_and_in_0_1"] is True


# ==============================================================================
# CASE 6a — Critic revision trigger then success
# ==============================================================================
async def test_case06a_critic_revision_then_success():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Round-1 critic fails with instructions; round-2 revised answer passes."""
    llm = SequenceFakeLLM([
        QUA_RESP, GAP_SUFFICIENT, RERANK_OK,
        GEN_OK, CRITIC_FAIL_THEN_FIX_R1,
        GEN_REVISED, CRITIC_PASS,
    ])
    orch = MultiAgentOrchestrator(llm_wrapper=llm, retrieval_provider=stub_retrieval_provider)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case06a"),
        original_question="Which drug classes treat essential hypertension?",
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    _record_case(
        test_id="case06a_revision_success",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Critic revision loop: fail -> re-generate -> pass; verified.",
        llm=llm, final=final,
        expected={"answerable": True, "abstain": False, "status": "verified",
                  "critic_rounds": 2},
        extra_checks={"revision_instruction_in_round2_prompt":
                      "Specify ACE inhibitors or thiazide diuretics." in llm.prompts[5]},
        latency_ms=latency,
    )

    assert final.critic_round_count == 2
    assert nodes.count("generator") == 2
    assert nodes.count("critic") == 2
    assert final.status == "verified"
    assert len(final.critic_reviews) == 2
    assert final.critic_reviews[0]["passed"] is False
    assert final.critic_reviews[1]["passed"] is True
    # Round-2 generator prompt carries the critic's explicit revision instruction.
    assert "Specify ACE inhibitors or thiazide diuretics." in llm.prompts[5]
    assert final.answer == GEN_REVISED


# ==============================================================================
# CASE 6b — Revision loop caps at MAX_CRITIC_ROUNDS (2)
# ==============================================================================
async def test_case06b_revision_loop_caps_at_two():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Critic never passes -> exactly 2 gen/2 critic -> force_abstention -> ABSTAIN."""
    llm = SequenceFakeLLM([
        QUA_RESP, GAP_SUFFICIENT, RERANK_OK,
        GEN_OK, CRITIC_FAIL_THEN_FIX_R1,
        GEN_OK, CRITIC_FAIL_THEN_FIX_R1,
    ])
    orch = MultiAgentOrchestrator(
        llm_wrapper=llm, retrieval_provider=stub_retrieval_provider,
        max_critic_rounds=MAX_CRITIC_ROUNDS,
    )
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case06b"),
        original_question="Which drug classes treat essential hypertension?",
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    _record_case(
        test_id="case06b_revision_cap",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Critic never passes -> revision cap (2) enforced -> force_abstention -> ABSTAIN.",
        llm=llm, final=final,
        expected={"answerable": False, "abstain": True, "status": "abstained",
                  "critic_rounds": 2, "generator_nodes": 2},
        latency_ms=latency,
    )

    assert final.critic_round_count == 2
    assert nodes.count("generator") == 2
    assert nodes.count("critic") == 2
    assert final.force_abstention is True
    assert final.status == "abstained"
    assert nodes[-2:] == ["confidence", "abstention"]


# ==============================================================================
# CASE 7 — Repeated-question determinism (generator temperature 0.0)
# ==============================================================================
async def test_case07_repeated_question_determinism():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Same question + same contract run twice -> identical outputs."""
    def _run_once():
        llm = SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS])
        orch = MultiAgentOrchestrator(
            llm_wrapper=llm, retrieval_provider=stub_retrieval_provider,
            generator_temperature=0.0,  # test-local override to remove sampling (not a config change)
        )
        state = AgentState(
            query_id=create_deterministic_uuid("smoke_case07"),
            original_question="What is the first-line treatment for essential hypertension?",
            include_trace=True,
        )
        return orch.run(state)

    t0 = time.perf_counter()
    final_a = await _run_once()
    final_b = await _run_once()
    latency = int((time.perf_counter() - t0) * 1000)

    nodes_a = _graph_nodes(final_a.agent_trace)
    nodes_b = _graph_nodes(final_b.agent_trace)
    identical = (
        final_a.answer == final_b.answer
        and final_a.confidence_calibration["calibrated_confidence"]
        == final_b.confidence_calibration["calibrated_confidence"]
        and final_a.abstention_decision["decision"] == final_b.abstention_decision["decision"]
        and nodes_a == nodes_b
    )
    _record_case(
        test_id="case07_determinism",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Repeated identical run -> identical answer/confidence/decision/node-order.",
        llm=None, final=final_b,
        expected={"deterministic": True},
        extra_checks={
            "answers_equal": final_a.answer == final_b.answer,
            "confidence_equal": final_a.confidence_calibration["calibrated_confidence"]
            == final_b.confidence_calibration["calibrated_confidence"],
            "decision_equal": final_a.abstention_decision["decision"]
            == final_b.abstention_decision["decision"],
            "node_order_equal": nodes_a == nodes_b,
        },
        latency_ms=latency,
    )

    assert identical


# ==============================================================================
# CASE 8 — Two sequential questions on same orchestrator -> state isolation
# ==============================================================================
async def test_case08_sequential_state_isolation():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Q_A then Q_B on same orchestrator -> no state bleed."""
    orch = MultiAgentOrchestrator(
        llm_wrapper=SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS]),
        retrieval_provider=stub_retrieval_provider,
    )
    state_a = AgentState(
        query_id=create_deterministic_uuid("smoke_case08_A"),
        original_question="What is the first-line treatment for essential hypertension?",
        evidence=[_evidence_item("isoA", "[E1]", "ACE inhibitors are first-line for hypertension.")],
        include_trace=True,
    )
    t0 = time.perf_counter()
    final_a = await orch.run(state_a)

    # Fresh LLM contract + fresh state for Q_B on the SAME orchestrator instance.
    orch.llm_wrapper = SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS])
    # Re-point agents at the swapped wrapper so Q_B uses the fresh contract.
    orch.query_agent.llm_wrapper = orch.llm_wrapper
    orch.gap_agent.llm_wrapper = orch.llm_wrapper
    orch.reranking_agent.llm_wrapper = orch.llm_wrapper
    orch.generator_agent.llm_wrapper = orch.llm_wrapper
    orch.critic_agent.llm_wrapper = orch.llm_wrapper
    state_b = AgentState(
        query_id=create_deterministic_uuid("smoke_case08_B"),
        original_question="What are the symptoms of type 2 diabetes?",
        evidence=[_evidence_item("isoB", "[E1]", "Type 2 diabetes presents with polyuria and polydipsia.")],
        include_trace=True,
    )
    final_b = await orch.run(state_b)
    latency = int((time.perf_counter() - t0) * 1000)

    a_doc_ids = {e["document_id"] for e in final_a.evidence}
    b_doc_ids = {e["document_id"] for e in final_b.evidence}
    no_doc_bleed = a_doc_ids.isdisjoint(b_doc_ids)
    isolated = (
        final_a.query_id != final_b.query_id
        and no_doc_bleed
        and final_b.answer == final_a.answer  # same mock answer text, but from B's own run
        and "polyuria" not in " ".join(e["cited_text_excerpt"] or "" for e in _evidence_ids(final_a))
    )
    _record_case(
        test_id="case08_state_isolation",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Two sequential questions, same orchestrator -> no evidence/id/citation bleed.",
        llm=None, final=final_b,
        expected={"isolated": True},
        extra_checks={
            "distinct_query_ids": final_a.query_id != final_b.query_id,
            "no_document_id_bleed": no_doc_bleed,
            "qB_evidence_doc_ids": sorted(b_doc_ids),
            "qA_evidence_doc_ids": sorted(a_doc_ids),
        },
        latency_ms=latency,
    )

    assert final_a.query_id != final_b.query_id
    assert no_doc_bleed, "document IDs from Q_A must not appear in Q_B evidence"
    # Q_B must not carry Q_A's disease-specific evidence text.
    assert all("hypertension" not in (e["cited_text_excerpt"] or "").lower() for e in _evidence_ids(final_b))


# ==============================================================================
# CASE 9 — Controlled evidence-swap (same question, different evidence)
# ==============================================================================
async def test_case09_controlled_evidence_swap():
    """[MOCK-LLM + SYNTHETIC-EVIDENCE] Hold question constant, swap evidence; prove swapped evidence is plumbed end-to-end.

    The mock LLM cannot react to content, so this asserts PLUMBING and SCORING, not
    answer-text change: (i) the generator prompt reflects the run's own evidence,
    (ii) confidence components differ with the differing evidence scores, (iii) cited
    IDs map to the supplied evidence for that run. Content-reactive answer change is
    only meaningful under the real LLM (case 10).
    """
    question = "What is the first-line treatment for essential hypertension?"

    high = [_evidence_item("swapA-support", "[E1]",
                           "First-line therapy for essential hypertension includes thiazide diuretics.",
                           scores=0.95)]
    low = [_evidence_item("swapB-unrelated", "[E1]",
                          "Photosynthesis in C4 plants concentrates CO2 around RuBisCO.",
                          scores=0.40)]

    # Variant A: supporting evidence, high scores.
    llm_a = SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS])
    orch_a = MultiAgentOrchestrator(llm_wrapper=llm_a, retrieval_provider=stub_retrieval_provider)
    final_a = await orch_a.run(AgentState(
        query_id=create_deterministic_uuid("smoke_case09_A"),
        original_question=question, evidence=[dict(e) for e in high], include_trace=True,
    ))

    llm_b = SequenceFakeLLM([QUA_RESP, GAP_SUFFICIENT, RERANK_OK, GEN_OK, CRITIC_PASS])
    orch_b = MultiAgentOrchestrator(llm_wrapper=llm_b, retrieval_provider=stub_retrieval_provider)
    final_b = await orch_b.run(AgentState(
        query_id=create_deterministic_uuid("smoke_case09_B"),
        original_question=question, evidence=[dict(e) for e in low], include_trace=True,
    ))

    # Generator prompt index along happy path: QUA(0), gap(1), rerank(2), generator(3).
    gen_prompt_a = llm_a.prompts[3]
    gen_prompt_b = llm_b.prompts[3]
    a_text = high[0]["cited_text"]
    b_text = low[0]["cited_text"]

    plumbing_ok = (
        a_text in gen_prompt_a and b_text not in gen_prompt_a
        and b_text in gen_prompt_b and a_text not in gen_prompt_b
    )
    conf_a = final_a.confidence_calibration
    conf_b = final_b.confidence_calibration
    scores_differ = (
        conf_a["retrieval_confidence"] != conf_b["retrieval_confidence"]
        or conf_a["evidence_quality_score"] != conf_b["evidence_quality_score"]
    )

    _record_case(
        test_id="case09_evidence_swap",
        label="MOCK-LLM + SYNTHETIC-EVIDENCE",
        description="Evidence swap: swapped evidence is plumbed to generator; confidence components track it.",
        llm=None, final=final_b,
        expected={"generator_sees_run_evidence": True, "confidence_tracks_evidence": True},
        extra_checks={
            "variantA_gen_prompt_excerpt": _snippet(gen_prompt_a, 200),
            "variantB_gen_prompt_excerpt": _snippet(gen_prompt_b, 200),
            "plumbing_ok": plumbing_ok,
            "variantA_retrieval_confidence": conf_a["retrieval_confidence"],
            "variantB_retrieval_confidence": conf_b["retrieval_confidence"],
            "scores_differ": scores_differ,
        },
        latency_ms=None,
    )

    assert plumbing_ok, "generator must receive THIS run's evidence, not the other variant's"
    assert scores_differ, "confidence components must reflect the swapped evidence scores"
    # Cited IDs map to the supplied evidence for each run.
    assert {e["document_id"] for e in final_b.evidence} == {low[0]["document_id"]}


# ==============================================================================
# CASE 10 — REAL end-to-end with the local GPU LLM (gated)
# ==============================================================================
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RAGMED_SMOKE_REAL_LLM") != "1",
    reason="real-LLM case runs only when RAGMED_SMOKE_REAL_LLM=1 on the GPU interpreter",
)
async def test_case10_real_llm_end_to_end():
    """[REAL-LLM + SYNTHETIC-EVIDENCE] One true end-to-end case using the local Qwen GPU model.

    Uses LocalLLMWrapper(LOCAL) (strict GPU load, no silent fallback) with the
    index-free synthetic retrieval stub so NO production index is touched. If the
    model cannot load within budget, the case is recorded NOT TESTED with a reason
    and the test is skipped — never fabricated, never presented as mocked.
    """
    from app.llm.local_llm import LocalLLMWrapper, LLMRuntimeMode

    load_error = None
    wrapper = None
    load_ms = None
    try:
        t_load = time.perf_counter()
        wrapper = LocalLLMWrapper(mode=LLMRuntimeMode.LOCAL, allow_fallback=False)
        wrapper.load_model()
        load_ms = int((time.perf_counter() - t_load) * 1000)
    except Exception as exc:  # ModelLoadError or any load failure
        load_error = f"{type(exc).__name__}: {exc}"

    if load_error is not None:
        RESULTS["cases"].append({
            "test_id": "case10_real_llm",
            "label": "REAL-LLM + SYNTHETIC-EVIDENCE",
            "description": "True end-to-end with local Qwen GPU model.",
            "status": "NOT TESTED",
            "reason": f"model load failed: {load_error}",
        })
        pytest.skip(f"real LLM model could not load: {load_error}")

    orch = MultiAgentOrchestrator(llm_wrapper=wrapper, retrieval_provider=stub_retrieval_provider)
    state = AgentState(
        query_id=create_deterministic_uuid("smoke_case10_real"),
        original_question="What is the first-line treatment for essential hypertension?",
        include_trace=True,
    )
    t0 = time.perf_counter()
    final = await orch.run(state)
    latency = int((time.perf_counter() - t0) * 1000)

    nodes = _graph_nodes(final.agent_trace)
    rec = _record_case(
        test_id="case10_real_llm",
        label="REAL-LLM + SYNTHETIC-EVIDENCE",
        description="True end-to-end with local Qwen GPU model + index-free synthetic retrieval.",
        llm=None, final=final,
        expected={"end_to_end_executes": True},
        extra_checks={"model_load_ms": load_ms, "visited_nodes": nodes},
        latency_ms=latency,
    )

    # Only structural assertions — we do NOT assert generation quality from one real case.
    assert nodes[0] == "query_understanding"
    assert nodes[-1] == "abstention"
    assert "confidence" in nodes
    assert rec["confidence"]["finite_and_in_0_1"] is True
    assert final.status in ("verified", "abstained")


# ==============================================================================
# tau confirmation (read-only) + results.json writer
# ==============================================================================
def test_zzz_tau_is_065_and_write_results():
    """Confirm tau=0.65 is present in source (read-only) and flush results.json.

    Named with a 'zzz' prefix so pytest's alphabetical ordering runs it last,
    after every case has appended its record.
    """
    assert CONFIDENCE_THRESHOLD == 0.65, "tau/CONFIDENCE_THRESHOLD must remain 0.65"
    RESULTS["tau"] = CONFIDENCE_THRESHOLD

    # Summary counts with explicit denominators (NOT general accuracy claims).
    cases = [c for c in RESULTS["cases"] if "status" in c]
    answerable_cases = [c for c in cases if c.get("expected", {}).get("abstain") is not None]
    abstain_expected = sum(1 for c in answerable_cases if c["expected"]["abstain"] is True)
    abstain_actual = sum(
        1 for c in answerable_cases
        if c.get("abstention", {}).get("should_abstain") is True
    )
    abstain_match = sum(
        1 for c in answerable_cases
        if c["expected"]["abstain"] == (c.get("abstention", {}).get("should_abstain"))
    )
    confidence_valid = sum(
        1 for c in cases
        if c.get("confidence", {}).get("finite_and_in_0_1") is True
    )
    confidence_denominator = sum(
        1 for c in cases if "confidence" in c and c["confidence"].get("calibrated_confidence") is not None
    )
    RESULTS["summary"] = {
        "total_recorded_cases": len(RESULTS["cases"]),
        "answerability_abstention_denominator": len(answerable_cases),
        "abstain_expected_count": abstain_expected,
        "abstain_actual_count": abstain_actual,
        "expected_vs_actual_abstention_matches": abstain_match,
        "confidence_finite_in_0_1": f"{confidence_valid}/{confidence_denominator}",
        "note": "Smoke-test counts only. NOT a general-accuracy or significance claim.",
    }

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "results.json").write_text(
        json.dumps(RESULTS, indent=2, default=str), encoding="utf-8"
    )
    assert (TASK_DIR / "results.json").is_file()
