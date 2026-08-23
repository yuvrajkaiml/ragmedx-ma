import pytest
from uuid import UUID

from app.agents.state import (
    AgentState,
    AgentType,
    ConfidenceCalibrationOutput,
    CriticVerdict,
    EvidenceSnippet,
    QueryUnderstandingOutput,
    create_deterministic_uuid,
)
from app.llm.temporary_retrieval_stub import retrieve_evidence
from app.schemas.response import (
    CitationOutput,
    ConfidenceScoreOutput,
    EvidenceSnippet as ResponseEvidenceSnippet,
)


def test_agent_state_initialization_and_serialization():
    """Verify AgentState initializes correctly and serializes to dict."""
    q_id = create_deterministic_uuid("query_123")
    c_id = create_deterministic_uuid("conv_456")

    state = AgentState(
        query_id=q_id,
        conversation_id=c_id,
        original_question="What is hypertension?",
        include_trace=True,
    )

    assert state.query_id == q_id
    assert state.conversation_id == c_id
    assert state.status == "initialized"
    assert state.retrieval_iteration_count == 0
    assert state.critic_round_count == 0

    state_dict = state.model_dump()
    assert state_dict["query_id"] == q_id
    assert state_dict["original_question"] == "What is hypertension?"
    assert state_dict["include_trace"] is True


def test_evidence_snippet_requires_explicit_ids():
    """Verify EvidenceSnippet enforces explicit UUIDs and valid score ranges."""
    doc_id = create_deterministic_uuid("doc_1")
    chunk_id = create_deterministic_uuid("chunk_1")

    evidence = EvidenceSnippet(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_type="pubmed",
        title="Hypertension Guidelines",
        cited_text="Treatment includes ACE inhibitors.",
        relevance_score=0.95,
        cross_encoder_score=0.92,
        evidence_quality_score=0.90,
        citation_marker="[E1]",
    )

    assert evidence.chunk_id == chunk_id
    assert evidence.document_id == doc_id
    assert evidence.relevance_score == 0.95

    # Converting to response evidence snippet
    res_evidence = ResponseEvidenceSnippet(
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        source_type="pubmed",
        title=evidence.title,
        cited_text=evidence.cited_text,
        relevance_score=evidence.relevance_score,
    )
    assert res_evidence.chunk_id == chunk_id
async def test_temporary_retrieval_stub_contract():
    """Verify retrieve_evidence returns valid fixture evidence with deterministic IDs."""
    query = "hypertension first line treatment"
    evidence_list, iterations = await retrieve_evidence(
        query=query,
        top_k_final=2,
    )

    assert len(evidence_list) == 2
    assert len(iterations) == 1
    assert iterations[0]["status"] == "completed"

    for ev in evidence_list:
        # Validate that chunk_id and document_id are valid UUID strings
        chunk_uuid = UUID(ev["chunk_id"])
        doc_uuid = UUID(ev["document_id"])
        assert chunk_uuid is not None
        assert doc_uuid is not None
        assert ev["relevance_score"] >= 0.0
        assert ev["citation_marker"].startswith("[E")
async def test_temporary_retrieval_stub_empty_query():
    """Verify empty query returns empty evidence list."""
    evidence_list, iterations = await retrieve_evidence(query="")
    assert evidence_list == []
    assert len(iterations) == 1
    assert iterations[0]["status"] == "empty_query"
