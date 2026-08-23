"""
Temporary isolated retrieval stub implementing Sana's verified retrieval boundary.

NOTE: This stub exists solely for deterministic unit testing and pipeline scaffolding
before Sana's real vector database and retriever are connected.
Replacing this stub with Sana's production retrieve_evidence() is a one-line dependency change.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.agents.state import EvidenceSnippet, create_deterministic_uuid


async def retrieve_evidence(
    query: str,
    optimized_query: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    max_iterations: int = 3,
    top_k_initial: int = 50,
    top_k_final: int = 5,
    enable_contradiction_check: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve clinical evidence snippets for a medical question.

    Args:
        query: User's original or expanded query string.
        optimized_query: Optional search-optimized query from Agent 1.
        query_embedding: Optional precomputed dense query embedding.
        max_iterations: Maximum iterative retrieval iterations.
        top_k_initial: Initial candidate pool size for dense/sparse retrieval.
        top_k_final: Final top-K reranked evidence snippets.
        enable_contradiction_check: Whether to perform inter-source conflict checks.

    Returns:
        Tuple of (evidence_snippet_dicts, retrieval_iteration_metadata_list).
    """
    clean_query = str(optimized_query or query or "").strip()

    # Empty query produces empty retrieval
    if not clean_query:
        return [], [{"iteration": 1, "query_used": "", "count": 0, "status": "empty_query"}]

    # Deterministic test fixture generation based on query content
    doc_id_1 = create_deterministic_uuid(f"doc_1_{clean_query[:20]}")
    chunk_id_1 = create_deterministic_uuid(f"chunk_1_{clean_query[:20]}")
    doc_id_2 = create_deterministic_uuid(f"doc_2_{clean_query[:20]}")
    chunk_id_2 = create_deterministic_uuid(f"chunk_2_{clean_query[:20]}")

    evidence_items: List[Dict[str, Any]] = [
        {
            "chunk_id": str(chunk_id_1),
            "document_id": str(doc_id_1),
            "source_type": "pubmed",
            "title": "Clinical Practice Guidelines for Hypertension Management",
            "journal": "New England Journal of Medicine",
            "publication_year": 2024,
            "doi": "10.1056/NEJMcp2401234",
            "url": "https://doi.org/10.1056/NEJMcp2401234",
            "cited_text": (
                "First-line pharmacotherapy for primary essential hypertension includes "
                "thiazide diuretics, calcium channel blockers, and ACE inhibitors or ARBs."
            ),
            "relevance_score": 0.92,
            "cross_encoder_score": 0.89,
            "evidence_quality_score": 0.95,
            "citation_marker": "[E1]",
        },
        {
            "chunk_id": str(chunk_id_2),
            "document_id": str(doc_id_2),
            "source_type": "pmc",
            "title": "Comparative Efficacy of Antihypertensive Drug Classes: Systematic Review",
            "journal": "Lancet",
            "publication_year": 2023,
            "doi": "10.1016/S0140-6736(23)00123-4",
            "url": "https://doi.org/10.1016/S0140-6736(23)00123-4",
            "cited_text": (
                "Major guideline classes show similar cardiovascular mortality reduction; "
                "initial monotherapy choice is tailored by patient demographics and tolerability."
            ),
            "relevance_score": 0.88,
            "cross_encoder_score": 0.86,
            "evidence_quality_score": 0.91,
            "citation_marker": "[E2]",
        },
    ]

    iterations = [
        {
            "iteration": 1,
            "query_used": clean_query,
            "count": len(evidence_items),
            "status": "completed",
            "contradiction_detected": False,
        }
    ]

    return evidence_items[:top_k_final], iterations
