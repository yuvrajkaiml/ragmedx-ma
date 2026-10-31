"""End-to-end LangGraph test using a persisted FAISS index of real PubMed records.

The language-model responses are deterministic because this test host cannot run
the local GPU LLM. Evidence is not mocked: it is retrieved through BGE-M3,
FAISS, BM25/RRF, and the BGE cross-encoder from five PubMed seed records.
"""

import json
from pathlib import Path

import pytest

from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.state import AgentState, create_deterministic_uuid
from app.rag.chunking.models import MedicalChunk
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.pipeline import RAGPipeline, RAGRetrievalPipeline
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.services.retrieval_service import RetrievalService


class _GraphLLM:
    """Deterministic control-plane LLM; it never supplies retrieval evidence."""

    def __init__(self):
        self.responses = iter([
            '{"medical_specialty":"cardiology","intent":"treatment","expanded_query":"hypertension exercise treatment systematic review","abbreviations":{},"confidence":0.95}',
            '{"sufficient":true,"reasoning":"The retrieved reviews directly address non-drug hypertension treatment.","missing_information":[],"sub_query":null,"confidence":0.95}',
            '{"contradictory":false,"selected_markers":[],"reasoning":"The selected systematic reviews are compatible for this scoped question."}',
            "The retrieved systematic reviews report that exercise interventions can lower blood pressure in adults with hypertension [E1].",
            '{"passed":true,"factual_consistency":0.95,"citation_support":0.95,"hallucination_risk":0.05,"issues":[],"revision_instructions":null,"confidence":0.95}',
        ])

    def generate(self, prompt, **kwargs):
        return next(self.responses), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "is_mock": True}


def _load_real_pubmed_chunks() -> list[MedicalChunk]:
    fixture = Path(__file__).parent / "fixtures" / "pubmed_hypertension_seed.jsonl"
    records = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 5
    return [
        MedicalChunk(
            id=f"pubmed-{record['pmid']}",
            document_id=record["pmid"],
            content=record["content"],
            title=record["title"],
            source="pubmed",
            journal=record["journal"],
            publication_year=record["publication_year"],
            metadata={"pmid": record["pmid"], "doi": record["doi"]},
        )
        for record in records
    ]


@pytest.mark.integration
async def test_langgraph_runs_against_real_pubmed_seed_faiss_corpus(tmp_path):
    chunks = _load_real_pubmed_chunks()
    embedder = BGEEmbedding()
    store = FAISSVectorStore(dimension=1024)
    store.add(embedder.embed_documents([chunk.content for chunk in chunks]), chunks)
    index_dir = tmp_path / "pubmed_seed_faiss"
    store.save(index_dir)
    assert (index_dir / "index.faiss").is_file()
    persisted_store = FAISSVectorStore.load(index_dir)
    assert persisted_store.size == 5

    dense = DenseRetriever(embedder=embedder, vector_store=persisted_store)
    sparse = SparseRetriever()
    sparse.add_chunks(chunks)
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    native = RAGRetrievalPipeline(
        hybrid_retriever=hybrid,
        reranker=CrossEncoderReranker(),
        retrieval_top_k=5,
        final_top_k=3,
    )
    service = RetrievalService(RAGPipeline(native, top_k_final=3))
    orchestrator = MultiAgentOrchestrator(
        llm_wrapper=_GraphLLM(),
        retrieval_provider=service.retrieve_evidence,
    )

    result = await orchestrator.run(AgentState(
        query_id=create_deterministic_uuid("real-pubmed-seed-graph"),
        original_question="Can exercise lower blood pressure in adults with hypertension?",
        include_trace=True,
    ))

    assert result.evidence
    assert all(item["source_type"] == "pubmed" for item in result.evidence)
    assert all(item["document_id"] in {chunk.document_id for chunk in chunks} for item in result.evidence)
    visited_nodes = [step["details"].get("node") for step in result.agent_trace if step["agent_type"] == "langgraph"]
    assert visited_nodes == [
        "query_understanding", "retrieval", "gap_analysis", "evidence_reranking",
        "generator", "critic", "confidence", "abstention",
    ]
    assert result.answer and "[E1]" in result.answer
    assert result.abstention_decision is not None
