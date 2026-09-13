"""Tests for the async compatibility boundary over the native pipeline."""

import asyncio

from app.rag.chunking.models import MedicalChunk
from app.rag.pipeline import RAGPipeline


class _NativePipeline:
    def retrieve(self, query, final_top_k=None):
        assert query == "hypertension treatment"
        assert final_top_k == 1
        return [
            (
                MedicalChunk(
                    id="chunk-1",
                    document_id="doc-1",
                    content="ACE inhibitors are first-line treatment.",
                    title="Hypertension",
                    source="medqa",
                    metadata={"doi": "10.1/example"},
                ),
                0.9,
            )
        ]


def test_facade_maps_native_evidence_to_agent_contract():
    pipeline = RAGPipeline(_NativePipeline(), top_k_final=1)

    evidence, trace = asyncio.run(pipeline.retrieve_evidence("hypertension treatment"))

    assert pipeline.is_ready
    assert evidence[0]["chunk_id"] == "chunk-1"
    assert evidence[0]["cited_text"] == "ACE inhibitors are first-line treatment."
    assert evidence[0]["citation_marker"] == "[E1]"
    assert trace[0]["status"] == "completed"


def test_facade_preserves_empty_and_unconfigured_behavior():
    pipeline = RAGPipeline()

    empty_evidence, empty_trace = asyncio.run(pipeline.retrieve_evidence("   "))
    absent_evidence, absent_trace = asyncio.run(pipeline.retrieve_evidence("query"))

    assert not pipeline.is_ready
    assert empty_evidence == []
    assert empty_trace[0]["status"] == "empty_query"
    assert absent_evidence == []
    assert absent_trace[0]["status"] == "no_retriever"
