"""Tests for the async compatibility boundary over the native pipeline."""

import asyncio

import pytest

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
    assert evidence[0]["citation_marker"] == "[Doc 1]"
    assert evidence[0]["cross_encoder_logit"] == 0.9
    assert 0.0 <= evidence[0]["cross_encoder_score"] <= 1.0
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


def test_facade_normalizes_unbounded_cross_encoder_logit():
    class _LogitNativePipeline:
        def retrieve(self, query, final_top_k=None):
            return [
                (
                    MedicalChunk(
                        id="chunk-logit",
                        document_id="doc-logit",
                        content="Evidence with an unbounded reranker logit.",
                        title="Logit test",
                        source="medqa",
                    ),
                    2.0,
                )
            ]

    pipeline = RAGPipeline(_LogitNativePipeline(), top_k_final=1)
    evidence, _ = asyncio.run(pipeline.retrieve_evidence("logit normalization"))

    assert evidence[0]["cross_encoder_logit"] == 2.0
    assert evidence[0]["cross_encoder_score"] == pytest.approx(0.8808, abs=0.0001)
