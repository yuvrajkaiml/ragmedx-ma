"""Integration coverage for the native RetrievalService orchestration boundary."""

import pytest

from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.state import AgentState, create_deterministic_uuid
from app.rag.chunking.models import MedicalChunk
from app.rag.pipeline import RAGPipeline
from app.services.retrieval_service import RetrievalService


class _SequenceLLM:
    def __init__(self, responses):
        self._responses = iter(responses)

    def generate(self, prompt, **kwargs):
        return next(self._responses), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _ThreePassNativePipeline:
    """Native pipeline double wrapped by the real RAGPipeline/RetrievalService."""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, final_top_k=None):
        self.queries.append(query)
        number = len(self.queries)
        return [
            (
                MedicalChunk(
                    id=f"chunk-{number}",
                    document_id=f"doc-{number}",
                    content=f"Evidence returned by native pass {number}.",
                    title=f"Native source {number}",
                    source="medqa",
                ),
                float(number),
            )
        ]


@pytest.mark.integration
async def test_retrieval_service_runs_three_mga_controlled_passes_without_stub():
    native = _ThreePassNativePipeline()
    service = RetrievalService(RAGPipeline(native, top_k_final=1))
    llm = _SequenceLLM([
        '{"medical_specialty":"general","expanded_query":"initial clinical query","confidence":0.9}',
        '{"sufficient":false,"reasoning":"Need dosing evidence.","missing_information":["dosing"],"sub_query":"hypertension first-line dose","confidence":0.9}',
        '{"sufficient":false,"reasoning":"Need contraindications.","missing_information":["contraindications"],"sub_query":"hypertension ACE inhibitor contraindications","confidence":0.9}',
        '{"sufficient":true,"reasoning":"Evidence is now sufficient.","missing_information":[],"sub_query":null,"confidence":0.9}',
        '{"contradictory":false,"selected_markers":[],"reasoning":"Evidence agrees."}',
        "Use the supplied evidence only [E1].",
        '{"passed":true,"factual_consistency":0.95,"citation_support":0.95,"hallucination_risk":0.05,"confidence":0.95}',
    ])
    orchestrator = MultiAgentOrchestrator(
        llm_wrapper=llm,
        retrieval_provider=service.retrieve_evidence,
    )

    result = await orchestrator.run(AgentState(
        query_id=create_deterministic_uuid("native-three-pass"),
        original_question="What is initial hypertension management?",
        include_trace=True,
    ))

    assert native.queries == [
        "initial clinical query",
        "hypertension first-line dose",
        "hypertension ACE inhibitor contraindications",
    ]
    assert result.retrieval_iteration_count == 3
    assert [entry["orchestrator_iteration"] for entry in result.retrieval_iterations] == [1, 2, 3]
    assert all(step["details"].get("action") != "temporary_retrieval_stub" for step in result.agent_trace)
    assert sum(step["details"].get("action") == "native_retrieval" for step in result.agent_trace) == 3
    assert result.status == "verified"
