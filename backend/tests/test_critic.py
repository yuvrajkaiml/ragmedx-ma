import pytest

from app.agents.critic import CriticAgent


class FakeLLM:

    def __init__(self, response):
        self.response = response

    def generate(
        self,
        prompt,
        temperature=0.0,
    ):
        return self.response
async def test_critic_passes_supported_answer():

    llm = FakeLLM(
        """
        {
            "passed": true,
            "factual_consistency": 0.95,
            "citation_support": 0.92,
            "hallucination_risk": 0.05,
            "issues": [],
            "revision_instructions": null,
            "confidence": 0.94
        }
        """
    )

    agent = CriticAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are common symptoms of type 2 diabetes?",

        "answer":
            "Common symptoms include increased thirst [E1].",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Common symptoms include increased thirst."
            }
        ],
    }

    result = await agent.run(state)

    verdict = result["critic_verdict"]

    assert verdict["passed"] is True
    assert verdict["factual_consistency"] == 0.95
    assert verdict["citation_support"] == 0.92
    assert verdict["issues"] == []

    assert (
        result["agent_trace"][0]["status"]
        == "completed"
    )
async def test_critic_requests_revision():

    llm = FakeLLM(
        """
        {
            "passed": false,
            "factual_consistency": 0.55,
            "citation_support": 0.40,
            "hallucination_risk": 0.70,
            "issues": [
                "The treatment recommendation is not supported by the evidence."
            ],
            "revision_instructions":
                "Remove the unsupported treatment recommendation.",
            "confidence": 0.91
        }
        """
    )

    agent = CriticAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What is the treatment for type 2 diabetes?",

        "answer":
            "Patients should always receive treatment X [E1].",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Type 2 diabetes is associated with insulin resistance."
            }
        ],
    }

    result = await agent.run(state)

    verdict = result["critic_verdict"]

    assert verdict["passed"] is False

    assert (
        verdict["hallucination_risk"]
        == 0.70
    )

    assert len(verdict["issues"]) == 1

    assert (
        verdict["revision_instructions"]
        is not None
    )
async def test_critic_rejects_invalid_score():

    llm = FakeLLM(
        """
        {
            "passed": true,
            "factual_consistency": 1.5,
            "citation_support": 0.9,
            "hallucination_risk": 0.1,
            "issues": [],
            "revision_instructions": null,
            "confidence": 0.9
        }
        """
    )

    agent = CriticAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are symptoms of diabetes?",

        "answer":
            "Increased thirst [E1].",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Increased thirst is a symptom."
            }
        ],
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)
async def test_critic_requires_answer():

    llm = FakeLLM(
        """
        {
            "passed": true,
            "factual_consistency": 0.9,
            "citation_support": 0.9,
            "hallucination_risk": 0.1,
            "issues": [],
            "revision_instructions": null,
            "confidence": 0.9
        }
        """
    )

    agent = CriticAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are symptoms of diabetes?",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Increased thirst is a symptom."
            }
        ],
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)
async def test_critic_requires_revision_instructions():

    llm = FakeLLM(
        """
        {
            "passed": false,
            "factual_consistency": 0.4,
            "citation_support": 0.3,
            "hallucination_risk": 0.8,
            "issues": [
                "Unsupported claim."
            ],
            "revision_instructions": null,
            "confidence": 0.9
        }
        """
    )

    agent = CriticAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are symptoms of diabetes?",

        "answer":
            "Unsupported claim [E1].",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Diabetes is a chronic disease."
            }
        ],
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)


async def test_critic_rejects_answer_marker_without_a_real_evidence_span():
    llm = FakeLLM(
        '{"passed": true, "factual_consistency": 0.9, "citation_support": 0.9, '
        '"hallucination_risk": 0.1, "issues": [], "revision_instructions": null, "confidence": 0.9}'
    )
    result = await CriticAgent(llm_wrapper=llm).run({
        "original_question": "What is hypertension?",
        "answer": "Hypertension is high blood pressure [Doc 99].",
        "evidence": [{
            "chunk_id": "chunk-1", "document_id": "doc-1", "title": "Review",
            "journal": "Journal", "publication_year": 2025,
            "citation_marker": "[Doc 1]",
            "cited_text": "Hypertension is high blood pressure.",
        }],
    })

    verdict = result["critic_verdict"]
    assert verdict["passed"] is False
    assert verdict["citation_support"] == 0.0
    assert verdict["evidence_verification"]["unresolved_markers"] == ["[Doc 99]"]
    assert verdict["revision_instructions"]
