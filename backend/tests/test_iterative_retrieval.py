import pytest

from app.agents.iterative_retrieval import (
    IterativeRetrievalAgent,
)


class FakeLLM:

    def __init__(self, response):
        self.response = response

    def generate(
        self,
        prompt,
        temperature=0.0,
    ):
        return self.response
async def test_gap_analysis_evidence_is_sufficient():

    llm = FakeLLM(
        """
        {
            "sufficient": true,
            "reasoning": "The evidence directly addresses the question.",
            "missing_information": [],
            "sub_query": null,
            "confidence": 0.92
        }
        """
    )

    agent = IterativeRetrievalAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are common symptoms of type 2 diabetes?",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Common symptoms include increased thirst and frequent urination."
            }
        ],
    }

    result = await agent.run(state)

    gap = result["gap_analysis"]

    assert gap["sufficient"] is True
    assert gap["sub_query"] is None
    assert gap["confidence"] == 0.92

    assert "agent_trace" in result
    assert (
        result["agent_trace"][0]["status"]
        == "completed"
    )
async def test_gap_analysis_generates_subquery():

    llm = FakeLLM(
        """
        {
            "sufficient": false,
            "reasoning": "The evidence does not address treatment outcomes.",
            "missing_information": [
                "treatment outcomes"
            ],
            "sub_query": "treatment outcomes for type 2 diabetes mellitus",
            "confidence": 0.88
        }
        """
    )

    agent = IterativeRetrievalAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What is the best treatment for type 2 diabetes?",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Type 2 diabetes is characterized by insulin resistance."
            }
        ],
    }

    result = await agent.run(state)

    gap = result["gap_analysis"]

    assert gap["sufficient"] is False

    assert (
        "treatment outcomes"
        in gap["missing_information"]
    )

    assert (
        gap["sub_query"]
        == "treatment outcomes for type 2 diabetes mellitus"
    )
async def test_gap_analysis_requires_question():

    llm = FakeLLM(
        """
        {
            "sufficient": true,
            "reasoning": "",
            "missing_information": [],
            "sub_query": null,
            "confidence": 0.9
        }
        """
    )

    agent = IterativeRetrievalAgent(
        llm_wrapper=llm
    )

    with pytest.raises(RuntimeError):

        await agent.run(
            {
                "evidence": []
            }
        )