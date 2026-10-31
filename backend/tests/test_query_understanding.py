import pytest

from app.agents.query_understanding import (
    QueryUnderstandingAgent,
)


class FakeLLM:

    def generate(
        self,
        prompt,
        temperature=0.0,
    ):
        return """
        {
            "medical_specialty": "endocrinology",
            "intent": "symptom identification",
            "expanded_query": "symptoms of type 2 diabetes mellitus",
            "abbreviations": {
                "T2DM": "type 2 diabetes mellitus"
            },
            "confidence": 0.95
        }
        """
async def test_query_understanding():
    agent = QueryUnderstandingAgent(
        llm_wrapper=FakeLLM()
    )

    state = {
        "original_question":
            "What are the symptoms of T2DM?"
    }

    result = await agent.run(state)

    assert "processed_query" in result

    processed = result["processed_query"]

    assert (
        processed["medical_specialty"]
        == "endocrinology"
    )

    assert (
        processed["intent"]
        == "symptom identification"
    )

    assert (
        processed["expanded_query"]
        == "symptoms of type 2 diabetes mellitus"
    )

    assert (
        processed["abbreviations"]["T2DM"]
        == "type 2 diabetes mellitus"
    )

    assert processed["confidence"] == 0.95

    assert "agent_trace" in result
    assert len(result["agent_trace"]) == 1
    assert (
        result["agent_trace"][0]["status"]
        == "completed"
    )
async def test_query_understanding_rejects_invalid_json():

    class InvalidLLM:

        def generate(
            self,
            prompt,
            temperature=0.0,
        ):
            return "This is not JSON."

    agent = QueryUnderstandingAgent(
        llm_wrapper=InvalidLLM()
    )

    state = {
        "original_question":
            "What are the symptoms of diabetes?"
    }

    with pytest.raises(RuntimeError):
        await agent.run(state)
async def test_query_understanding_requires_question():

    agent = QueryUnderstandingAgent(
        llm_wrapper=FakeLLM()
    )

    with pytest.raises(RuntimeError):
        await agent.run({})