import pytest

from app.agents.generator import GeneratorAgent


class FakeLLM:

    def __init__(self, response):
        self.response = response

    def generate(
        self,
        prompt,
        temperature=0.0,
    ):
        return self.response


@pytest.mark.asyncio
async def test_generator_creates_grounded_answer():

    llm = FakeLLM(
        """
        Common symptoms of type 2 diabetes include
        increased thirst and frequent urination [E1].
        """
    )

    agent = GeneratorAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are common symptoms of type 2 diabetes?",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "title": "Diabetes Review",
                "journal": "Medical Journal",
                "publication_year": 2025,
                "cited_text":
                    "Common symptoms include increased thirst and frequent urination."
            }
        ],
    }

    result = await agent.run(state)

    assert "answer" in result

    assert (
        "increased thirst"
        in result["answer"]
    )

    assert (
        "[E1]"
        in result["answer"]
    )

    assert result["answer_citations"] == [
        "[E1]"
    ]

    assert "agent_trace" in result

    assert (
        result["agent_trace"][0]["status"]
        == "completed"
    )


@pytest.mark.asyncio
async def test_generator_rejects_unknown_citation():

    llm = FakeLLM(
        """
        Type 2 diabetes commonly causes
        increased thirst [E99].
        """
    )

    agent = GeneratorAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are symptoms of type 2 diabetes?",

        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Increased thirst is a common symptom."
            }
        ],
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)


@pytest.mark.asyncio
async def test_generator_requires_evidence():

    llm = FakeLLM(
        "There is insufficient evidence."
    )

    agent = GeneratorAgent(
        llm_wrapper=llm
    )

    state = {
        "original_question":
            "What are symptoms of diabetes?",
        "evidence": [],
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)


@pytest.mark.asyncio
async def test_generator_requires_question():

    llm = FakeLLM(
        "There is insufficient evidence."
    )

    agent = GeneratorAgent(
        llm_wrapper=llm
    )

    state = {
        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text":
                    "Diabetes is a chronic disease."
            }
        ]
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)