import pytest

from app.agents.abstention_decision import (
    AbstentionDecisionAgent,
)


def make_calibration(**overrides):
    calibration = {
        "calibrated_confidence": 0.85,
        "evidence_score": 0.90,
        "factual_consistency_score": 0.90,
        "citation_support_score": 0.90,
        "hallucination_safety_score": 0.95,
        "confidence_band": "high",
        "requires_abstention": False,
        "reasoning": "High confidence answer.",
    }

    calibration.update(overrides)

    return calibration
async def test_high_confidence_proceeds():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration()
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "PROCEED"
    assert output["should_abstain"] is False
    assert output["reasons"] == []
async def test_a7_abstention_is_respected():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                calibrated_confidence=0.90,
                requires_abstention=True,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"
    assert output["should_abstain"] is True

    assert (
        "A7 requires abstention."
        in output["reasons"]
    )
async def test_low_confidence_abstains():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                calibrated_confidence=0.30,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"
async def test_low_evidence_abstains():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                evidence_score=0.30,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"
async def test_low_factual_consistency_abstains():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                factual_consistency_score=0.40,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"
async def test_low_citation_support_abstains():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                citation_support_score=0.40,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"
async def test_low_hallucination_safety_abstains():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                hallucination_safety_score=0.40,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"
async def test_multiple_reasons_are_reported():

    agent = AbstentionDecisionAgent()

    state = {
        "confidence_calibration":
            make_calibration(
                calibrated_confidence=0.30,
                evidence_score=0.20,
                factual_consistency_score=0.30,
                citation_support_score=0.30,
                hallucination_safety_score=0.30,
            )
    }

    result = await agent.run(state)

    output = result[
        "abstention_decision"
    ]

    assert output["decision"] == "ABSTAIN"

    assert len(output["reasons"]) == 5
async def test_missing_calibration_fails():

    agent = AbstentionDecisionAgent()

    with pytest.raises(RuntimeError):

        await agent.run({})