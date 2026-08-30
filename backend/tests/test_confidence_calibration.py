import pytest

from app.agents.confidence_calibration import (
    ConfidenceCalibrationAgent,
)
async def test_high_confidence_answer():

    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [
            {
                "relevance_score": 0.95,
                "cross_encoder_score": 0.90,
                "evidence_quality_score": 0.95,
            }
        ],
        "critic_verdict": {
            "factual_consistency": 0.95,
            "citation_support": 0.95,
            "hallucination_risk": 0.05,
            "confidence": 0.95,
        },
    }

    result = await agent.run(state)

    output = result[
        "confidence_calibration"
    ]

    assert (
        output["calibrated_confidence"]
        >= 0.80
    )

    assert output["confidence_band"] == "high"

    assert (
        output["requires_abstention"]
        is False
    )
async def test_low_confidence_requires_abstention():

    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [
            {
                "relevance_score": 0.20,
                "cross_encoder_score": 0.15,
                "evidence_quality_score": 0.20,
            }
        ],
        "critic_verdict": {
            "factual_consistency": 0.30,
            "citation_support": 0.25,
            "hallucination_risk": 0.80,
            "confidence": 0.30,
        },
    }

    result = await agent.run(state)

    output = result[
        "confidence_calibration"
    ]

    assert (
        output["calibrated_confidence"]
        < 0.50
    )

    assert (
        output["requires_abstention"]
        is True
    )
async def test_high_hallucination_risk_requires_abstention():

    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [
            {
                "relevance_score": 0.90,
                "cross_encoder_score": 0.90,
                "evidence_quality_score": 0.90,
            }
        ],
        "critic_verdict": {
            "factual_consistency": 0.90,
            "citation_support": 0.90,
            "hallucination_risk": 0.75,
            "confidence": 0.90,
        },
    }

    result = await agent.run(state)

    output = result[
        "confidence_calibration"
    ]

    assert (
        output["requires_abstention"]
        is True
    )
async def test_missing_critic_verdict():

    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [
            {
                "relevance_score": 0.90,
                "cross_encoder_score": 0.90,
                "evidence_quality_score": 0.90,
            }
        ]
    }

    with pytest.raises(RuntimeError):

        await agent.run(state)
async def test_missing_evidence_produces_low_confidence():

    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [],
        "critic_verdict": {
            "factual_consistency": 0.90,
            "citation_support": 0.90,
            "hallucination_risk": 0.05,
            "confidence": 0.90,
        },
    }

    result = await agent.run(state)

    output = result[
        "confidence_calibration"
    ]

    assert (
        output["evidence_score"] == 0.0
    )

    assert (
        output["requires_abstention"]
        is True
    )