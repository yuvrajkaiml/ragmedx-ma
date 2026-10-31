import pytest

from app.agents.abstention_decision import (
    COPY_CONTRADICTORY_EVIDENCE,
    COPY_MODEL_FAILURE,
    COPY_OUT_OF_DOMAIN,
    COPY_RETRIEVAL_FAILURE,
    AbstentionDecisionAgent,
)
from app.agents.confidence_calibration import (
    CONFIDENCE_THRESHOLD,
    ConfidenceCalibrationAgent,
)
from app.agents.critic import CriticAgent
from app.agents.generator import GeneratorAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.llm.parser import LLMParseError, ModelLoadError, parse_json_from_llm
from app.llm.prompt_templates import load_prompt_template, render_prompt


class StaticLLM:
    """Deterministic static LLM for component testing."""
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, **kwargs):
        return self.response, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}


# ==============================================================================
# 1. LocalLLMWrapper & Runtime Mode Tests
# ==============================================================================

def test_mock_mode_is_explicit_and_does_not_claim_real_inference():
    """Verify that mock mode is explicit and returns is_mock=True metadata."""
    llm = LocalLLMWrapper(mode=LLMRuntimeMode.MOCK)
    assert llm.load_model() is True
    assert llm.active_runtime_mode == LLMRuntimeMode.MOCK

    response, usage = llm.generate("What is diabetes?")
    assert response is not None
    assert usage["is_mock"] is True
    assert usage["mode"] == "mock"


def test_local_mode_raises_error_without_silent_fallback():
    """Verify that local mode raises ModelLoadError if CUDA/model is unavailable."""
    llm = LocalLLMWrapper(mode=LLMRuntimeMode.LOCAL, allow_fallback=False)
    with pytest.raises(ModelLoadError) as exc_info:
        llm.load_model()

    assert "Failed to load local model" in str(exc_info.value) or "CUDA is not available" in str(exc_info.value)
    assert llm.is_loaded is False


def test_auto_mode_without_fallback_raises_model_load_error():
    """Verify that auto mode without explicit fallback raises ModelLoadError."""
    llm = LocalLLMWrapper(mode=LLMRuntimeMode.AUTO, allow_fallback=False)
    with pytest.raises(ModelLoadError):
        llm.load_model()


def test_auto_mode_with_fallback_marks_mock_metadata():
    """Verify that auto mode with explicit fallback marks runtime mode as mock."""
    llm = LocalLLMWrapper(mode=LLMRuntimeMode.AUTO, allow_fallback=True)
    loaded = llm.load_model()
    # On CPU-only system without CUDA, it falls back to mock with is_mock=True
    assert llm.active_runtime_mode == LLMRuntimeMode.MOCK
    response, usage = llm.generate("Medical query")
    assert usage["is_mock"] is True


# ==============================================================================
# 2. Robust JSON Parser Tests
# ==============================================================================

def test_parser_handles_markdown_code_fences():
    """Verify parser extracts JSON from markdown ```json ``` code blocks."""
    raw = "```json\n{\"medical_specialty\": \"Cardiology\", \"confidence\": 0.95}\n```"
    parsed = parse_json_from_llm(raw)
    assert parsed["medical_specialty"] == "Cardiology"
    assert parsed["confidence"] == 0.95


def test_parser_handles_conversational_preambles():
    """Verify parser extracts JSON when surrounded by conversational preambles."""
    raw = (
        "Here is the requested clinical evaluation in JSON format:\n"
        "{\n"
        '    "passed": true,\n'
        '    "factual_consistency": 0.90,\n'
        '    "citation_support": 0.85,\n'
        '    "hallucination_risk": 0.05,\n'
        '    "issues": [],\n'
        '    "revision_instructions": null,\n'
        '    "confidence": 0.92\n'
        "}\n"
        "Hope this helps with your analysis!"
    )
    parsed = parse_json_from_llm(raw)
    assert parsed["passed"] is True
    assert parsed["factual_consistency"] == 0.90


def test_parser_rejects_unrecoverable_malformed_json():
    """Verify parser raises typed LLMParseError on unrecoverable input."""
    with pytest.raises(LLMParseError):
        parse_json_from_llm("There is no JSON in this response.")


# ==============================================================================
# 3. Prompt Centralization & Versioning Tests
# ==============================================================================

def test_prompt_version_loading_and_rendering():
    """Verify prompt loading and Jinja2 rendering under prompts/v1/."""
    template = load_prompt_template("query_understanding", version="v1")
    assert template is not None
    assert "{{ question }}" in template

    rendered = render_prompt("query_understanding", {"question": "What is hypertension?"}, version="v1")
    assert "What is hypertension?" in rendered


# ==============================================================================
# 4. Confidence Calibration Formula & Component Tests
# ==============================================================================
async def test_confidence_calibration_exact_four_components():
    """
    Verify exact 4-component calculation:
    composite = 0.25 * retrieval + 0.30 * evidence_quality + 0.25 * consistency + 0.20 * agreement
    """
    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [
            {
                "citation_marker": "[E1]",
                "relevance_score": 0.80,
                "evidence_quality_score": 0.90,
                "cross_encoder_score": 0.90,
            }
        ],
        "critic_verdict": {
            "passed": True,
            "factual_consistency": 0.90,
            "citation_support": 0.80,
            "hallucination_risk": 0.10,  # safety = 0.90, consistency = (0.90+0.90)/2 = 0.90
            "confidence": 0.90,
        },
    }

    result = await agent.run(state)
    output = result["confidence_calibration"]

    # retrieval = 0.80
    # evidence_quality = (0.90 + 0.90)/2 = 0.90
    # consistency = (0.90 + 0.90)/2 = 0.90
    # agreement = 0.80
    # expected = 0.25*0.80 + 0.30*0.90 + 0.25*0.90 + 0.20*0.80
    #          = 0.20 + 0.27 + 0.225 + 0.16 = 0.855
    expected_composite = 0.855
    assert abs(output["calibrated_confidence"] - expected_composite) < 0.01
    assert output["confidence_band"] == "high"
    assert output["requires_abstention"] is False
async def test_confidence_calibration_abstains_below_threshold():
    """Verify confidence below 0.65 sets requires_abstention=True."""
    agent = ConfidenceCalibrationAgent()

    state = {
        "evidence": [
            {
                "citation_marker": "[E1]",
                "relevance_score": 0.40,
                "evidence_quality_score": 0.40,
                "cross_encoder_score": 0.40,
            }
        ],
        "critic_verdict": {
            "passed": False,
            "factual_consistency": 0.40,
            "citation_support": 0.40,
            "hallucination_risk": 0.60,
            "revision_instructions": "Revise answer.",
            "confidence": 0.50,
        },
    }

    result = await agent.run(state)
    output = result["confidence_calibration"]
    assert output["calibrated_confidence"] < CONFIDENCE_THRESHOLD
    assert output["requires_abstention"] is True


# ==============================================================================
# 5. Abstention Safety Gates & Standard Copy Tests
# ==============================================================================
async def test_abstention_on_contradictory_evidence():
    """Verify contradictory evidence triggers structured abstention with standard copy."""
    agent = AbstentionDecisionAgent()
    state = {
        "contradictory_evidence": True,
        "confidence_calibration": {
            "calibrated_confidence": 0.85,
            "evidence_score": 0.90,
            "factual_consistency_score": 0.90,
            "citation_support_score": 0.90,
            "hallucination_safety_score": 0.90,
            "requires_abstention": False,
        },
    }
    result = await agent.run(state)
    output = result["abstention_decision"]
    assert output["decision"] == "ABSTAIN"
    assert COPY_CONTRADICTORY_EVIDENCE in output["reasons"]
async def test_abstention_on_out_of_domain():
    """Verify out-of-domain query triggers abstention with standard copy."""
    agent = AbstentionDecisionAgent()
    state = {
        "out_of_domain": True,
        "confidence_calibration": {
            "calibrated_confidence": 0.85,
            "evidence_score": 0.90,
            "factual_consistency_score": 0.90,
            "citation_support_score": 0.90,
            "hallucination_safety_score": 0.90,
            "requires_abstention": False,
        },
    }
    result = await agent.run(state)
    output = result["abstention_decision"]
    assert output["decision"] == "ABSTAIN"
    assert COPY_OUT_OF_DOMAIN in output["reasons"]
async def test_generator_rejects_hallucinated_citation():
    """Verify generator rejects citation markers not present in supplied evidence."""
    fake_llm = StaticLLM("Metformin is first-line pharmacotherapy for type 2 diabetes [E99].")
    agent = GeneratorAgent(llm_wrapper=fake_llm)

    state = {
        "original_question": "What is first-line therapy for T2D?",
        "evidence": [
            {
                "citation_marker": "[E1]",
                "cited_text": "Metformin is standard initial therapy for type 2 diabetes.",
            }
        ],
    }

    with pytest.raises(RuntimeError) as exc_info:
        await agent.run(state)

    assert "unsupported citation markers" in str(exc_info.value)
