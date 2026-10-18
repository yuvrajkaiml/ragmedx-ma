"""
Regression tests for the ``[Doc E1]`` citation-marker bug.

Root cause: ``CITATION_MARKER_PATTERN`` previously only accepted digits after
``Doc`` (``\\[(?:Doc\\s+\\d+|...)\\]``), so an alpha-prefixed id such as
``[Doc E1]`` -- emitted by Qwen3-8B in the 2026-10-18 smoke test (case10) --
matched nothing. The marker was silently dropped from ``answer_citations`` even
though the critic reported ``citation_support = 1.0``. The ``Doc`` branch now
accepts an optional leading letter (``[A-Za-z]?\\d+``), so ``[Doc E1]`` is
captured whole while ``[Doc 1]`` and legacy ``[E1]`` still match.

Mirrors the conventions of test_citation_marker_consistency.py (async tests,
asyncio_mode=auto, FakeLLM / FakeCriticLLM stubs).
"""

import pytest

from app.agents.critic import CriticAgent
from app.agents.generator import GeneratorAgent
from app.rag.citations.markers import extract_markers, normalize_marker


class FakeLLM:
    """Returns a fixed string for generate(); used for the generator path."""

    def __init__(self, response):
        self.response = response

    def generate(self, prompt, temperature=0.0):
        return self.response


class FakeCriticLLM:
    """Returns a fixed JSON verdict; used to exercise the critic LLM path."""

    def __init__(self, json_text):
        self.json_text = json_text

    def generate(self, prompt, temperature=0.0):
        return self.json_text


# ----------------------------------------------------------------------------
# Direct extract_markers / normalize_marker assertions
# ----------------------------------------------------------------------------

def test_extract_doc_e1_marker():
    """The exact bug: ``[Doc E1]`` must now extract (previously dropped)."""
    assert extract_markers("ARBs lower blood pressure [Doc E1].") == {"[Doc E1]"}


def test_extract_mixed_doc_forms_including_e1():
    text = "A [Doc 1]. B [Doc 12]. C [Doc E1]. Legacy [E1]."
    assert extract_markers(text) == {"[Doc 1]", "[Doc 12]", "[Doc E1]", "[E1]"}


def test_normalize_doc_e1_collapses_whitespace():
    assert normalize_marker("[Doc  E1]") == "[Doc E1]"
    assert normalize_marker("[Doc E1]") == "[Doc E1]"


# ----------------------------------------------------------------------------
# VALID CITATION: answer cites [Doc E1] (and [Doc 1]) present in evidence
# ----------------------------------------------------------------------------

async def test_generator_extracts_doc_e1_and_doc_1():
    """A ``[Doc E1]`` + ``[Doc 1]`` answer must populate answer_citations."""
    llm = FakeLLM(
        "ARBs reduce blood pressure [Doc E1]. Thiazides are first-line [Doc 1]."
    )
    agent = GeneratorAgent(llm_wrapper=llm)
    state = {
        "original_question": "First-line therapy for hypertension?",
        "evidence": [
            {
                "citation_marker": "[Doc E1]",
                "cited_text": "ARBs reduce blood pressure.",
            },
            {
                "citation_marker": "[Doc 1]",
                "cited_text": "Thiazide diuretics are first-line for hypertension.",
            },
        ],
    }
    result = await agent.run(state)
    assert result["answer_citations"] == ["[Doc 1]", "[Doc E1]"]
    assert result["agent_trace"][0]["details"]["citation_count"] == 2


# ----------------------------------------------------------------------------
# MISSING EVIDENCE: [Doc E9] not in evidence -> generator guard + critic block
# ----------------------------------------------------------------------------

async def test_generator_rejects_doc_e9_missing_evidence():
    """``[Doc E9]`` absent from evidence must be caught by the generator guard."""
    llm = FakeLLM("Metformin lowers HbA1c [Doc E9].")
    agent = GeneratorAgent(llm_wrapper=llm)
    state = {
        "original_question": "Does metformin lower HbA1c?",
        "evidence": [
            {"citation_marker": "[Doc E1]", "cited_text": "Metformin lowers HbA1c."}
        ],
    }
    with pytest.raises(RuntimeError):
        await agent.run(state)


async def test_critic_flags_doc_e9_missing_evidence():
    """Critic's deterministic verifier blocks an unresolved ``[Doc E9]`` marker."""
    critic = CriticAgent(
        llm_wrapper=FakeCriticLLM(
            '{"passed": true, "factual_consistency": 1.0, "citation_support": 1.0, '
            '"hallucination_risk": 0.0, "issues": [], "revision_instructions": null, '
            '"confidence": 1.0}'
        )
    )
    result = await critic.run({
        "original_question": "Does metformin lower HbA1c?",
        "answer": "Metformin lowers HbA1c [Doc E9].",
        "evidence": [
            {"citation_marker": "[Doc E1]", "cited_text": "Metformin lowers HbA1c."}
        ],
        "review_round": 1,
    })
    v = result["critic_verdict"]
    ver = v["evidence_verification"]
    assert "[Doc E9]" in ver["unresolved_markers"]
    assert ver["blocking_issues"]
    assert v["passed"] is False
    assert v["citation_support"] == 0.0


# ----------------------------------------------------------------------------
# RESOLVABLE-but-UNSUPPORTED: [Doc E1] resolves, but claim is unsupported
# ----------------------------------------------------------------------------

async def test_critic_doc_e1_resolves_but_claim_unsupported():
    """``[Doc E1]`` resolves to evidence, yet the claim is not supported.

    The deterministic verifier must not block (marker resolves), but the LLM
    critic's low support must stand -- a resolvable marker is not proof of
    support.
    """
    critic = CriticAgent(
        llm_wrapper=FakeCriticLLM(
            '{"passed": false, "factual_consistency": 0.0, "citation_support": 0.0, '
            '"hallucination_risk": 1.0, '
            '"issues": ["Cited span does not support the cure claim."], '
            '"revision_instructions": "Only state claims supported by the evidence span.", '
            '"confidence": 0.2}'
        )
    )
    result = await critic.run({
        "original_question": "Does metformin cure type 2 diabetes?",
        "answer": "Metformin permanently cures type 2 diabetes [Doc E1].",
        "evidence": [
            {
                "citation_marker": "[Doc E1]",
                "cited_text": "Metformin helps control blood glucose in type 2 diabetes.",
            }
        ],
        "review_round": 1,
    })
    v = result["critic_verdict"]
    ver = v["evidence_verification"]
    assert ver["unresolved_markers"] == []
    assert ver["blocking_issues"] == []
    assert "[Doc E1]" in ver["mapped_markers"]
    assert v["passed"] is False
    assert v["citation_support"] == 0.0
    assert v["factual_consistency"] == 0.0
