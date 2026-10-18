"""
Regression tests for the citation-marker contract shared by the Generator and
Critic agents.

Root cause being guarded against: the generator prompt instructs the model to
emit `[Doc N]` markers, and the native retriever emits `[Doc N]`, but the old
extractor regex (`\\[[A-Za-z0-9_-]+\\]`) could not match a marker containing a
space. Correct citations were silently dropped from `answer_citations`, and a
hallucinated `[Doc 9]` slipped past the generator's own unsupported-citation
guard. These tests pin the fixed, consistent behavior.

Covers (per task): valid citations, citations to missing evidence, and
unsupported claims. A resolvable marker is explicitly NOT treated as proof of
claim support.
"""

import pytest

from app.agents.critic import CriticAgent
from app.agents.generator import GeneratorAgent
from app.rag.citations.markers import (
    extract_markers,
    evidence_markers,
    normalize_marker,
)


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
# Marker utility unit tests
# ----------------------------------------------------------------------------

def test_normalize_marker_forms():
    assert normalize_marker("E1") == "[E1]"
    assert normalize_marker("[E1]") == "[E1]"
    assert normalize_marker("[Doc 1]") == "[Doc 1]"
    assert normalize_marker("[Doc  1]") == "[Doc 1]"  # collapses whitespace
    assert normalize_marker("") == ""
    assert normalize_marker(None) == ""


def test_extract_markers_handles_doc_and_e_forms():
    text = "Claim one [Doc 1]. Claim two [Doc 12]. Legacy [E2]."
    assert extract_markers(text) == {"[Doc 1]", "[Doc 12]", "[E2]"}


def test_extract_markers_empty():
    assert extract_markers("") == set()
    assert extract_markers("no markers here") == set()


def test_evidence_markers_fallback_positional():
    ev = [{"citation_marker": "[Doc 1]"}, {"cited_text": "x"}]
    assert evidence_markers(ev) == ["[Doc 1]", "[Doc 2]"]


# ----------------------------------------------------------------------------
# VALID CITATIONS
# ----------------------------------------------------------------------------

async def test_generator_extracts_doc_marker_regression():
    """The exact bug: a `[Doc 1]` answer must populate answer_citations."""
    llm = FakeLLM("Amyloid beta drives AD progression [Doc 1].")
    agent = GeneratorAgent(llm_wrapper=llm)
    state = {
        "original_question": "Role of amyloid beta in Alzheimer's?",
        "evidence": [
            {
                "citation_marker": "[Doc 1]",
                "title": "AD Review",
                "journal": "Neuro Journal",
                "publication_year": 2024,
                "cited_text": "Amyloid beta plays a role in AD progression.",
            }
        ],
    }
    result = await agent.run(state)
    assert result["answer_citations"] == ["[Doc 1]"]
    assert result["agent_trace"][0]["details"]["citation_count"] == 1


async def test_generator_still_supports_legacy_e_marker():
    """Backward compatibility: `[E1]` markers still extracted."""
    llm = FakeLLM("Increased thirst is common [E1].")
    agent = GeneratorAgent(llm_wrapper=llm)
    state = {
        "original_question": "Symptoms of diabetes?",
        "evidence": [{"citation_marker": "[E1]", "cited_text": "Increased thirst."}],
    }
    result = await agent.run(state)
    assert result["answer_citations"] == ["[E1]"]


# ----------------------------------------------------------------------------
# CITATIONS TO MISSING EVIDENCE
# ----------------------------------------------------------------------------

async def test_generator_rejects_doc_marker_to_missing_evidence():
    """`[Doc 9]` not in evidence must now be caught by the generator guard.

    Before the fix the guard was blind to `[Doc N]` markers, so this would have
    passed silently. It must now raise.
    """
    llm = FakeLLM("ACE inhibitors reduce mortality [Doc 9].")
    agent = GeneratorAgent(llm_wrapper=llm)
    state = {
        "original_question": "Do ACE inhibitors reduce mortality?",
        "evidence": [{"citation_marker": "[Doc 1]", "cited_text": "ACE inhibitors reduce mortality in HFrEF."}],
    }
    with pytest.raises(RuntimeError):
        await agent.run(state)


async def test_critic_flags_missing_evidence_citation():
    """Critic's deterministic verifier blocks an unresolved `[Doc 9]` marker."""
    # High LLM scores on purpose: the deterministic verifier must still block.
    critic = CriticAgent(
        llm_wrapper=FakeCriticLLM(
            '{"passed": true, "factual_consistency": 1.0, "citation_support": 1.0, '
            '"hallucination_risk": 0.0, "issues": [], "revision_instructions": null, '
            '"confidence": 1.0}'
        )
    )
    result = await critic.run({
        "original_question": "Do ACE inhibitors reduce mortality?",
        "answer": "ACE inhibitors reduce mortality in HFrEF [Doc 9].",
        "evidence": [{"citation_marker": "[Doc 1]", "cited_text": "ACE inhibitors reduce mortality in HFrEF."}],
        "review_round": 1,
    })
    v = result["critic_verdict"]
    ver = v["evidence_verification"]
    assert "[Doc 9]" in ver["unresolved_markers"]
    assert ver["blocking_issues"]
    assert v["passed"] is False
    assert v["citation_support"] == 0.0


# ----------------------------------------------------------------------------
# UNSUPPORTED CLAIMS (marker resolves, but claim is not supported)
# ----------------------------------------------------------------------------

async def test_critic_marker_resolves_but_claim_unsupported():
    """A resolvable marker must NOT be treated as proof of claim support.

    The marker `[Doc 1]` exists in evidence (deterministic verifier does not
    block), but the LLM critic judges the claim unsupported. The verdict must
    reflect the LLM's low support, not pass because the marker resolved.
    """
    critic = CriticAgent(
        llm_wrapper=FakeCriticLLM(
            '{"passed": false, "factual_consistency": 0.0, "citation_support": 0.0, '
            '"hallucination_risk": 1.0, '
            '"issues": ["Cited span does not support the aspirin claim."], '
            '"revision_instructions": "Only state claims supported by the evidence span.", '
            '"confidence": 0.2}'
        )
    )
    result = await critic.run({
        "original_question": "Does daily aspirin prevent all heart attacks?",
        "answer": "Daily aspirin eliminates all myocardial infarction risk [Doc 1].",
        "evidence": [{
            "citation_marker": "[Doc 1]",
            "cited_text": "Adequate daily water intake supports normal kidney function.",
        }],
        "review_round": 1,
    })
    v = result["critic_verdict"]
    ver = v["evidence_verification"]
    # Marker resolves -> no blocking issue from the deterministic layer.
    assert ver["unresolved_markers"] == []
    assert ver["blocking_issues"] == []
    assert "[Doc 1]" in ver["mapped_markers"]
    # ...yet the claim is judged unsupported.
    assert v["passed"] is False
    assert v["citation_support"] == 0.0
    assert v["factual_consistency"] == 0.0


async def test_critic_passes_supported_doc_marker_answer():
    """Positive control: a supported `[Doc 1]` answer passes with no blocks."""
    critic = CriticAgent(
        llm_wrapper=FakeCriticLLM(
            '{"passed": true, "factual_consistency": 1.0, "citation_support": 1.0, '
            '"hallucination_risk": 0.0, "issues": [], "revision_instructions": null, '
            '"confidence": 0.95}'
        )
    )
    result = await critic.run({
        "original_question": "Role of amyloid beta in Alzheimer's?",
        "answer": "Amyloid beta plays a role in AD progression [Doc 1].",
        "evidence": [{
            "citation_marker": "[Doc 1]",
            "cited_text": "Amyloid beta plays an important role in AD progression.",
        }],
        "review_round": 1,
    })
    v = result["critic_verdict"]
    assert v["evidence_verification"]["unresolved_markers"] == []
    assert v["evidence_verification"]["blocking_issues"] == []
    assert v["passed"] is True
    assert v["citation_support"] == 1.0
