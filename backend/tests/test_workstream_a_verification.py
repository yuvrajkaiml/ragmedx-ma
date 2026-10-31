"""
Phase 3 — Workstream A: Evidence Verification Tests

Tests for:
  - ClaimVerifier (claim_verifier.py)
  - ContradictionDetector (contradiction_detector.py)
  - SufficiencyAssessor (sufficiency.py)
  - EvidenceRerankingAgent Phase 3 fixes (evidence_reranking.py)
  - Provenance (doc_id / chunk_id propagation)

All 15 required tests plus 4 regression / provenance tests.
NLI model is mocked to avoid GPU/download in CI.

Evidence tier: SOFTWARE TEST
"""
from __future__ import annotations

import sys
import os

# Add backend root to sys.path so imports resolve in both pytest and direct run
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from verification.claim_verifier import ClaimVerifier
from verification.contradiction_detector import ContradictionDetector
from verification.sufficiency import SufficiencyAssessor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_mock_nli(label: str = "entailment", score: float = 0.85):
    """Return a mock NLI pipeline that returns a numpy-like array."""
    def _pipeline(passage, claim):
        # Simulate numpy array: [contradiction, entailment, neutral] logits
        label_to_idx = {"contradiction": 0, "entailment": 1, "neutral": 2}
        arr = np.array([0.05, 0.05, 0.05])
        idx = label_to_idx.get(label, 2)
        arr[idx] = 3.0  # high logit for the chosen label
        return arr
    return _pipeline


def make_verifier(nli_label="entailment", nli_score=0.85):
    """Return a ClaimVerifier with a mocked NLI pipeline."""
    mock_pipe = make_mock_nli(nli_label, nli_score)
    return ClaimVerifier(_nli_pipeline=mock_pipe)


SILDENAFIL_DOC = "sildenafil is contraindicated with nitrates due to severe hypotension risk"
PROLIA_DOC = "Prolia (denosumab) is used for the treatment of osteoporosis in postmenopausal women"


# ===========================================================================
# Test 1: valid claim with valid citation → supported
# ===========================================================================
def test_valid_claim_with_valid_citation():
    """Claim supported by its cited doc → label='supported'."""
    verifier = make_verifier(nli_label="entailment", nli_score=0.9)
    docs = {"doc_A": "Aspirin reduces fever and pain by inhibiting prostaglandin synthesis."}
    claim = "Aspirin reduces fever by inhibiting prostaglandin synthesis [doc_A]."
    result = verifier.verify_claim(claim, ["doc_A"], docs)
    assert result["id_exists"] is True, f"Expected id_exists=True, got {result['id_exists']}"
    assert result["label"] == "supported", f"Expected 'supported', got {result['label']!r}"
    assert result["score"] > 0.0


# ===========================================================================
# Test 2: citation exists but content doesn't support claim → unsupported
# ===========================================================================
def test_valid_citation_unrelated_claim():
    """Citation exists but NLI says neutral → 'unsupported'."""
    verifier = make_verifier(nli_label="neutral", nli_score=0.7)
    docs = {"doc_B": "Metformin is used for type 2 diabetes management."}
    claim = "Metformin cures all cancers [doc_B]."
    result = verifier.verify_claim(claim, ["doc_B"], docs)
    assert result["id_exists"] is True
    assert result["label"] == "unsupported", f"Expected 'unsupported', got {result['label']!r}"


# ===========================================================================
# Test 3: nonexistent doc_id → id_exists=False, label='unverifiable'
# ===========================================================================
def test_nonexistent_doc_id():
    """Claim cites doc_id not in retrieved set → id_exists=False, label='unverifiable'."""
    verifier = make_verifier()
    docs = {"doc_real": "Real document content here."}
    result = verifier.verify_claim("Some claim [doc_fake].", ["doc_fake"], docs)
    assert result["id_exists"] is False, f"Expected id_exists=False, got {result['id_exists']}"
    assert result["label"] == "unverifiable", f"Expected 'unverifiable', got {result['label']!r}"
    assert result["reason"] == "cited_id_not_in_retrieved_set"


# ===========================================================================
# Test 4: fabricated quote → low quote_fidelity, label='unsupported'
# ===========================================================================
def test_fabricated_quote():
    """Claim contains quote not present in cited doc → low fidelity → 'unsupported'."""
    verifier = make_verifier(nli_label="neutral", nli_score=0.4)
    docs = {"doc_C": "Aspirin is an NSAID used for pain relief."}
    # The claim makes up a quote that doesn't appear in the document
    claim = "doc_C states 'Aspirin cures all autoimmune diseases permanently' [doc_C]."
    result = verifier.verify_claim(claim, ["doc_C"], docs)
    assert result["id_exists"] is True
    # Low fidelity + no entailment → unsupported
    assert result["label"] in ("unsupported", "unverifiable"), f"Got {result['label']!r}"
    assert result["quote_fidelity"] < 0.6, f"Expected low fidelity, got {result['quote_fidelity']}"


# ===========================================================================
# Test 5: NLI returns contradiction → label='contradicted'
# ===========================================================================
def test_claim_contradicted_by_passage():
    """NLI returns contradiction → label='contradicted'."""
    verifier = make_verifier(nli_label="contradiction", nli_score=0.92)
    docs = {SILDENAFIL_DOC[:20]: SILDENAFIL_DOC}
    claim = "Sildenafil is safe to use with nitrates [" + SILDENAFIL_DOC[:20] + "]."
    result = verifier.verify_claim(claim, [SILDENAFIL_DOC[:20]], docs)
    assert result["id_exists"] is True
    assert result["label"] == "contradicted", f"Expected 'contradicted', got {result['label']!r}"
    assert result["entailment"]["label"] == "contradiction"


# ===========================================================================
# Test 6: two passages directly contradict each other → detected
# ===========================================================================
def test_contradictory_sources():
    """Two passages with opposing negation on the same topic → detected as contradiction."""
    detector = ContradictionDetector()
    passages = [
        {"id": "doc1", "text": "Metformin does not cause lactic acidosis in patients with normal renal function."},
        {"id": "doc2", "text": "Metformin causes lactic acidosis in patients with impaired renal function."},
    ]
    results = detector.detect_contradictions(passages)
    assert len(results) >= 1, f"Expected at least 1 contradiction, got {results}"
    types = {r["type"] for r in results}
    assert "true_contradiction" in types or "scope_difference" in types, f"Got types: {types}"


# ===========================================================================
# Test 7: on-topic passage that doesn't answer the specific claim → unverifiable
# ===========================================================================
def test_on_topic_non_answering_passage():
    """Passage is medically relevant but NLI is neutral → 'unsupported'."""
    verifier = make_verifier(nli_label="neutral", nli_score=0.5)
    docs = {
        "doc_cardio": (
            "Cardiovascular disease is the leading cause of death globally. "
            "Risk factors include hypertension, smoking, and diabetes."
        )
    }
    claim = "ACE inhibitors reduce mortality in heart failure patients [doc_cardio]."
    result = verifier.verify_claim(claim, ["doc_cardio"], docs)
    assert result["id_exists"] is True
    assert result["label"] in ("unsupported", "unverifiable"), f"Got {result['label']!r}"


# ===========================================================================
# Test 8: insufficient evidence → sufficient=False
# ===========================================================================
def test_insufficient_evidence():
    """Retrieved docs don't cover query → sufficient=False."""
    assessor = SufficiencyAssessor()
    result = assessor.assess(
        query="What is the first-line treatment for pulmonary arterial hypertension?",
        retrieved_docs=[
            {"cited_text": "Bananas are high in potassium.", "relevance_score": 0.05}
        ],
        verified_claims=[{"label": "unsupported"}],
    )
    assert result["sufficient"] is False, f"Expected sufficient=False, got {result['sufficient']}"
    assert "coverage" in result
    assert "reasoning" in result
    assert result["reasoning"]  # must be non-empty


# ===========================================================================
# Test 9: passages from different time periods → temporal_difference
# ===========================================================================
def test_conflicting_dates_contexts():
    """Passages from different time periods → classified as temporal_difference."""
    detector = ContradictionDetector()
    passages = [
        {
            "id": "old_guideline",
            "text": (
                "As of 1998, hormone replacement therapy was recommended "
                "for postmenopausal women to reduce cardiovascular risk."
            ),
            "publication_year": 1998,
        },
        {
            "id": "new_guideline",
            "text": (
                "Current 2020 guidelines do not recommend hormone replacement therapy "
                "for postmenopausal women primarily for cardiovascular risk reduction."
            ),
            "publication_year": 2020,
        },
    ]
    results = detector.detect_contradictions(passages)
    types = {r["type"] for r in results}
    assert "temporal_difference" in types or "true_contradiction" in types, (
        f"Expected temporal_difference or true_contradiction, got {types}"
    )


# ===========================================================================
# Test 10: invalid/malformed verifier JSON → handled gracefully
# ===========================================================================
def test_invalid_verifier_json():
    """
    Verifier encounters malformed input (empty cited_ids) → handles gracefully,
    returns 'unverifiable' with reason.
    """
    verifier = make_verifier()
    docs = {"doc_X": "Some document text."}
    # Empty cited_ids — should return unverifiable, not crash
    result = verifier.verify_claim("Some claim.", [], docs)
    assert isinstance(result, dict)
    assert "label" in result
    assert result["label"] == "unverifiable"
    assert "reason" in result


# ===========================================================================
# Test 11: NLI model throws exception → returns unverifiable with reason=verifier_error
# ===========================================================================
def test_verifier_failure():
    """NLI model throws exception → verify_claim returns {'label': 'unverifiable', 'reason': 'verifier_error'}."""
    def _failing_pipeline(passage, claim):
        raise RuntimeError("GPU OOM")

    verifier = ClaimVerifier(_nli_pipeline=_failing_pipeline)
    docs = {"doc_Y": "Some document text about treatment."}
    result = verifier.verify_claim("Treatment claim [doc_Y].", ["doc_Y"], docs)
    assert result["label"] == "unverifiable", f"Expected 'unverifiable', got {result['label']!r}"
    assert result["reason"] == "verifier_error", f"Expected 'verifier_error', got {result['reason']!r}"


# ===========================================================================
# Test 12: empty retrieval → all claims unverifiable, sufficient=False
# ===========================================================================
def test_empty_retrieval():
    """No docs retrieved → all claims unverifiable, sufficient=False."""
    verifier = make_verifier()
    docs = {}
    result = verifier.verify_claim("Some claim about medication.", ["doc_any"], docs)
    assert result["label"] == "unverifiable"

    assessor = SufficiencyAssessor()
    suf = assessor.assess(
        query="What drug treats condition X?",
        retrieved_docs=[],
        verified_claims=[result],
    )
    assert suf["sufficient"] is False
    assert suf["confidence"] == 0.0


# ===========================================================================
# Test 13: mixed supported/unsupported claims → all three labeled correctly
# ===========================================================================
def test_mixed_supported_unsupported():
    """2 supported + 1 unsupported claims → each labeled correctly."""
    supported_verifier = make_verifier(nli_label="entailment", nli_score=0.9)
    unsupported_verifier = make_verifier(nli_label="neutral", nli_score=0.6)

    docs = {
        "doc_sup_1": "Lisinopril is an ACE inhibitor used to treat hypertension.",
        "doc_sup_2": "Atorvastatin lowers LDL cholesterol by inhibiting HMG-CoA reductase.",
        "doc_unsup": "The study examined dietary patterns in healthy adults.",
    }

    result1 = supported_verifier.verify_claim(
        "Lisinopril treats hypertension [doc_sup_1].", ["doc_sup_1"], docs
    )
    result2 = supported_verifier.verify_claim(
        "Atorvastatin lowers LDL [doc_sup_2].", ["doc_sup_2"], docs
    )
    result3 = unsupported_verifier.verify_claim(
        "Atorvastatin eliminates all cancer risks [doc_unsup].", ["doc_unsup"], docs
    )

    assert result1["label"] == "supported", f"r1: {result1['label']!r}"
    assert result2["label"] == "supported", f"r2: {result2['label']!r}"
    assert result3["label"] == "unsupported", f"r3: {result3['label']!r}"


# ===========================================================================
# Test 14: evidence contradictory, sufficiency=False → abstention_recommended=True
# ===========================================================================
def test_should_abstain():
    """Contradictory evidence + insufficient evidence → abstention_recommended=True."""
    detector = ContradictionDetector()
    passages = [
        {
            "id": "p1",
            "text": "Drug X does not interact with anticoagulants in standard doses.",
        },
        {
            "id": "p2",
            "text": "Drug X causes serious bleeding when combined with anticoagulants.",
        },
    ]
    contradictions = detector.detect_contradictions(passages)
    # At minimum one contradiction should be detected
    assert len(contradictions) >= 1
    # At least one should recommend abstention
    abstention_recs = [c for c in contradictions if c.get("abstention_recommended")]
    assert len(abstention_recs) >= 1, f"Expected abstention_recommended=True in: {contradictions}"

    # Sufficiency check: with contradictions present → sufficient=False
    assessor = SufficiencyAssessor()
    suf = assessor.assess(
        query="Is Drug X safe with anticoagulants?",
        retrieved_docs=[
            {"cited_text": passages[0]["text"], "relevance_score": 0.8},
            {"cited_text": passages[1]["text"], "relevance_score": 0.8},
        ],
        verified_claims=[{"label": "contradicted"}, {"label": "contradicted"}],
    )
    assert suf["sufficient"] is False
    assert suf["contradictions_present"] is True


# ===========================================================================
# Test 15: clear supporting evidence, no contradictions → sufficient=True, answerable
# ===========================================================================
def test_should_answer():
    """Clear supporting evidence, no contradictions → sufficient=True, answerable."""
    assessor = SufficiencyAssessor()
    suf = assessor.assess(
        query="What is the mechanism of action of metformin in type 2 diabetes?",
        retrieved_docs=[
            {
                "cited_text": (
                    "Metformin works primarily by reducing hepatic glucose production "
                    "and improving insulin sensitivity in type 2 diabetes patients."
                ),
                "relevance_score": 0.92,
                "evidence_grade": "randomized_trial",
            },
            {
                "cited_text": (
                    "Metformin activates AMP-activated protein kinase, which inhibits "
                    "hepatic gluconeogenesis in diabetic patients."
                ),
                "relevance_score": 0.88,
                "evidence_grade": "systematic_review",
            },
        ],
        verified_claims=[
            {"label": "supported"},
            {"label": "supported"},
        ],
    )
    assert suf["sufficient"] is True, f"Expected sufficient=True, got {suf}"
    assert suf["contradictions_present"] is False
    assert suf["confidence"] > 0.0
    assert suf["reasoning"]  # non-empty


# ===========================================================================
# Provenance test: doc_id and chunk_id propagate from retrieval
# ===========================================================================
def test_provenance_ids_in_verify_claim():
    """
    Given a retrieval result with known doc_ids, verify_claim operates
    on those exact IDs (provenance chain).
    """
    verifier = make_verifier(nli_label="entailment", nli_score=0.9)
    known_id = "chunk-abc-123"
    docs = {known_id: "Statins reduce LDL cholesterol levels in hyperlipidemia."}
    claim = f"Statins reduce LDL [{known_id}]."
    result = verifier.verify_claim(claim, [known_id], docs)
    assert result["id_exists"] is True, "ID should propagate correctly"
    id_check = verifier.check_ids_exist([known_id], list(docs.keys()))
    assert id_check[known_id] is True


# ===========================================================================
# Reranker structural fix tests (3b)
# ===========================================================================

class _MockLLMWrapper:
    """Deterministic mock LLM wrapper for reranker tests."""
    def __init__(self, response_json: str):
        self._response = response_json

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._response


def test_reranker_rejects_unknown_ids():
    """
    Phase 3b fix: LLM-invented markers not in the input set are rejected.
    The reranker must only select markers from the retrieved set.
    """
    from app.agents.evidence_reranking import EvidenceRerankingAgent
    import asyncio

    # Input: two real docs
    evidence = [
        {"citation_marker": "[Doc 1]", "cited_text": SILDENAFIL_DOC, "relevance_score": 0.8, "cross_encoder_score": 0.8},
        {"citation_marker": "[Doc 2]", "cited_text": PROLIA_DOC, "relevance_score": 0.7, "cross_encoder_score": 0.7},
    ]
    # LLM returns a fabricated marker "[Doc 1 states trade name of sildenafil as Prolia]"
    fabricated_response = '{"selected_markers": ["[Doc 1 states trade name of sildenafil as Prolia]"], "contradictory": false, "reasoning": "Fabricated reasoning"}'
    llm = _MockLLMWrapper(fabricated_response)
    agent = EvidenceRerankingAgent(llm_wrapper=llm)

    async def _run():
        return await agent.run({"evidence": evidence, "original_question": "What is sildenafil?"})

    result = asyncio.run(_run())
    # The fabricated marker is not in the input set → should fall back to all evidence
    # (empty resolved set → original ranked order used) OR raise, but must NOT
    # return fabricated evidence as if it were real
    selected_markers = result["evidence_reranking"]["selected_markers"]
    for marker in selected_markers:
        assert marker in ("[Doc 1]", "[Doc 2]"), (
            f"Fabricated marker should not appear in results: {marker!r}"
        )


def test_reranker_reasoning_tagged_non_source():
    """
    Phase 3b fix: LLM reasoning text is tagged as non_source_text=True.
    It must never be treated as a retrieved passage.
    """
    from app.agents.evidence_reranking import EvidenceRerankingAgent
    import asyncio

    evidence = [
        {"citation_marker": "[Doc 1]", "cited_text": SILDENAFIL_DOC, "relevance_score": 0.8, "cross_encoder_score": 0.8},
    ]
    valid_response = '{"selected_markers": ["[Doc 1]"], "contradictory": false, "reasoning": "Doc 1 is the most relevant."}'
    llm = _MockLLMWrapper(valid_response)
    agent = EvidenceRerankingAgent(llm_wrapper=llm)

    async def _run():
        return await agent.run({"evidence": evidence, "original_question": "About sildenafil?"})

    result = asyncio.run(_run())
    assert result["evidence_reranking"].get("non_source_text") is True, (
        "Reasoning must be tagged as non_source_text=True"
    )


def test_reranker_malformed_json_safe_fallback():
    """
    Phase 3b fix: malformed LLM output triggers bounded retry then safe fallback.
    After max retries, returns original order (not an exception).
    """
    from app.agents.evidence_reranking import EvidenceRerankingAgent
    import asyncio

    evidence = [
        {"citation_marker": "[Doc 1]", "cited_text": SILDENAFIL_DOC, "relevance_score": 0.8, "cross_encoder_score": 0.8},
        {"citation_marker": "[Doc 2]", "cited_text": PROLIA_DOC, "relevance_score": 0.7, "cross_encoder_score": 0.7},
    ]
    # Consistently returns invalid JSON
    llm = _MockLLMWrapper("This is not valid JSON at all!!!")
    agent = EvidenceRerankingAgent(llm_wrapper=llm)

    async def _run():
        return await agent.run({"evidence": evidence, "original_question": "Question?"})

    result = asyncio.run(_run())
    # Should return original evidence order (safe fallback), not raise
    assert isinstance(result, dict)
    assert "evidence" in result
    assert len(result["evidence"]) == 2
    # Validated=False when fallback is used
    assert result["evidence_reranking"].get("validated") is False


# ===========================================================================
# Sildenafil/Prolia regression test (3b)
# ===========================================================================

def test_sildenafil_prolia_regression():
    """
    Regression test for the sildenafil/Prolia hallucination bug.

    The old reranker allowed the LLM to fabricate the statement
    'Doc 1 states trade name of sildenafil as Prolia' as reasoning
    without tagging it as non-source text.

    This test verifies that:
    1. The fabricated marker "[Doc 1 states trade name of sildenafil as Prolia]"
       is NOT accepted as a valid selected marker.
    2. The reasoning text is explicitly tagged as non_source_text=True.
    3. No marker appears in results that was not in the input retrieval set.

    This test MUST PASS on the fixed reranker (Phase 3b).
    The regression file records show it FAILED on the old code.
    """
    from app.agents.evidence_reranking import EvidenceRerankingAgent
    import asyncio

    evidence = [
        {
            "citation_marker": "[Doc 1]",
            "cited_text": SILDENAFIL_DOC,
            "title": "Sildenafil and nitrate interactions",
            "relevance_score": 0.9,
            "cross_encoder_score": 0.9,
        },
        {
            "citation_marker": "[Doc 2]",
            "cited_text": PROLIA_DOC,
            "title": "Prolia for osteoporosis",
            "relevance_score": 0.8,
            "cross_encoder_score": 0.8,
        },
    ]

    # Simulate the exact hallucination: LLM invents a fabricated statement as selected_marker
    hallucinated_response = (
        '{"selected_markers": ["[Doc 1 states trade name of sildenafil as Prolia]"], '
        '"contradictory": false, '
        '"reasoning": "Doc 1 states trade name of sildenafil as Prolia"}'
    )
    llm = _MockLLMWrapper(hallucinated_response)
    agent = EvidenceRerankingAgent(llm_wrapper=llm)

    async def _run():
        return await agent.run({
            "evidence": evidence,
            "original_question": "What is the trade name of sildenafil?",
        })

    result = asyncio.run(_run())

    # ASSERTION 1: No fabricated marker appears in results
    selected = result["evidence_reranking"]["selected_markers"]
    for marker in selected:
        assert "states trade name" not in marker, (
            f"Fabricated marker text leaked into selected_markers: {marker!r}"
        )
    # All selected markers must be from the original input set
    input_markers = {"[Doc 1]", "[Doc 2]"}
    for marker in selected:
        assert marker in input_markers, (
            f"Unknown marker not in input set: {marker!r}"
        )

    # ASSERTION 2: Reasoning is tagged as non_source_text
    assert result["evidence_reranking"].get("non_source_text") is True, (
        "LLM reasoning must be tagged as non_source_text=True to prevent "
        "downstream agents from treating it as retrieved evidence."
    )
