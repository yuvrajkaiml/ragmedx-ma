from app.agents.evidence_reranking import EvidenceRerankingAgent


def test_reranker_resolves_legacy_marker_only_by_unambiguous_ordinal():
    ranked = [
        {"citation_marker": "[Doc 1]"},
        {"citation_marker": "[Doc 2]"},
    ]

    assert EvidenceRerankingAgent._resolve_selected_markers(["[E2]"], ranked) == {"[Doc 2]"}
    assert EvidenceRerankingAgent._resolve_selected_markers(["[E9]", "[unknown]"], ranked) == set()
