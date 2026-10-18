"""Focused unit tests for PMID-aware near-duplicate collapsing in
``HybridRetriever._fuse_results``.

These tests construct ``MedicalChunk`` objects directly and feed ranked
retrieval lists into ``_fuse_results``. The dense/sparse retrievers are not
invoked by ``_fuse_results`` (it uses only ``chunk.id``, ``chunk.content``,
``chunk.metadata``, and ``chunk.document_id``), so they are supplied as mocks
and no model or index is loaded.
"""

from unittest.mock import Mock

from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.hybrid_retriever import HybridRetriever


def make_retriever() -> HybridRetriever:
    """Build a HybridRetriever whose retrievers are never called here."""
    return HybridRetriever(
        dense_retriever=Mock(),
        sparse_retriever=Mock(),
        query_expander=Mock(),
    )


def chunk(chunk_id, content, pmid=None, document_id=None):
    metadata = {"pmid": pmid} if pmid is not None else {}
    return MedicalChunk(
        id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        content=content,
        metadata=metadata,
    )


def pmids(fused):
    return [c.metadata.get("pmid") for c, _ in fused]


def ids(fused):
    return [c.id for c, _ in fused]


def test_same_pmid_near_identical_chunks_collapse_to_one_best_ranked():
    """3 chunks with the same PMID and identical text collapse to one copy,
    keeping the best-RRF-ranked (highest fused score) copy."""
    retriever = make_retriever()

    text = "Riociguat improves exercise capacity in pulmonary hypertension."
    # Three distinct chunk ids, same PMID, identical text. Placing them at
    # different ranks across two lists gives the first one the best RRF score.
    a = chunk("c_s0", text, pmid="25395817")
    b = chunk("c_s1", text, pmid="25395817")
    c = chunk("c_s6", text, pmid="25395817")
    other = chunk("other", "Unrelated passage about diabetes.", pmid="999")

    list_one = [(a, 0.9), (b, 0.8), (c, 0.7), (other, 0.6)]
    list_two = [(a, 0.9), (b, 0.8), (c, 0.7)]

    fused = retriever._fuse_results([list_one, list_two])

    result_pmids = pmids(fused)
    assert result_pmids.count("25395817") == 1
    # The kept copy is the best-ranked one (a), which has the highest RRF score.
    kept = [c for c, _ in fused if c.metadata.get("pmid") == "25395817"][0]
    assert kept.id == "c_s0"
    # The unrelated PMID survives.
    assert "999" in result_pmids


def test_different_pmids_are_not_merged():
    """Chunks with different PMIDs and identical text are all kept."""
    retriever = make_retriever()

    text = "Hypertension management with first-line agents."
    a = chunk("a", text, pmid="111")
    b = chunk("b", text, pmid="222")
    c = chunk("c", text, pmid="333")

    fused = retriever._fuse_results([[(a, 0.9), (b, 0.8), (c, 0.7)]])

    assert sorted(pmids(fused)) == ["111", "222", "333"]
    assert len(fused) == 3


def test_same_pmid_low_overlap_passages_are_not_merged():
    """Two chunks with the same PMID but Jaccard < 0.90 both survive."""
    retriever = make_retriever()

    a = chunk(
        "a",
        "Metformin reduces hepatic glucose output in type 2 diabetes.",
        pmid="20840734",
    )
    b = chunk(
        "b",
        "Lifestyle counseling and dietary sodium restriction lower "
        "cardiovascular risk markedly over ten years.",
        pmid="20840734",
    )

    fused = retriever._fuse_results([[(a, 0.9), (b, 0.8)]])

    assert len(fused) == 2
    assert ids(fused) == ["a", "b"]


def test_pmidless_chunks_with_distinct_document_ids_never_merge():
    """PMID-less chunks keyed by distinct document_ids are never merged even
    when their text is identical."""
    retriever = make_retriever()

    text = "A clinical vignette describing a patient presentation."
    a = chunk("medqa_a", text, pmid=None, document_id="medqa_doc_1")
    b = chunk("medqa_b", text, pmid=None, document_id="medqa_doc_2")

    fused = retriever._fuse_results([[(a, 0.9), (b, 0.8)]])

    assert len(fused) == 2
    assert ids(fused) == ["medqa_a", "medqa_b"]


def test_stable_descending_order_of_survivors_is_preserved():
    """Surviving non-duplicates keep their RRF-descending order; a collapsed
    duplicate does not reorder the rest."""
    retriever = make_retriever()

    top = chunk("top", "First-line antihypertensive therapy overview.", pmid="1")
    dup_text = "Duplicate near-identical passage about renal outcomes."
    mid1 = chunk("mid1", dup_text, pmid="2")
    mid2 = chunk("mid2", dup_text, pmid="2")  # near-identical duplicate of mid1
    tail = chunk("tail", "Distinct passage on metformin pharmacology.", pmid="3")

    # mid1 ranked above mid2 so mid1 is the kept copy; order should be
    # top, mid1, tail (mid2 dropped), preserving descending RRF order.
    fused = retriever._fuse_results(
        [[(top, 0.9), (mid1, 0.8), (mid2, 0.7), (tail, 0.6)]]
    )

    assert ids(fused) == ["top", "mid1", "tail"]
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
