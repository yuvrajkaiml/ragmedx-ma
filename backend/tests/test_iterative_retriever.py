
from app.rag.chunking.models import MedicalChunk
from app.rag.retrievers.iterative_retriever import IterativeRetriever


class FakeHybridRetriever:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append(query)
        return self.responses.get(query, [])[:top_k]


def chunk(chunk_id, content):
    return MedicalChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        title="Medical research",
        journal="Test Journal",
        publication_year=2024,
    )


def test_iterative_retrieval_adds_follow_up_evidence():
    first = chunk("1", "Treatment therapy evidence for patients.")
    second = chunk("2", "Safety adverse effect outcomes evidence.")

    retriever = FakeHybridRetriever({
        "What are the treatment outcomes?": [(first, 0.9)],
        "What are the treatment outcomes? evidence about outcomes": [
            (second, 0.8)
        ],
    })

    result = IterativeRetriever(
        retriever,
        max_iterations=3,
        top_k=5,
    ).retrieve("What are the treatment outcomes?")

    assert len(result.evidence) == 2
    assert result.iteration_count <= 3
    assert len(retriever.calls) >= 2


def test_duplicate_chunks_are_not_repeated():
    item = chunk("same", "Treatment outcome evidence for patients.")
    retriever = FakeHybridRetriever({
        "What are the treatment outcomes?": [(item, 0.9)],
        "What are the treatment outcomes? evidence about outcomes": [
            (item, 0.95)
        ],
    })

    result = IterativeRetriever(
        retriever,
        gap_analyzer=lambda query, evidence: (
            ["outcomes"] if len(evidence) == 1 else []
        ),
    ).retrieve("What are the treatment outcomes?")

    assert len(result.evidence) == 1
    assert result.evidence[0][1] == 0.95


def test_stops_when_no_new_evidence_is_found():
    item = chunk("same", "Treatment outcome evidence for patients.")
    retriever = FakeHybridRetriever({
        "What are the treatment outcomes?": [(item, 0.9)],
        "What are the treatment outcomes? evidence about outcomes": [
            (item, 0.8)
        ],
    })

    result = IterativeRetriever(
        retriever,
        gap_analyzer=lambda query, evidence: ["outcomes"],
    ).retrieve("What are the treatment outcomes?")

    assert result.stop_reason == "no_new_evidence"
    assert result.iteration_count == 2


def test_maximum_iterations_cannot_exceed_three():
    retriever = FakeHybridRetriever({})
    iterative = IterativeRetriever(
        retriever,
        max_iterations=3,
        gap_analyzer=lambda query, evidence: ["treatment"],
    )

    result = iterative.retrieve("Medical treatment evidence")

    assert result.iteration_count == 1
    assert result.stop_reason == "no_new_evidence"


def test_rejects_more_than_three_iterations():
    retriever = FakeHybridRetriever({})

    try:
        IterativeRetriever(retriever, max_iterations=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for max_iterations=4")
