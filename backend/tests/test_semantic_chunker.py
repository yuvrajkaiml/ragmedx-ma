from types import SimpleNamespace

from app.rag.chunking.semantic_chunker import MedicalChunker


def make_document(content, title="Test medical article"):
    return SimpleNamespace(
        id="test_doc",
        content=content,
        title=title,
        source="test",
        journal="Test Journal",
        publication_year=2025,
        metadata={},
    )


def test_chunk_size_and_overlap():
    text = (
        "This is a medical research sentence about patient outcomes. "
        "It contains useful clinical information for retrieval. "
    ) * 100

    chunker = MedicalChunker(
        chunk_size=2000,
        chunk_overlap=400,
        min_chunk_size=1,
    )

    chunks = chunker.chunk_document(make_document(text))

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 2000 for chunk in chunks)

    # Confirm that adjacent chunks retain some shared context.
    for previous, current in zip(chunks, chunks[1:]):
        assert (
            previous.content[-100:] in current.content
            or current.content[:100] in previous.content
        ), "Adjacent chunks do not appear to preserve overlap."


def test_explicit_section_heading():
    text = (
        "Introduction\n\n"
        + ("This paragraph describes the clinical background. " * 8)
        + "\n\nMethods\n\n"
        + ("This paragraph describes the research methods. " * 8)
    )

    chunker = MedicalChunker(
        chunk_size=2000,
        chunk_overlap=400,
        min_chunk_size=1,
    )

    chunks = chunker.chunk_document(make_document(text))

    sections = {chunk.section for chunk in chunks}
    assert "Introduction" in sections
    assert "Methods" in sections


def test_long_paragraph_respects_limit():
    text = "clinical " * 1000

    chunker = MedicalChunker(
        chunk_size=2000,
        chunk_overlap=400,
        min_chunk_size=1,
    )

    chunks = chunker.chunk_document(make_document(text))

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 2000 for chunk in chunks)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
