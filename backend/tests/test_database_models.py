
"""Tests for SQLAlchemy medical knowledge-base models."""

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Base, Chunk, Document, RetrievedEvidence


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    with Session(engine) as db:
        yield db

    engine.dispose()


def test_document_and_chunk_relationship(session):
    document = Document(
        id="pubmed-123",
        title="Example Medical Study",
        source="PubMed",
        journal="Example Journal",
        publication_year=2024,
        metadata_json={"pmid": "123"},
    )
    chunk = Chunk(
        id="pubmed-123-chunk-0",
        document_id="pubmed-123",
        content="The study evaluated a treatment.",
        section="Results",
        chunk_index=0,
        metadata_json={"token_count": 8},
    )

    document.chunks.append(chunk)
    session.add(document)
    session.commit()

    loaded = session.get(Document, "pubmed-123")

    assert loaded is not None
    assert loaded.title == "Example Medical Study"
    assert loaded.chunks[0].content == "The study evaluated a treatment."
    assert loaded.chunks[0].document.source == "PubMed"
    assert loaded.metadata_json["pmid"] == "123"


def test_retrieved_evidence_is_saved(session):
    document = Document(
        id="doc-1",
        title="Medical Study",
        source="PMC",
    )
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        content="Evidence about treatment outcomes.",
    )
    evidence = RetrievedEvidence(
        query_id="query-1",
        query_text="What are the treatment outcomes?",
        chunk=chunk,
        relevance_score=0.92,
        quality_score=0.85,
        iteration=2,
        rank=1,
    )

    session.add_all([document, evidence])
    session.commit()

    saved = session.scalar(
        select(RetrievedEvidence).where(
            RetrievedEvidence.query_id == "query-1"
        )
    )

    assert saved is not None
    assert saved.chunk_id == "chunk-1"
    assert saved.relevance_score == pytest.approx(0.92)
    assert saved.quality_score == pytest.approx(0.85)
    assert saved.iteration == 2
    assert saved.rank == 1
    assert saved.retrieved_at is not None


def test_document_deletion_cascades_to_chunks(session):
    document = Document(id="doc-2", title="Study", source="PubMed")
    document.chunks.append(
        Chunk(id="chunk-2", content="Study evidence", document_id="doc-2")
    )
    session.add(document)
    session.commit()

    session.delete(document)
    session.commit()

    assert session.get(Chunk, "chunk-2") is None


def test_chunk_requires_existing_document(session):
    session.add(
        Chunk(
            id="orphan-chunk",
            document_id="missing-document",
            content="This should not persist.",
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()
