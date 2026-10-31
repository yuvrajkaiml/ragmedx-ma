
import pickle

import faiss
import numpy as np
import pytest

from app.rag.chunking.models import MedicalChunk
from app.rag.vectorstore.faiss_store import FAISSVectorStore


def make_chunk(chunk_id, content=None):
    return MedicalChunk(
        id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content or f"Medical evidence for {chunk_id}",
        title="Test medical paper",
        section="Results",
        journal="Test Journal",
        publication_year=2024,
    )


@pytest.fixture
def sample_data():
    chunks = [
        make_chunk("chunk-1", "Evidence about diabetes"),
        make_chunk("chunk-2", "Evidence about heart disease"),
        make_chunk("chunk-3", "Evidence about cancer"),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return embeddings, chunks


def test_add_and_search(sample_data):
    embeddings, chunks = sample_data
    store = FAISSVectorStore(dimension=3)

    store.add(embeddings, chunks)
    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=2,
    )

    assert store.size == 3
    assert len(results) == 2
    assert results[0][0].content == chunks[0].content
    assert results[0][1] == pytest.approx(1.0)


def test_rejects_wrong_embedding_dimension(sample_data):
    _, chunks = sample_data
    store = FAISSVectorStore(dimension=3)

    with pytest.raises((ValueError, TypeError)):
        store.add(np.ones((3, 4), dtype=np.float32), chunks)


def test_rejects_mismatched_chunk_count(sample_data):
    embeddings, chunks = sample_data
    store = FAISSVectorStore(dimension=3)

    with pytest.raises((ValueError, TypeError)):
        store.add(embeddings, chunks[:2])


def test_save_and_load(tmp_path, sample_data):
    embeddings, chunks = sample_data
    store = FAISSVectorStore(dimension=3)
    store.add(embeddings, chunks)
    store.save(tmp_path)

    loaded = FAISSVectorStore.load(tmp_path)
    assert loaded.size == store.size

    results = loaded.search(
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        top_k=1,
    )
    assert len(results) == 1
    assert results[0][0].content == chunks[1].content
    assert results[0][1] == pytest.approx(1.0)


def test_incremental_addition(sample_data):
    embeddings, chunks = sample_data
    store = FAISSVectorStore(dimension=3)

    store.add(embeddings[:1], chunks[:1])
    assert store.size == 1

    store.add(embeddings[1:], chunks[1:])
    assert store.size == 3


def test_search_empty_store():
    store = FAISSVectorStore(dimension=3)

    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=5,
    )
    assert results == []


def test_ivf_index_training_and_search(sample_data):
    embeddings, chunks = sample_data
    store = FAISSVectorStore(
        dimension=3,
        index_type="ivf",
        nlist=1,
    )

    store.train(embeddings)
    store.add(embeddings, chunks)

    results = store.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert store.size == 3
    assert len(results) == 1
    assert results[0][0].content == chunks[0].content


def test_duplicate_chunks_are_not_added_twice(sample_data):
    embeddings, chunks = sample_data
    store = FAISSVectorStore(dimension=3)

    store.add(embeddings, chunks)
    store.add(embeddings, chunks)

    assert store.size == 3
    assert len(store.chunks) == 3


def test_add_after_loading_legacy_flat_index(tmp_path, sample_data):
    embeddings, chunks = sample_data

    # Simulate the previous persistence format.
    legacy_index = faiss.IndexFlatIP(3)
    legacy_index.add(embeddings)
    faiss.write_index(legacy_index, str(tmp_path / "index.faiss"))

    with (tmp_path / "chunks.pkl").open("wb") as file:
        pickle.dump(chunks, file)

    loaded = FAISSVectorStore.load(tmp_path)

    results = loaded.search(embeddings[0], top_k=1)
    assert results[0][0].content == chunks[0].content

    new_chunk = make_chunk(
        "chunk-4",
        "Evidence about kidney disease",
    )
    loaded.add(
        np.array([[0.5, 0.5, 0.0]], dtype=np.float32),
        [new_chunk],
    )

    assert loaded.size == 4
