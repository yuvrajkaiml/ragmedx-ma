
"""Tests for BGE-M3 and PubMedBERT embedding adapters."""

import numpy as np
import pytest

from app.rag.embeddings.bge_embedding import (
    BGEEmbedding,
    PubMedBERTEmbedding,
)


@pytest.mark.parametrize(
    ("embedding_class", "expected_dimension"),
    [
        (BGEEmbedding, 1024),
        (PubMedBERTEmbedding, 768),
    ],
)
def test_embedding_model_output_shape(
    embedding_class,
    expected_dimension,
):
    model = embedding_class(device="cpu", batch_size=2)

    vectors = model.embed(
        [
            "Medical research studies patient outcomes.",
            "Clinical trials evaluate treatment effectiveness.",
        ]
    )

    assert vectors.shape == (2, expected_dimension)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()

    if model.normalize:
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_empty_input_has_model_dimension():
    # Avoid loading a model just to verify empty-input behavior.
    model = BGEEmbedding.__new__(BGEEmbedding)
    model.embedding_dimension = 1024
    model.batch_size = 2

    vectors = model.embed([])

    assert vectors.shape == (0, 1024)
    assert vectors.dtype == np.float32


def test_query_embedding_is_one_dimensional():
    model = BGEEmbedding(device="cpu", batch_size=2)

    vector = model.embed_query(
        "What are the symptoms of iron deficiency?"
    )

    assert vector.shape == (1024,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
