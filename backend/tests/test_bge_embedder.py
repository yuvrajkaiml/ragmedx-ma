"""Compatibility tests for the retired BGEEmbedder import name.

The pre-merge tests asserted the deleted bge-small 384-dimensional contract.
The supported embedding contract is now Sana's BGE-M3 implementation.
"""

import numpy as np

from app.rag.embeddings import BGEEmbedding, BGEEmbedder


def test_bge_embedder_is_the_native_bge_m3_adapter():
    assert BGEEmbedder is BGEEmbedding
    assert BGEEmbedder.DEFAULT_MODEL == "BAAI/bge-m3"


def test_bge_embedder_import_boundaries_resolve_to_native_adapter():
    from app.retrieval import BGEEmbedder as retrieval_embedder
    from app.retrieval.embeddings import BGEEmbedder as boundary_embedder

    assert retrieval_embedder is BGEEmbedding
    assert boundary_embedder is BGEEmbedding


def test_native_embedder_contract_uses_normalized_numpy_vectors():
    embedder = BGEEmbedding.__new__(BGEEmbedding)
    embedder.normalize = True
    embedder.device = "cpu"

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            import torch

            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class Model:
        def __call__(self, **_kwargs):
            import torch

            return type(
                "Output",
                (),
                {"last_hidden_state": torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])},
            )()

    embedder.tokenizer = Tokenizer()
    embedder.model = Model()

    result = embedder.embed_query("medical query")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert np.isclose(np.linalg.norm(result), 1.0)
