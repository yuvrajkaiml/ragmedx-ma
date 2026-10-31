"""Compatibility tests for the pre-merge cross_encoder module path.

Reranker behavior is covered by ``test_cross_encoder_reranker.py``.  This
module verifies that legacy imports resolve to Sana's single implementation.
"""


def test_legacy_cross_encoder_module_reexports_native_reranker():
    from app.rag.rerankers import CrossEncoderReranker as package_reranker
    from app.rag.rerankers.cross_encoder import CrossEncoderReranker as legacy_reranker
    from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker as native_reranker
    from app.retrieval.reranker import CrossEncoderReranker as retrieval_reranker

    assert legacy_reranker is native_reranker
    assert package_reranker is native_reranker
    assert retrieval_reranker is native_reranker
