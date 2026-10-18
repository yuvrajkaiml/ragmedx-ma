"""Assert the reranker candidate-pool default (``top_k_retrieval``) is 40 and
the final result count (``top_k_final``) remains 5, in the two chained build
entry points that configure production retrieval.
"""

import inspect

from app.rag.pipeline import RAGPipeline
from app.services.retrieval_service import RetrievalService


def test_rag_pipeline_build_defaults():
    params = inspect.signature(RAGPipeline.build).parameters
    assert params["top_k_retrieval"].default == 40
    assert params["top_k_final"].default == 5


def test_retrieval_service_build_defaults():
    params = inspect.signature(RetrievalService.build).parameters
    assert params["top_k_retrieval"].default == 40
    assert params["top_k_final"].default == 5
