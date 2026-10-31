"""
app.retrieval -- retrieval pipeline boundary package.

Public surface
--------------
``DocumentLoader``         -- unified ingestion entry-point.
``BGEEmbedder``            -- compatibility alias of the BGE-M3 adapter.
``FAISSVectorStore``       -- FAISS inner-product vector store for MedicalChunk.
``DenseRetriever``         -- query -> BGE-M3 embedding -> FAISS.
``SparseRetriever``        -- BM25 keyword retrieval over MedicalChunk.
``HybridRetriever``        -- RRF fusion of dense + sparse.
``CrossEncoderReranker``   -- cross-encoder re-scoring of first-stage results.
``RAGPipeline``            -- end-to-end retrieval provider (hybrid + reranking).
``Embedder``               -- abstract base class for embedders.
``VectorStore``            -- abstract base class for vector stores.
"""

from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.document_loader import DocumentLoader
from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.interface import Embedder, VectorStore
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse_retriever import SparseRetriever
from app.retrieval.vector_store import FAISSVectorStore
from app.rag.pipeline import RAGPipeline

__all__ = [
    "DocumentLoader",
    "BGEEmbedder",
    "FAISSVectorStore",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "RAGPipeline",
    "Embedder",
    "VectorStore",
]
