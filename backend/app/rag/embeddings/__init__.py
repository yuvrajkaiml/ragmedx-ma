"""Embedding-model adapters for the RAG pipeline."""

from app.rag.embeddings.bge_embedding import BGEEmbedding, BGEEmbedder

__all__ = ["BGEEmbedding", "BGEEmbedder"]
