"""
app.rag — canonical home for all RAG pipeline internals.

Package layout
--------------
loaders/
    models.py           — MedicalDocument dataclass (shared data contract)
    medqa_loader.py     — MedQA JSON/JSONL loader  (IMPLEMENTED)
    pdf_loader.py       — PDF page-by-page loader   (IMPLEMENTED)
    text_loader.py      — Plain-text file loader     (IMPLEMENTED)

chunking/
    recursive_chunker.py — Fixed-size recursive chunker  (TODO)
    semantic_chunker.py  — Semantic boundary chunker      (TODO)

embeddings/
    bge_embedding.py    — BGE-small-en-v1.5 adapter (TODO)

retrievers/
    dense_retriever.py  — Vector similarity retrieval     (TODO)
    sparse_retriever.py — BM25 / lexical retrieval        (TODO)
    hybrid_retriever.py — Reciprocal-rank-fusion hybrid   (TODO)

rerankers/
    cross_encoder.py    — Cross-encoder reranking         (TODO)

vectorstore/
    faiss_store.py      — FAISS persistence layer         (TODO)

prompts/
    qa_prompt.py        — Q&A prompt templates            (IMPLEMENTED)
    citation_prompt.py  — Citation prompt templates       (IMPLEMENTED)
    verification_prompt.py — Verification prompts         (IMPLEMENTED)

interfaces.py           — Embedder & VectorStore ABCs     (IMPLEMENTED)
pipeline.py             — End-to-end pipeline composition (TODO)

Boundary with app.retrieval
---------------------------
``app.retrieval`` is the *public* retrieval API used by the FastAPI routes and
the multi-agent orchestrator.  It re-exports the stable contracts from here:

    app.retrieval.DocumentLoader  →  app.rag.loaders  (via DocumentLoader)
    app.retrieval.Embedder        →  app.rag.interfaces.Embedder
    app.retrieval.VectorStore     →  app.rag.interfaces.VectorStore

The stub files in ``app.retrieval`` (embeddings.py, vector_store.py,
hybrid_search.py, reranker.py) will be removed once the corresponding
``app.rag`` implementations land.
"""

from app.rag.interfaces import Embedder, VectorStore

__all__ = ["Embedder", "VectorStore"]
