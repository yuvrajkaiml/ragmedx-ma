from pathlib import Path

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.vectorstore.faiss_store import FAISSVectorStore

from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker


def main():
    print("=" * 80)
    print("Testing HybridRetriever (Dense + BM25 + Query Expansion + RRF + Reranking)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load documents
    # ------------------------------------------------------------------

    data_dir = Path("data/datasets/pmc")

    loader = PMCLoader(data_dir)
    documents = loader.load()

    print(f"\nDocuments loaded: {len(documents)}")

    # ------------------------------------------------------------------
    # 2. Chunk documents
    # ------------------------------------------------------------------

    chunker = MedicalChunker(
        chunk_size=800,
        overlap=100,
        min_chunk_size=150,
    )

    chunks = chunker.chunk_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # ------------------------------------------------------------------
    # 3. Initialize dense retriever
    # ------------------------------------------------------------------

    print("\nInitializing dense retriever...")

    embedder = BGEEmbedding()

    vector_store = FAISSVectorStore(
        dimension=1024,
    )

    dense_retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # ------------------------------------------------------------------
    # 4. Initialize sparse retriever
    # ------------------------------------------------------------------

    sparse_retriever = SparseRetriever()

    # ------------------------------------------------------------------
    # 5. Initialize cross-encoder reranker
    # ------------------------------------------------------------------

    print("\nInitializing cross-encoder reranker...")

    reranker = CrossEncoderReranker(
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 6. Initialize integrated hybrid retriever
    # ------------------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
    )

    # ------------------------------------------------------------------
    # 7. Index chunks
    # ------------------------------------------------------------------

    print("\nIndexing chunks into dense + sparse retrievers...")

    hybrid_retriever.add_chunks(chunks)

    print(
        f"Hybrid indexed chunks: "
        f"{hybrid_retriever.size}"
    )

    print(
        f"Dense indexed chunks: "
        f"{dense_retriever.size}"
    )

    print(
        f"Sparse indexed chunks: "
        f"{sparse_retriever.size}"
    )

    # ------------------------------------------------------------------
    # 8. Validate indexing
    # ------------------------------------------------------------------

    assert hybrid_retriever.size == len(chunks)

    assert dense_retriever.size == len(chunks)

    assert sparse_retriever.size == len(chunks)

    # ------------------------------------------------------------------
    # 9. Define query
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    # ------------------------------------------------------------------
    # 10. Perform integrated retrieval
    # ------------------------------------------------------------------

    print(
        "\nRetrieving candidates using:"
        "\n  Query Expansion"
        "\n  Dense Retrieval"
        "\n  Sparse Retrieval"
        "\n  RRF Fusion"
        "\n  Cross-Encoder Reranking"
    )

    final_results = hybrid_retriever.retrieve(
        query=query,
        top_k=5,
        dense_top_k=10,
        sparse_top_k=10,
        rerank_top_k=10,
    )

    print(
        f"\nFinal results: "
        f"{len(final_results)}"
    )

    # ------------------------------------------------------------------
    # 11. Display final reranked results
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL RERANKED RESULTS")
    print("=" * 80)

    for rank, (chunk, score) in enumerate(
        final_results,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"Result {rank}")
        print("-" * 80)

        print(
            f"Cross-Encoder Score: "
            f"{score:.6f}"
        )

        print(f"Chunk ID: {chunk.id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Title: {chunk.title}")
        print(f"Section: {chunk.section}")
        print(f"Journal: {chunk.journal}")
        print(
            f"Publication Year: "
            f"{chunk.publication_year}"
        )

        print("\nContent:")
        print(chunk.content[:500])

    # ------------------------------------------------------------------
    # 12. Basic validation
    # ------------------------------------------------------------------

    assert len(final_results) <= 5

    assert len(final_results) > 0

    assert all(
        isinstance(chunk.id, str)
        for chunk, _ in final_results
    )

    assert all(
        isinstance(score, float)
        for _, score in final_results
    )

    scores = [
        score
        for _, score in final_results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # Verify that the reranker is actually attached.
    assert hybrid_retriever.reranker is reranker

    print("\n" + "=" * 80)
    print(
        "SUCCESS: Integrated hybrid retrieval + "
        "cross-encoder reranking test completed."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()