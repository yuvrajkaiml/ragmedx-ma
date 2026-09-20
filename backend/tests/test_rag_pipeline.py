from pathlib import Path

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.pipeline import RAGRetrievalPipeline


def main():
    print("=" * 80)
    print("Testing Complete RAG Retrieval Pipeline")
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

    print("\nInitializing sparse retriever...")

    sparse_retriever = SparseRetriever()

    # ------------------------------------------------------------------
    # 5. Initialize hybrid retriever
    # ------------------------------------------------------------------

    print("\nInitializing hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    # ------------------------------------------------------------------
    # 6. Initialize reranker
    # ------------------------------------------------------------------

    print("\nInitializing cross-encoder reranker...")

    reranker = CrossEncoderReranker(
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 7. Initialize complete pipeline
    # ------------------------------------------------------------------

    print("\nInitializing RAG retrieval pipeline...")

    pipeline = RAGRetrievalPipeline(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        retrieval_top_k=10,
        final_top_k=5,
    )

    # ------------------------------------------------------------------
    # 8. Index chunks
    # ------------------------------------------------------------------

    print("\nIndexing chunks...")

    pipeline.add_chunks(chunks)

    print(
        f"Pipeline indexed chunks: {pipeline.size}"
    )

    assert pipeline.size == len(chunks)

    # ------------------------------------------------------------------
    # 9. Query
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    print("\nExecuting complete retrieval pipeline:")

    print("  Query Expansion")
    print("  Dense Retrieval")
    print("  Sparse Retrieval")
    print("  RRF Fusion")
    print("  Cross-Encoder Reranking")

    # ------------------------------------------------------------------
    # 10. Retrieve final evidence
    # ------------------------------------------------------------------

    results = pipeline.retrieve(
        query=query,
    )

    print(f"\nFinal results: {len(results)}")

    # ------------------------------------------------------------------
    # 11. Display results
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL EVIDENCE RESULTS")
    print("=" * 80)

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"Result {rank}")
        print("-" * 80)

        print(f"Reranker Score: {score:.6f}")
        print(f"Chunk ID: {chunk.id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Title: {chunk.title}")
        print(f"Section: {chunk.section}")
        print(f"Journal: {chunk.journal}")
        print(f"Publication Year: {chunk.publication_year}")

        print("\nContent:")
        print(chunk.content[:500])

    # ------------------------------------------------------------------
    # 12. Validation
    # ------------------------------------------------------------------

    assert len(results) <= 5

    assert all(
        isinstance(chunk.id, str)
        for chunk, _ in results
    )

    assert all(
        isinstance(score, float)
        for _, score in results
    )

    scores = [
        score
        for _, score in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # Verify that final results contain actual medical content.
    assert all(
        chunk.content.strip()
        for chunk, _ in results
    )

    print("\n" + "=" * 80)
    print(
        "SUCCESS: Complete RAG retrieval pipeline "
        "test completed."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()