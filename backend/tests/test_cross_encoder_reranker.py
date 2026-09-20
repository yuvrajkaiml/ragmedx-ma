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
    print("Testing Cross-Encoder Reranker")
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
    # 3. Initialize dense + sparse retrieval
    # ------------------------------------------------------------------

    print("\nInitializing dense retriever...")

    embedder = BGEEmbedding()

    vector_store = FAISSVectorStore(
        dimension=1024
    )

    dense_retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    sparse_retriever = SparseRetriever()

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    # ------------------------------------------------------------------
    # 4. Index chunks
    # ------------------------------------------------------------------

    print("\nIndexing chunks...")

    hybrid_retriever.add_chunks(chunks)

    print(f"Hybrid indexed chunks: {hybrid_retriever.size}")

    # ------------------------------------------------------------------
    # 5. Hybrid retrieval
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    print("\nRetrieving hybrid candidates...")

    hybrid_results = hybrid_retriever.retrieve(
        query=query,
        top_k=10,
    )

    print(f"Hybrid candidates: {len(hybrid_results)}")

    # ------------------------------------------------------------------
    # 6. Initialize reranker
    # ------------------------------------------------------------------

    print("\nInitializing cross-encoder reranker...")

    reranker = CrossEncoderReranker(
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 7. Rerank
    # ------------------------------------------------------------------

    print("\nReranking candidates...")

    reranked_results = reranker.rerank(
        query=query,
        results=hybrid_results,
        top_k=5,
    )

    # ------------------------------------------------------------------
    # 8. Display results
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for rank, (chunk, score) in enumerate(
        reranked_results,
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
    # 9. Basic validation
    # ------------------------------------------------------------------

    assert len(reranked_results) <= 5

    assert all(
        isinstance(chunk.id, str)
        for chunk, _ in reranked_results
    )

    assert all(
        isinstance(score, float)
        for _, score in reranked_results
    )

    scores = [
        score
        for _, score in reranked_results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("SUCCESS: Cross-encoder reranker test completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()