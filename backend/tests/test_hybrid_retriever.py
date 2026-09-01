from pathlib import Path

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore


def main():
    print("=" * 80)
    print("Testing HybridRetriever (Dense + BM25 + RRF)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load PMC documents
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
    # 3. Create dense retriever
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

    # ------------------------------------------------------------------
    # 4. Create sparse retriever
    # ------------------------------------------------------------------

    sparse_retriever = SparseRetriever()

    # ------------------------------------------------------------------
    # 5. Create hybrid retriever
    # ------------------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rrf_k=60,
    )

    # ------------------------------------------------------------------
    # 6. Index chunks
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
    # 7. Hybrid retrieval
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    results = hybrid_retriever.retrieve(
        query=query,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult {rank}")
        print("-" * 80)
        print(f"RRF Score: {score:.6f}")
        print(f"Chunk ID: {chunk.id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Title: {chunk.title}")
        print(f"Section: {chunk.section}")
        print(f"Journal: {chunk.journal}")
        print(f"Publication Year: {chunk.publication_year}")

        print("\nContent:")
        print(chunk.content[:500])

    # ------------------------------------------------------------------
    # 8. Assertions
    # ------------------------------------------------------------------

    assert hybrid_retriever.size == len(chunks)

    assert dense_retriever.size == len(chunks)

    assert sparse_retriever.size == len(chunks)

    assert len(results) <= 5

    if results:
        scores = [
            score
            for _, score in results
        ]

        assert all(
            scores[i] >= scores[i + 1]
            for i in range(len(scores) - 1)
        )

        assert all(
            score > 0
            for score in scores
        )

    print("\n" + "=" * 80)
    print("SUCCESS: Hybrid retrieval test completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()