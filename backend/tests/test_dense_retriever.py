from pathlib import Path

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore


def main():
    print("=" * 80)
    print("Testing DenseRetriever")
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
    # 3. Initialize BGE-M3
    # ------------------------------------------------------------------

    embedder = BGEEmbedding()

    # ------------------------------------------------------------------
    # 4. Create FAISS store
    # ------------------------------------------------------------------

    store = FAISSVectorStore(
        dimension=1024
    )

    # ------------------------------------------------------------------
    # 5. Create dense retriever
    # ------------------------------------------------------------------

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=store,
    )

    # ------------------------------------------------------------------
    # 6. Index chunks
    # ------------------------------------------------------------------

    print("\nIndexing chunks...")

    retriever.add_chunks(chunks)

    print(f"Indexed chunks: {retriever.size}")

    assert retriever.size == len(chunks)

    # ------------------------------------------------------------------
    # 7. Retrieve
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    # ------------------------------------------------------------------
    # 8. Display results
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("DENSE RETRIEVAL RESULTS")
    print("=" * 80)

    assert len(results) <= 5

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print("-" * 80)

        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {chunk.id}")
        print(f"Document ID: {chunk.document_id}")
        print(f"Title: {chunk.title}")
        print(f"Section: {chunk.section}")
        print(f"Journal: {chunk.journal}")
        print(f"Publication Year: {chunk.publication_year}")

        print("\nContent:")
        print(chunk.content[:500])

    # ------------------------------------------------------------------
    # 9. Test persistence
    # ------------------------------------------------------------------

    vectorstore_dir = Path(
        "data/vectorstore/dense_test"
    )

    print("\n" + "=" * 80)
    print("Testing DenseRetriever persistence")
    print("=" * 80)

    retriever.save(vectorstore_dir)

    loaded_retriever = DenseRetriever.load(
        directory=vectorstore_dir,
        embedder=embedder,
    )

    print(
        f"Reloaded chunks: "
        f"{loaded_retriever.size}"
    )

    assert loaded_retriever.size == retriever.size

    # Test retrieval after reload.
    loaded_results = loaded_retriever.retrieve(
        query=query,
        top_k=5,
    )

    assert len(loaded_results) == len(results)

    print(
        "\nSUCCESS: DenseRetriever test completed."
    )


if __name__ == "__main__":
    main()