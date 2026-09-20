from pathlib import Path

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.retrievers.sparse_retriever import SparseRetriever


def main():
    print("=" * 80)
    print("Testing SparseRetriever (BM25)")
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
    # 3. Build BM25 index
    # ------------------------------------------------------------------

    retriever = SparseRetriever()

    print("\nIndexing chunks...")

    retriever.add_chunks(chunks)

    print(f"Indexed chunks: {retriever.size}")
    print(f"Vocabulary size: {retriever.vocabulary_size}")

    # ------------------------------------------------------------------
    # 4. Sparse retrieval
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

    print("\n" + "=" * 80)
    print("SPARSE RETRIEVAL RESULTS")
    print("=" * 80)

    for rank, (chunk, score) in enumerate(
        results,
        start=1,
    ):
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
    # 5. Basic assertions
    # ------------------------------------------------------------------

    assert retriever.size == len(chunks)
    assert retriever.vocabulary_size > 0
    assert len(results) <= 5

    # ------------------------------------------------------------------
    # 6. Test persistence
    # ------------------------------------------------------------------

    sparse_path = Path(
        "data/vectorstore/pmc_test/bm25.pkl"
    )

    print("\n" + "=" * 80)
    print("Testing SparseRetriever persistence")
    print("=" * 80)

    retriever.save(sparse_path)

    print(f"Saved sparse index to: {sparse_path}")

    loaded_retriever = SparseRetriever.load(
        sparse_path
    )

    print(
        f"Reloaded chunks: "
        f"{loaded_retriever.size}"
    )

    assert loaded_retriever.size == retriever.size
    assert (
        loaded_retriever.vocabulary_size
        == retriever.vocabulary_size
    )

    # Verify retrieval still works after loading.
    loaded_results = loaded_retriever.retrieve(
        query=query,
        top_k=5,
    )

    assert len(loaded_results) == len(results)

    print(
        "\nSUCCESS: BM25 sparse retrieval "
        "and persistence test completed."
    )


if __name__ == "__main__":
    main()