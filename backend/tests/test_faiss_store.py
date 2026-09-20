from pathlib import Path

from app.rag.chunking.models import MedicalChunk
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.vectorstore.faiss_store import FAISSVectorStore


def main():
    print("=" * 80)
    print("Testing BGE-M3 + FAISS Medical Retrieval")
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
    # 3. Generate BGE-M3 embeddings
    # ------------------------------------------------------------------

    embedder = BGEEmbedding()

    texts = [chunk.content for chunk in chunks]

    print("\nGenerating embeddings...")

    embeddings = embedder.embed_documents(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. Build FAISS index
    # ------------------------------------------------------------------

    store = FAISSVectorStore(
        dimension=embeddings.shape[1]
    )

    store.add(
        embeddings=embeddings,
        chunks=chunks,
    )

    print(f"FAISS vectors stored: {store.size}")

    # ------------------------------------------------------------------
    # 5. Semantic search
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    query_embedding = embedder.embed_query(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)

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
    # 6. Test persistence
    # ------------------------------------------------------------------

    vectorstore_dir = Path("data/vectorstore/pmc_test")

    print("\n" + "=" * 80)
    print("Testing persistence")
    print("=" * 80)

    store.save(vectorstore_dir)

    print(f"Saved vector store to: {vectorstore_dir}")

    loaded_store = FAISSVectorStore.load(vectorstore_dir)

    print(f"Reloaded vectors: {loaded_store.size}")

    assert loaded_store.size == store.size
    assert len(loaded_store.chunks) == len(store.chunks)

    print("\nSUCCESS: FAISS retrieval and persistence test completed.")


if __name__ == "__main__":
    main()