from app.rag.embeddings.bge_embedding import BGEEmbedding


def main():
    print("=" * 80)
    print("Testing BGE-M3 Embedding")
    print("=" * 80)

    embedder = BGEEmbedding()

    texts = [
        "Immune checkpoint inhibitors activate anti-tumor immune responses.",
        "FDG-PET may produce false-positive findings because inflammation can increase FDG uptake.",
        "RECIST criteria evaluate tumor response using changes in lesion dimensions.",
    ]

    print("\nGenerating document embeddings...")

    embeddings = embedder.embed_documents(texts)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    print("\nGenerating query embedding...")

    query = (
        "What are the limitations of FDG-PET during "
        "immune checkpoint inhibitor treatment?"
    )

    query_embedding = embedder.embed_query(query)

    print(f"Query embedding shape: {query_embedding.shape}")

    print("\nChecking vector normalization...")

    norm = (query_embedding ** 2).sum() ** 0.5

    print(f"Query vector L2 norm: {norm:.6f}")

    print("\nSUCCESS: BGE-M3 embedding test completed.")


if __name__ == "__main__":
    main()