from pathlib import Path

from app.rag.chunking.semantic_chunker import MedicalChunker
from app.rag.loaders.pmc_loader import PMCLoader
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.verification.evidence_verifier import EvidenceVerifier


def main():
    print("=" * 80)
    print("Testing Evidence Verification Layer")
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
    # 3. Initialize retrieval components
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

    print("\nInitializing sparse retriever...")

    sparse_retriever = SparseRetriever()

    print("\nInitializing hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    print("\nInitializing cross-encoder reranker...")

    reranker = CrossEncoderReranker(
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 4. Index chunks
    # ------------------------------------------------------------------

    print("\nIndexing chunks...")

    hybrid_retriever.add_chunks(chunks)

    print(
        f"Hybrid indexed chunks: "
        f"{hybrid_retriever.size}"
    )

    # ------------------------------------------------------------------
    # 5. Query
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    # ------------------------------------------------------------------
    # 6. Hybrid retrieval
    # ------------------------------------------------------------------

    print("\nRetrieving hybrid candidates...")

    hybrid_results = hybrid_retriever.retrieve(
        query=query,
        top_k=10,
    )

    print(
        f"Hybrid candidates retrieved: "
        f"{len(hybrid_results)}"
    )

    # ------------------------------------------------------------------
    # 7. Cross-encoder reranking
    # ------------------------------------------------------------------

    print("\nReranking candidates...")

    reranked_results = reranker.rerank(
        query=query,
        results=hybrid_results,
        top_k=5,
    )

    print(
        f"Reranked candidates: "
        f"{len(reranked_results)}"
    )

    # ------------------------------------------------------------------
    # 8. Evidence verification
    # ------------------------------------------------------------------

    print("\nInitializing evidence verifier...")

    verifier = EvidenceVerifier(
        min_content_length=100,
    )

    print("\nVerifying evidence...")

    valid_results, verification_report = (
        verifier.verify_and_filter(
            reranked_results
        )
    )

    # ------------------------------------------------------------------
    # 9. Verification report
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EVIDENCE VERIFICATION REPORT")
    print("=" * 80)

    for rank, result in enumerate(
        verification_report,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"Candidate {rank}")
        print("-" * 80)

        print(
            f"Chunk ID: {result.chunk.id}"
        )

        print(
            f"Score: {result.score:.6f}"
        )

        print(
            f"Valid: {result.is_valid}"
        )

        if result.reasons:
            print("Reasons:")

            for reason in result.reasons:
                print(f"  - {reason}")

        else:
            print("Reasons: None")

    # ------------------------------------------------------------------
    # 10. Validated evidence
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATED EVIDENCE")
    print("=" * 80)

    print(
        f"\nValid evidence chunks: "
        f"{len(valid_results)}"
    )

    for rank, (chunk, score) in enumerate(
        valid_results,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"Evidence {rank}")
        print("-" * 80)

        print(
            f"Score: {score:.6f}"
        )

        print(
            f"Chunk ID: {chunk.id}"
        )

        print(
            f"Document ID: {chunk.document_id}"
        )

        print(
            f"Title: {chunk.title}"
        )

        print(
            f"Section: {chunk.section}"
        )

        print(
            f"Journal: {chunk.journal}"
        )

        print(
            f"Publication Year: "
            f"{chunk.publication_year}"
        )

        print("\nContent:")
        print(chunk.content[:500])

    # ------------------------------------------------------------------
    # 11. Validation
    # ------------------------------------------------------------------

    assert len(verification_report) == len(
        reranked_results
    )

    assert len(valid_results) <= len(
        reranked_results
    )

    assert all(
        isinstance(result.is_valid, bool)
        for result in verification_report
    )

    assert all(
        isinstance(result.score, float)
        for result in verification_report
    )

    assert all(
        result.is_valid
        for result in (
            EvidenceVerifier().verify(
                valid_results
            )
        )
    )

    valid_scores = [
        score
        for _, score in valid_results
    ]

    assert valid_scores == sorted(
        valid_scores,
        reverse=True,
    )

    print("\n" + "=" * 80)
    print(
        "SUCCESS: Evidence verification test completed."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()