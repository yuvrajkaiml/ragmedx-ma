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
from app.rag.citations.citation_generator import (
    Citation,
    CitationGenerator,
)


def main():
    print("=" * 80)
    print("Testing Citation / Evidence Attribution Layer")
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
    # 4. Initialize evidence verifier
    # ------------------------------------------------------------------

    print("\nInitializing evidence verifier...")

    verifier = EvidenceVerifier(
        min_content_length=100,
    )

    # ------------------------------------------------------------------
    # 5. Initialize citation generator
    # ------------------------------------------------------------------

    print("\nInitializing citation generator...")

    citation_generator = CitationGenerator()

    # ------------------------------------------------------------------
    # 6. Index chunks
    # ------------------------------------------------------------------

    print("\nIndexing chunks...")

    hybrid_retriever.add_chunks(chunks)

    print(
        f"Hybrid indexed chunks: "
        f"{hybrid_retriever.size}"
    )

    # ------------------------------------------------------------------
    # 7. Query
    # ------------------------------------------------------------------

    query = (
        "What are the limitations of FDG-PET when evaluating "
        "patients treated with immune checkpoint inhibitors?"
    )

    print("\nQuery:")
    print(query)

    # ------------------------------------------------------------------
    # 8. Hybrid retrieval
    # ------------------------------------------------------------------

    print("\nRetrieving hybrid candidates...")

    hybrid_results = hybrid_retriever.retrieve(
        query=query,
        top_k=10,
    )

    print(
        f"Hybrid candidates: "
        f"{len(hybrid_results)}"
    )

    # ------------------------------------------------------------------
    # 9. Cross-encoder reranking
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
    # 10. Evidence verification
    # ------------------------------------------------------------------

    print("\nVerifying evidence...")

    valid_evidence = verifier.filter_valid(
        reranked_results
    )

    print(
        f"Validated evidence: "
        f"{len(valid_evidence)}"
    )

    # ------------------------------------------------------------------
    # 11. Generate citations
    # ------------------------------------------------------------------

    print("\nGenerating citations...")

    citations = citation_generator.generate(
        valid_evidence
    )

    print(
        f"Citations generated: "
        f"{len(citations)}"
    )

    # ------------------------------------------------------------------
    # 12. Display citations
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("GENERATED CITATIONS")
    print("=" * 80)

    for citation in citations:

        print("\n" + "-" * 80)
        print(
            f"Citation {citation.citation_id}"
        )
        print("-" * 80)

        print(
            f"Chunk ID: "
            f"{citation.chunk_id}"
        )

        print(
            f"Document ID: "
            f"{citation.document_id}"
        )

        print(
            f"Title: "
            f"{citation.title}"
        )

        print(
            f"Journal: "
            f"{citation.journal}"
        )

        print(
            f"Publication Year: "
            f"{citation.publication_year}"
        )

        print(
            f"Section: "
            f"{citation.section}"
        )

        print(
            f"Source Type: "
            f"{citation.source_type}"
        )

        print(
            "\nFormatted:"
        )

        print(
            citation_generator.format_citation(
                citation
            )
        )

    # ------------------------------------------------------------------
    # 13. Citation map
    # ------------------------------------------------------------------

    citation_map = citation_generator.citation_map(
        valid_evidence
    )

    print("\n" + "=" * 80)
    print("CITATION MAP")
    print("=" * 80)

    for chunk_id, citation in citation_map.items():
        print(
            f"{chunk_id} -> "
            f"{citation.citation_id}"
        )

    # ------------------------------------------------------------------
    # 14. Validation
    # ------------------------------------------------------------------

    assert len(citations) == len(
        valid_evidence
    )

    assert all(
        isinstance(citation, Citation)
        for citation in citations
    )

    assert all(
        citation.chunk_id
        for citation in citations
    )

    assert all(
        citation.document_id
        for citation in citations
    )

    assert all(
        citation.title
        for citation in citations
    )

    assert all(
        citation.journal
        for citation in citations
    )

    assert all(
        citation.citation_id
        for citation in citations
    )

    assert len(citation_map) == len(
        citations
    )

    # Citation numbering should be sequential.

    expected_ids = [
        f"[{index}]"
        for index in range(
            1,
            len(citations) + 1,
        )
    ]

    actual_ids = [
        citation.citation_id
        for citation in citations
    ]

    assert actual_ids == expected_ids

    # ------------------------------------------------------------------
    # 15. Human-readable formatting validation
    # ------------------------------------------------------------------

    formatted = (
        citation_generator.format_citations(
            citations
        )
    )

    assert len(formatted) == len(
        citations
    )

    assert all(
        isinstance(item, str)
        for item in formatted
    )

    print("\n" + "=" * 80)
    print(
        "SUCCESS: Citation generation test completed."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()