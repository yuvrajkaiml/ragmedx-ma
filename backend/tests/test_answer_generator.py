"""

Integration test for the evidence-grounded answer generator.


Pipeline tested:


    PMC documents

        ↓

    Medical chunks

        ↓

    Hybrid retrieval

        ↓

    Cross-encoder reranking

        ↓

    Evidence verification

        ↓

    Citation generation

        ↓

    Evidence preparation

        ↓

    Local Qwen LLM

        ↓

    Evidence-grounded answer

"""


from __future__ import annotations


import logging

import sys

from pathlib import Path


# ---------------------------------------------------------------------------

# Ensure backend root is available on sys.path

# ---------------------------------------------------------------------------


BACKEND_ROOT = Path(__file__).resolve().parents[1]


if str(BACKEND_ROOT) not in sys.path:

    sys.path.insert(0, str(BACKEND_ROOT))


# ---------------------------------------------------------------------------

# Project imports

# ---------------------------------------------------------------------------


from app.rag.loaders.pmc_loader import PMCLoader

from app.rag.chunking.semantic_chunker import MedicalChunker

from app.rag.embeddings.bge_embedding import BGEEmbedding

from app.rag.vectorstore.faiss_store import FAISSVectorStore

from app.rag.retrievers.dense_retriever import DenseRetriever

from app.rag.retrievers.sparse_retriever import SparseRetriever

from app.rag.retrievers.hybrid_retriever import HybridRetriever

from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker

from app.rag.verification.evidence_verifier import EvidenceVerifier

from app.rag.citations.citation_generator import CitationGenerator

from app.rag.generation.answer_generator import AnswerGenerator

from app.llm.local_llm import LocalLLMWrapper


# ---------------------------------------------------------------------------

# Logging

# ---------------------------------------------------------------------------


logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

# Main integration test

# ---------------------------------------------------------------------------


def main() -> None:


    print("=" * 80)

    print("Testing Evidence-Grounded Answer Generator")

    print("=" * 80)


    # -----------------------------------------------------------------------

    # 1. Load medical documents

    # -----------------------------------------------------------------------


    print("\n[1/18] Loading PMC medical documents...")


    data_dir = BACKEND_ROOT / "data" / "datasets" / "pmc"


    loader = PMCLoader(data_dir)

    documents = loader.load()


    print(f"Documents loaded: {len(documents)}")


    assert documents, "No medical documents were loaded."


    # -----------------------------------------------------------------------

    # 2. Chunk documents

    # -----------------------------------------------------------------------


    print("\n[2/18] Creating medical chunks...")


    chunker = MedicalChunker(

        chunk_size=800,

        overlap=100,

        min_chunk_size=150,

    )


    chunks = chunker.chunk_documents(documents)


    print(f"Chunks created: {len(chunks)}")


    assert chunks, "No chunks were created."


    # -----------------------------------------------------------------------

    # 3. Initialize dense retrieval

    # -----------------------------------------------------------------------


    print("\n[3/18] Initializing dense retriever...")


    embedder = BGEEmbedding()


    vector_store = FAISSVectorStore(

        dimension=1024,

    )


    dense_retriever = DenseRetriever(

        embedder=embedder,

        vector_store=vector_store,

    )


    # -----------------------------------------------------------------------

    # 4. Initialize sparse retrieval

    # -----------------------------------------------------------------------


    print("\n[4/18] Initializing sparse retriever...")


    sparse_retriever = SparseRetriever()


    # -----------------------------------------------------------------------

    # 5. Initialize hybrid retrieval

    # -----------------------------------------------------------------------


    print("\n[5/18] Initializing hybrid retriever...")


    hybrid_retriever = HybridRetriever(

        dense_retriever=dense_retriever,

        sparse_retriever=sparse_retriever,

    )


    # -----------------------------------------------------------------------

    # 6. Initialize reranker

    # -----------------------------------------------------------------------


    print("\n[6/18] Initializing cross-encoder reranker...")


    reranker = CrossEncoderReranker(

        device="cpu",

    )


    # -----------------------------------------------------------------------

    # 7. Initialize evidence verifier

    # -----------------------------------------------------------------------


    print("\n[7/18] Initializing evidence verifier...")


    verifier = EvidenceVerifier()


    # -----------------------------------------------------------------------

    # 8. Initialize citation generator

    # -----------------------------------------------------------------------


    print("\n[8/18] Initializing citation generator...")


    citation_generator = CitationGenerator()


    # -----------------------------------------------------------------------

    # 9. Initialize local LLM

    # -----------------------------------------------------------------------


    print("\n[9/18] Initializing local Qwen LLM...")


    llm = LocalLLMWrapper(

        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",

    )


    # -----------------------------------------------------------------------

    # 10. Initialize answer generator

    # -----------------------------------------------------------------------


    print("\n[10/18] Initializing answer generator...")


    answer_generator = AnswerGenerator(

        llm_wrapper=llm,

    )


    # -----------------------------------------------------------------------

    # 11. Index chunks

    # -----------------------------------------------------------------------


    print("\n[11/18] Indexing chunks into hybrid retriever...")


    hybrid_retriever.add_chunks(chunks)


    print(

        f"Hybrid indexed chunks: "

        f"{hybrid_retriever.size}"

    )


    assert hybrid_retriever.size > 0, (

        "Hybrid retriever contains no indexed chunks."

    )


    # -----------------------------------------------------------------------

    # 12. Query

    # -----------------------------------------------------------------------


    query = (

        "What are the limitations of FDG-PET when evaluating "

        "patients treated with immune checkpoint inhibitors?"

    )


    print("\n[12/18] Query:")

    print(query)


    # -----------------------------------------------------------------------

    # 13. Hybrid retrieval

    # -----------------------------------------------------------------------


    print("\n[13/18] Retrieving hybrid candidates...")


    hybrid_results = hybrid_retriever.retrieve(

        query=query,

        top_k=10,

    )


    print(

        f"Hybrid candidates: "

        f"{len(hybrid_results)}"

    )


    assert hybrid_results, (

        "Hybrid retrieval returned no candidates."

    )


    # -----------------------------------------------------------------------

    # 14. Cross-encoder reranking

    # -----------------------------------------------------------------------


    print("\n[14/18] Reranking candidates...")


    reranked_results = reranker.rerank(

        query=query,

        results=hybrid_results,

        top_k=5,

    )


    print(

        f"Reranked candidates: "

        f"{len(reranked_results)}"

    )


    assert reranked_results, (

        "Reranking returned no candidates."

    )


    # -----------------------------------------------------------------------

    # 15. Evidence verification

    # -----------------------------------------------------------------------


    print("\n[15/18] Verifying evidence...")


    # EvidenceVerifier.verify() currently accepts the candidate results

    # directly. It does not accept query= as a keyword argument.

    verified_results = verifier.verify(

        reranked_results,

    )


    print(

        f"Validated evidence: "

        f"{len(verified_results)}"

    )


    assert verified_results, (

        "Evidence verification returned no evidence."

    )


    # -----------------------------------------------------------------------

    # 16. Citation generation

    # -----------------------------------------------------------------------


    print("\n[16/18] Generating citations...")


    citations = citation_generator.generate(

        verified_results,

    )


    print(

        f"Citations generated: "

        f"{len(citations)}"

    )


    assert citations, (

        "Citation generation returned no citations."

    )


        # -----------------------------------------------------------------------
    # 17. Prepare evidence for answer generation
    # -----------------------------------------------------------------------

    print("\n[17/18] Preparing evidence for answer generation...")

    citation_map = citation_generator.citation_map(
        verified_results,
    )

    generation_evidence = []

    for item in verified_results:
        # EvidenceVerifier returns EvidenceVerificationResult objects.
        # Keep only verified evidence and convert it to the dictionary
        # format consumed by AnswerGenerator.
        if not hasattr(item, "chunk") or not hasattr(item, "score"):
            raise TypeError(
                "Expected EvidenceVerificationResult with "
                "'chunk' and 'score' attributes, but received: "
                f"{type(item).__name__}"
            )

        if not item.is_valid:
            print(
                f"Skipping invalid evidence chunk: "
                f"{item.chunk.id} | Reasons: {item.reasons}"
            )
            continue

        chunk = item.chunk
        score = item.score
        citation = citation_map.get(chunk.id)

        if citation is None:
            logger.warning(
                "No citation found for chunk %s. Skipping.",
                chunk.id,
            )
            continue

        generation_evidence.append(
            {
                "chunk": chunk,
                "citation_marker": citation.citation_id,
                "title": citation.title,
                "journal": citation.journal,
                "publication_year": citation.publication_year,
                "cited_text": chunk.content,
                "retrieval_score": score,
            }
        )

    print(
        f"Evidence prepared for generation: "
        f"{len(generation_evidence)} chunks"
    )

    assert generation_evidence, (
        "No evidence was prepared for answer generation."
    )


# -----------------------------------------------------------------------

    # Evidence preview

    # -----------------------------------------------------------------------


    print("\nEvidence preview:")


    for evidence in generation_evidence[:3]:


        citation_marker = evidence["citation_marker"]

        title = evidence["title"]

        journal = evidence["journal"]

        year = evidence["publication_year"]


        print(

            f"  {citation_marker} "

            f"{title} — {journal} ({year})"

        )


    # -----------------------------------------------------------------------

    # 18. Generate evidence-grounded answer

    # -----------------------------------------------------------------------


    print("\n[18/18] Generating evidence-grounded answer...")


    # LocalLLMWrapper automatically selects:

    #

    #   CUDA/NVIDIA GPU when available

    #   CPU when CUDA is unavailable

    #

    # Therefore, answer generation is tested on CPU as well.


    llm.load_model()


    answer = answer_generator.generate(

        query=query,

        evidence=generation_evidence,

    )


    # -----------------------------------------------------------------------

    # Display generated answer

    # -----------------------------------------------------------------------


    print("\n" + "=" * 80)

    print("GENERATED ANSWER")

    print("=" * 80)


    print()

    print(answer)


    # -----------------------------------------------------------------------

    # Validation

    # -----------------------------------------------------------------------


    assert answer is not None, (

        "Answer generator returned None."

    )


    assert isinstance(answer, str), (

        "Answer generator did not return a string."

    )


    assert answer.strip(), (

        "Answer generator returned an empty answer."

    )


    print("\n" + "=" * 80)

    print("SUCCESS: Evidence-grounded answer generation test completed.")

    print("=" * 80)


# ---------------------------------------------------------------------------

# Entry point

# ---------------------------------------------------------------------------


if __name__ == "__main__":

    main()
