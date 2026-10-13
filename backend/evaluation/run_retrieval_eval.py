# Owner: Pranav Sukale
"""Real retrieval evaluation over BioASQ with gold relevance labels.

BioASQ provides, for each question, a set of gold evidence snippets. We build
a corpus from a sample of questions' snippets (each chunk tagged with its
originating ``question_id``), then for each question measure how well the
hybrid retriever surfaces that question's own gold snippets.

This yields genuinely paper-reportable retrieval metrics — Recall@K,
Precision@K, nDCG@K, MRR — computed against real ground truth, fully on CPU
and independent of the LLM (so the mock LLM does not affect these numbers).

Usage::

    cd backend
    python -m evaluation.run_retrieval_eval --num-questions 100 --top-k 10

Output: a JSON report written to data/results/ and printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from app.config import DATASETS_DIR, RESULTS_DIR
from app.rag.chunking.models import MedicalChunk
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.utils.logger import logger
from evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_mrr,
    compute_latency_metrics,
)


def _load_bioasq_questions(num_questions: int) -> List[Dict[str, Any]]:
    """Load the first N BioASQ questions that have at least 2 gold snippets."""
    path = DATASETS_DIR / "bioasq" / "training13b.json"
    if not path.exists():
        raise FileNotFoundError(f"BioASQ dataset not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    selected = []
    for q in data.get("questions", []):
        snippets = [s for s in q.get("snippets", []) if s.get("text", "").strip()]
        if len(snippets) >= 2:  # need gold evidence to measure recall
            selected.append({"id": q["id"], "body": q["body"], "snippets": snippets})
        if len(selected) >= num_questions:
            break
    return selected


def _build_corpus(questions: List[Dict[str, Any]]) -> List[MedicalChunk]:
    """One chunk per snippet, tagged with its originating question_id."""
    chunks: List[MedicalChunk] = []
    for q in questions:
        for s_idx, snippet in enumerate(q["snippets"]):
            text = snippet["text"].strip()
            chunk_id = f"bioasq_{q['id']}_s{s_idx}"
            chunks.append(MedicalChunk(
                id=chunk_id,
                document_id=chunk_id,
                source="bioasq",
                title=f"snippet {s_idx}",
                content=text,
                metadata={"question_id": q["id"]},
            ))
    return chunks


def run_evaluation(num_questions: int = 100, top_k: int = 10) -> Dict[str, Any]:
    """Run the retrieval evaluation and return a metrics report."""
    logger.info(f"Loading {num_questions} BioASQ questions with gold snippets…")
    questions = _load_bioasq_questions(num_questions)
    logger.info(f"Loaded {len(questions)} questions.")

    chunks = _build_corpus(questions)
    logger.info(f"Built corpus of {len(chunks)} snippet chunks.")

    # Build the full hybrid + reranker pipeline (same as production).
    logger.info("Loading BGE-M3 embedder and building hybrid retriever…")
    embedder = BGEEmbedding(model_name=BGEEmbedding.DEFAULT_MODEL)
    texts = [c.content for c in chunks]
    embeddings = embedder.embed_documents(texts, batch_size=32)

    store = FAISSVectorStore(dimension=embeddings.shape[1])
    store.add(embeddings, chunks)

    dense = DenseRetriever(embedder=embedder, vector_store=store)
    sparse = SparseRetriever()
    sparse.add_chunks(chunks)
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    reranker = CrossEncoderReranker(device="cpu")

    logger.info("Running retrieval over all questions…")

    # Accumulators for batch metrics
    retrieved_id_lists: List[List[str]] = []
    relevant_id_sets: List[Set[str]] = []
    per_k_precision: Dict[int, List[float]] = {1: [], 3: [], 5: [], 10: []}
    per_k_recall: Dict[int, List[float]] = {1: [], 3: [], 5: [], 10: []}
    per_k_ndcg: Dict[int, List[float]] = {1: [], 3: [], 5: [], 10: []}
    latencies_ms: List[float] = []

    for q in questions:
        gold_ids = {f"bioasq_{q['id']}_s{i}" for i in range(len(q["snippets"]))}

        t0 = time.perf_counter()
        # Hybrid retrieve a candidate pool, then cross-encoder rerank.
        hybrid_results = hybrid.retrieve(query=q["body"], top_k=max(top_k, 20))
        reranked = reranker.rerank(query=q["body"], results=hybrid_results, top_k=top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [chunk.id for chunk, _ in reranked]
        retrieved_id_lists.append(retrieved_ids)
        relevant_id_sets.append(gold_ids)

        for k in (1, 3, 5, 10):
            per_k_precision[k].append(compute_precision_at_k(retrieved_ids, gold_ids, k))
            per_k_recall[k].append(compute_recall_at_k(retrieved_ids, gold_ids, k))
            per_k_ndcg[k].append(compute_ndcg_at_k(retrieved_ids, gold_ids, k))

    def _mean(xs: List[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    report = {
        "dataset": "BioASQ 13b (gold-snippet retrieval)",
        "num_questions": len(questions),
        "corpus_size": len(chunks),
        "retriever": "Hybrid (BGE-M3 dense + BM25 sparse + RRF) → cross-encoder rerank",
        "metrics": {
            "precision_at_k": {str(k): _mean(per_k_precision[k]) for k in (1, 3, 5, 10)},
            "recall_at_k": {str(k): _mean(per_k_recall[k]) for k in (1, 3, 5, 10)},
            "ndcg_at_k": {str(k): _mean(per_k_ndcg[k]) for k in (1, 3, 5, 10)},
            "mrr": compute_mrr(retrieved_id_lists, relevant_id_sets),
            "retrieval_latency_ms": compute_latency_metrics(latencies_ms),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="BioASQ retrieval evaluation.")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    report = run_evaluation(num_questions=args.num_questions, top_k=args.top_k)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_eval_bioasq_{args.num_questions}q.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
