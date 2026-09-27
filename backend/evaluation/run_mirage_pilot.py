"""Timed, single-pass functional pilot over official MIRAGE questions.

This is deliberately *not* an accuracy evaluation: it uses only the committed
five-record PubMed seed corpus because MedRAG's official full corpus cannot fit
on this host.  Its purpose is to measure one warm local Qwen graph invocation per
question and preserve the exact configuration for later budget planning.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.state import AgentState, create_deterministic_uuid
from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper
from app.rag.chunking.models import MedicalChunk
from app.rag.embeddings.bge_embedding import BGEEmbedding
from app.rag.pipeline import RAGPipeline, RAGRetrievalPipeline
from app.rag.rerankers.cross_encoder_reranker import CrossEncoderReranker
from app.rag.retrievers.dense_retriever import DenseRetriever
from app.rag.retrievers.hybrid_retriever import HybridRetriever
from app.rag.retrievers.sparse_retriever import SparseRetriever
from app.rag.vectorstore.faiss_store import FAISSVectorStore
from app.services.retrieval_service import RetrievalService


class PilotLLMWrapper(LocalLLMWrapper):
    """Real local wrapper with deterministic, recorded pilot generation limits."""

    MAX_NEW_TOKENS = 128

    def generate(self, prompt: str, **kwargs: Any):
        kwargs.setdefault("max_new_tokens", self.MAX_NEW_TOKENS)
        kwargs["temperature"] = 0.0
        return super().generate(prompt, **kwargs)


def load_questions(benchmark_path: Path, per_task: int) -> list[dict[str, Any]]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = []
    for task_name in ("medqa", "medmcqa", "pubmedqa", "bioasq", "mmlu"):
        task_items = benchmark[task_name]
        for item_id, item in list(task_items.items())[:per_task]:
            questions.append({"task": task_name, "item_id": item_id, **item})
    return questions


def build_seed_service(index_dir: Path) -> RetrievalService:
    seed_path = Path("tests/fixtures/pubmed_hypertension_seed.jsonl")
    records = [json.loads(line) for line in seed_path.read_text(encoding="utf-8").splitlines() if line]
    chunks = [
        MedicalChunk(
            id=f"pubmed-{record['pmid']}",
            document_id=record["pmid"],
            content=record["content"],
            title=record["title"],
            source="pubmed",
            journal=record["journal"],
            publication_year=record["publication_year"],
            metadata={"pmid": record["pmid"], "doi": record["doi"]},
        )
        for record in records
    ]
    # Qwen occupies most of the 8 GB GPU, so retrieval models run on CPU.
    embedder = BGEEmbedding(device="cpu")
    store = FAISSVectorStore(dimension=1024)
    store.add(embedder.embed_documents([chunk.content for chunk in chunks]), chunks)
    store.save(index_dir)
    persisted_store = FAISSVectorStore.load(index_dir)
    sparse = SparseRetriever()
    sparse.add_chunks(chunks)
    native = RAGRetrievalPipeline(
        hybrid_retriever=HybridRetriever(
            dense_retriever=DenseRetriever(embedder=embedder, vector_store=persisted_store),
            sparse_retriever=sparse,
        ),
        reranker=CrossEncoderReranker(device="cpu"),
        retrieval_top_k=5,
        final_top_k=3,
    )
    return RetrievalService(RAGPipeline(native, top_k_final=3))


async def run_pilot(questions: list[dict[str, Any]], service: RetrievalService, llm: PilotLLMWrapper) -> list[dict[str, Any]]:
    orchestrator = MultiAgentOrchestrator(
        llm_wrapper=llm,
        retrieval_provider=service.retrieve_evidence,
    )
    results = []
    for ordinal, item in enumerate(questions, start=1):
        started = time.perf_counter()
        state = await orchestrator.run(AgentState(
            query_id=create_deterministic_uuid(f"mirage-pilot-{item['task']}-{item['item_id']}"),
            original_question=item["question"],
            include_trace=True,
        ))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        nodes = [
            step["details"].get("node") for step in state.agent_trace
            if step.get("agent_type") == "langgraph"
        ]
        result = {
            "ordinal": ordinal,
            "task": item["task"],
            "item_id": item["item_id"],
            "question": item["question"],
            "reference_answer": item["answer"],
            "status": state.status,
            "elapsed_ms": elapsed_ms,
            "graph_nodes": nodes,
            "error_code": state.error_code,
            "response_preview": (state.answer or "")[:500],
        }
        results.append(result)
        print(f"[{ordinal}/{len(questions)}] {item['task']} {elapsed_ms} ms status={state.status}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=Path("data/datasets/mirage/benchmark.json"))
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/cache/mirage_pilot_result.json"))
    args = parser.parse_args()

    questions = load_questions(args.benchmark, args.per_task)
    setup_started = time.perf_counter()
    service = build_seed_service(Path("data/cache/mirage_pilot_seed_index"))
    retrieval_setup_ms = int((time.perf_counter() - setup_started) * 1000)

    llm = PilotLLMWrapper(
        model_name_or_path="Qwen/Qwen3-8B",
        quantization_bits=4,
        device_map="auto",
        use_mock_fallback=False,
    )
    load_started = time.perf_counter()
    llm.load_model()
    model_load_ms = int((time.perf_counter() - load_started) * 1000)
    if llm.active_runtime_mode != LLMRuntimeMode.LOCAL:
        raise RuntimeError("Pilot requires the real local model; mock mode is not accepted.")

    questions_started = time.perf_counter()
    results = asyncio.run(run_pilot(questions, service, llm))
    questions_wall_clock_ms = int((time.perf_counter() - questions_started) * 1000)
    payload = {
        "status": "COMPLETED_FUNCTIONAL_COST_PILOT",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "official MIRAGE benchmark.json (Xiong et al., 2024)",
        "selection": {"method": "first N records per task in repository order", "per_task": args.per_task, "count": len(questions)},
        "corpus": {
            "name": "committed five-document real PubMed hypertension seed corpus",
            "is_official_mirage_corpus": False,
            "limitation": "Functional/cost pilot only; do not interpret output quality or accuracy as a MIRAGE result.",
            "retrieval_device": "cpu",
        },
        "model": {"name": llm.model_name_or_path, "mode": llm.active_runtime_mode.value, "quantization": "4-bit NF4", "max_new_tokens_per_agent_call": llm.MAX_NEW_TOKENS, "temperature": 0.0},
        "timing": {"retrieval_setup_ms": retrieval_setup_ms, "model_load_ms": model_load_ms, "questions_wall_clock_ms": questions_wall_clock_ms, "mean_question_ms": round(questions_wall_clock_ms / len(questions), 2)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["timing"], indent=2))
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
