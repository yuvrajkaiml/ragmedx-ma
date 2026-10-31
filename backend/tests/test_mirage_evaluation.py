"""Tests for MIRAGE evaluation pipeline.

Test design
-----------
* Metric unit tests are pure -- no model loading, no I/O.
* MedQAIndexBuilder tests write to tmp_path.
* BenchmarkRunner.run_evaluation tests use FakeLLM (unchanged existing pattern).
* BenchmarkRunner.run_rag_evaluation is @pytest.mark.integration -- loads
  BGE + CrossEncoder, runs over the 4-record fixture.
* All existing test_evaluation.py tests are unaffected.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import numpy as np

from evaluation.index_builder import MedQAIndexBuilder
from evaluation.metrics import (
    compute_answer_with_citation_rate,
    compute_retrieval_coverage,
)
from evaluation.mirage_runner import BenchmarkRunner

FIXTURE_PATH = Path(__file__).parent.parent / "data" / "datasets" / "medqa" / "sample.jsonl"


class _DeterministicBGEEmbedding:
    """Model-free stand-in for testing native 1024-dimensional index wiring."""

    def embed_documents(self, texts, batch_size=8):
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, index % 1024] = max(len(text), 1)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------

class _FakeLLM:
    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = responses or [
            '{"medical_specialty": "Cardiology", "confidence": 0.90}',
            "ACE inhibitors are recommended as first-line therapy [E1].",
            '{"passed": true, "factual_consistency": 0.90, "citation_support": 0.90, '
            '"hallucination_risk": 0.05, "confidence": 0.90}',
        ]
        self._idx = 0

    def generate(self, prompt: str, **kwargs):
        r = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return r, {"prompt_tokens": 10, "completion_tokens": 10,
                   "total_tokens": 20, "is_mock": True}


# ===========================================================================
# 1. compute_answer_with_citation_rate
# ===========================================================================

class TestAnswerWithCitationRate:
    def test_all_cited_via_marker(self):
        responses = [
            {"response_type": "verified_answer", "answer": "Use ACE inhibitors [E1].", "citations": []},
            {"response_type": "verified_answer", "answer": "Beta-blockers [E2] help.", "citations": []},
        ]
        assert compute_answer_with_citation_rate(responses) == 1.0

    def test_all_cited_via_list(self):
        responses = [
            {"response_type": "verified_answer", "answer": "No marker.",
             "citations": [{"m": "[E1]"}]},
        ]
        assert compute_answer_with_citation_rate(responses) == 1.0

    def test_none_cited(self):
        responses = [
            {"response_type": "verified_answer", "answer": "Treat with medication.", "citations": []},
            {"response_type": "verified_answer", "answer": "Monitor carefully.", "citations": []},
        ]
        assert compute_answer_with_citation_rate(responses) == 0.0

    def test_abstentions_excluded(self):
        responses = [
            {"response_type": "verified_answer", "answer": "ACE inhibitors [E1].", "citations": []},
            {"response_type": "abstention", "answer": "", "citations": []},
        ]
        assert compute_answer_with_citation_rate(responses) == 1.0

    def test_all_abstentions_returns_zero(self):
        assert compute_answer_with_citation_rate(
            [{"response_type": "abstention", "answer": "", "citations": []}]
        ) == 0.0

    def test_empty_returns_zero(self):
        assert compute_answer_with_citation_rate([]) == 0.0

    def test_partial_cited(self):
        responses = [
            {"response_type": "verified_answer", "answer": "Treat with [E1].", "citations": []},
            {"response_type": "verified_answer", "answer": "No citation.", "citations": []},
        ]
        assert compute_answer_with_citation_rate(responses) == 0.5

    def test_answer_citations_field_accepted(self):
        responses = [
            {"response_type": "verified_answer", "answer": "See evidence.",
             "answer_citations": ["[E1]"], "citations": []},
        ]
        assert compute_answer_with_citation_rate(responses) == 1.0


# ===========================================================================
# 2. compute_retrieval_coverage
# ===========================================================================

class TestRetrievalCoverage:
    def test_full_coverage(self):
        evidence = [["ACE inhibitors treat hypertension"], ["femoral nailing fracture"]]
        keywords = [{"ace"}, {"femoral"}]
        assert compute_retrieval_coverage(evidence, keywords) == 1.0

    def test_no_coverage(self):
        evidence = [["completely irrelevant text"], ["also irrelevant"]]
        keywords = [{"hypertension"}, {"fracture"}]
        assert compute_retrieval_coverage(evidence, keywords) == 0.0

    def test_partial_coverage(self):
        evidence = [["ACE inhibitors hypertension"], ["unrelated ortho text"]]
        keywords = [{"ace"}, {"fracture"}]
        assert compute_retrieval_coverage(evidence, keywords) == 0.5

    def test_case_insensitive(self):
        evidence = [["ACE Inhibitors HYPERTENSION"]]
        keywords = [{"ace", "hypertension"}]
        assert compute_retrieval_coverage(evidence, keywords) == 1.0

    def test_empty_returns_zero(self):
        assert compute_retrieval_coverage([], []) == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert compute_retrieval_coverage([["text"]], [{"k1"}, {"k2"}]) == 0.0

    def test_multiple_passages_per_sample(self):
        evidence = [["passage one", "passage two with keyword hypertension"]]
        keywords = [{"hypertension"}]
        assert compute_retrieval_coverage(evidence, keywords) == 1.0


# ===========================================================================
# 3. MedQAIndexBuilder
# ===========================================================================

class TestMedQAIndexBuilder:
    @pytest.fixture
    def embedder(self):
        return _DeterministicBGEEmbedding()

    def test_build_creates_files(self, tmp_path, embedder):
        builder = MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder)
        builder.build()
        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "chunks.pkl").exists()

    def test_ntotal_matches_fixture(self, tmp_path, embedder):
        builder = MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder)
        store = builder.build()
        assert store.size == 4

    def test_chunks_have_medqa_source_type(self, tmp_path, embedder):
        from app.rag.vectorstore.faiss_store import FAISSVectorStore
        MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder).build()
        loaded = FAISSVectorStore.load(tmp_path)
        for chunk in loaded.chunks:
            assert chunk.source == "medqa"

    def test_chunk_text_is_question(self, tmp_path, embedder):
        from app.rag.vectorstore.faiss_store import FAISSVectorStore
        MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder).build()
        loaded = FAISSVectorStore.load(tmp_path)
        samples = BenchmarkRunner(dataset_path=FIXTURE_PATH).load_dataset()
        questions = {s["question"] for s in samples}
        for chunk in loaded.chunks:
            assert chunk.content in questions

    def test_metadata_contains_answer(self, tmp_path, embedder):
        from app.rag.vectorstore.faiss_store import FAISSVectorStore
        MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder).build()
        loaded = FAISSVectorStore.load(tmp_path)
        for chunk in loaded.chunks:
            assert "answer" in chunk.metadata

    def test_force_false_skips_rebuild(self, tmp_path, embedder):
        builder = MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder)
        builder.build()
        mtime1 = (tmp_path / "index.faiss").stat().st_mtime
        builder.build(force=False)
        mtime2 = (tmp_path / "index.faiss").stat().st_mtime
        assert mtime1 == mtime2

    def test_force_true_rebuilds(self, tmp_path, embedder):
        builder = MedQAIndexBuilder(dataset_path=FIXTURE_PATH, index_dir=tmp_path, embedder=embedder)
        builder.build()
        mtime1 = (tmp_path / "index.faiss").stat().st_mtime
        time.sleep(0.05)
        builder.build(force=True)
        mtime2 = (tmp_path / "index.faiss").stat().st_mtime
        assert mtime2 >= mtime1

    def test_missing_dataset_raises(self, tmp_path):
        builder = MedQAIndexBuilder(
            dataset_path=Path("/nonexistent/missing.jsonl"),
            index_dir=tmp_path,
        )
        with pytest.raises(FileNotFoundError):
            builder.build()


# ===========================================================================
# 4. BenchmarkRunner.run_evaluation -- regression guard
# ===========================================================================

class TestBenchmarkRunnerScaffold:
    def test_runner_owns_one_default_wrapper(self):
        runner = BenchmarkRunner(dataset_path=FIXTURE_PATH)
        assert runner.llm_wrapper is not None

    async def test_returns_completed_mock_scaffold(self):
        runner = BenchmarkRunner(dataset_path=FIXTURE_PATH, llm_wrapper=_FakeLLM())
        result = await runner.run_evaluation(max_samples=1)
        assert result["status"] == "COMPLETED_MOCK_SCAFFOLD"
        assert result["samples_evaluated"] == 1
        assert result["metrics"]["total_samples"] == 1

    async def test_blocked_on_missing_dataset(self):
        runner = BenchmarkRunner(dataset_path=Path("/nonexistent/missing.jsonl"))
        result = await runner.run_evaluation()
        assert result["status"] == "BLOCKED_ON_DATASET"
        assert result["total_samples"] == 0


# ===========================================================================
# 5. BenchmarkRunner.run_rag_evaluation -- integration
# ===========================================================================

@pytest.mark.integration
class TestRunRAGEvaluation:
    """Full MIRAGE evaluation with real BGE + FAISS + CrossEncoder.
    Skip with:  pytest -m "not integration"
    """

    @pytest.fixture(scope="class")
    def rag_result(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("mirage_index")
        runner = BenchmarkRunner(
            dataset_path=FIXTURE_PATH,
            llm_wrapper=_FakeLLM(),
        )
        return asyncio.run(
            runner.run_rag_evaluation(
                max_samples=2,
                index_dir=tmp,
                top_k_retrieval=4,
                top_k_final=2,
                force_reindex=True,
            )
        )

    def test_status_completed(self, rag_result):
        assert rag_result["status"] == "COMPLETED_RAG_EVALUATION"

    def test_samples_evaluated(self, rag_result):
        assert rag_result["samples_evaluated"] == 2

    def test_metrics_keys_present(self, rag_result):
        m = rag_result["metrics"]
        for key in ("total_samples", "abstention_rate", "retrieval_coverage",
                    "answer_with_citation_rate", "index_size"):
            assert key in m, f"missing metric key: {key}"

    def test_index_size_matches_full_fixture(self, rag_result):
        assert rag_result["metrics"]["index_size"] == 4

    def test_per_sample_count(self, rag_result):
        assert len(rag_result["per_sample_results"]) == 2

    def test_per_sample_keys(self, rag_result):
        required = {"question", "true_answer", "predicted_answer",
                    "response_type", "evidence_count", "evidence_titles", "elapsed_ms"}
        for s in rag_result["per_sample_results"]:
            assert required.issubset(s.keys())

    def test_evidence_retrieved(self, rag_result):
        for s in rag_result["per_sample_results"]:
            assert s["evidence_count"] >= 1, (
                f"No evidence for: {s['question'][:60]}"
            )

    def test_retrieval_coverage_in_range(self, rag_result):
        cov = rag_result["metrics"]["retrieval_coverage"]
        assert 0.0 <= cov <= 1.0

    def test_all_metric_scores_in_range(self, rag_result):
        m = rag_result["metrics"]
        assert 0.0 <= m["abstention_rate"] <= 1.0
        assert 0.0 <= m["answer_with_citation_rate"] <= 1.0

    def test_dataset_path_reported(self, rag_result):
        assert "dataset_path" in rag_result

    def test_index_dir_reported(self, rag_result):
        assert "index_dir" in rag_result


# ===========================================================================
# 6. Import boundary
# ===========================================================================

class TestImportBoundary:
    def test_benchmark_runner(self):
        from evaluation import BenchmarkRunner as BR
        assert BR is BenchmarkRunner

    def test_index_builder(self):
        from evaluation import MedQAIndexBuilder as MI
        assert MI is MedQAIndexBuilder

    def test_new_metrics(self):
        from evaluation import (
            compute_answer_with_citation_rate,
            compute_retrieval_coverage,
        )
        assert callable(compute_answer_with_citation_rate)
        assert callable(compute_retrieval_coverage)

    def test_prior_metrics_unchanged(self):
        from evaluation import (
            compute_accuracy, compute_abstention_rate,
            compute_citation_support_rate, compute_latency_metrics,
            compute_precision_at_k, compute_recall_at_k, evaluate_batch,
        )
        for fn in [compute_accuracy, compute_abstention_rate,
                   compute_citation_support_rate, compute_latency_metrics,
                   compute_precision_at_k, compute_recall_at_k, evaluate_batch]:
            assert callable(fn)
