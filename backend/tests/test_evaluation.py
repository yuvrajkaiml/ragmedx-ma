import pytest
from pathlib import Path

from evaluation.metrics import (
    compute_accuracy,
    compute_abstention_rate,
    compute_citation_support_rate,
    compute_latency_metrics,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_batch,
)
from evaluation.mirage_runner import BenchmarkRunner


# ==============================================================================
# Metrics Unit Tests
# ==============================================================================


class TestAccuracy:
    def test_perfect_accuracy(self):
        preds = ["A", "B", "C"]
        truths = ["A", "B", "C"]
        assert compute_accuracy(preds, truths) == 1.0

    def test_zero_accuracy(self):
        preds = ["A", "B", "C"]
        truths = ["X", "Y", "Z"]
        assert compute_accuracy(preds, truths) == 0.0

    def test_partial_accuracy(self):
        preds = ["A", "B", "X"]
        truths = ["A", "B", "C"]
        assert compute_accuracy(preds, truths) == pytest.approx(2 / 3, abs=0.01)

    def test_case_insensitive_match(self):
        preds = ["ace inhibitors", "THIAZIDE"]
        truths = ["ACE Inhibitors", "thiazide"]
        assert compute_accuracy(preds, truths) == 1.0

    def test_empty_inputs(self):
        assert compute_accuracy([], []) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert compute_precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_none_relevant(self):
        retrieved = ["d4", "d5", "d6"]
        relevant = {"d1", "d2", "d3"}
        assert compute_precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_relevant(self):
        retrieved = ["d1", "d4", "d2", "d5", "d6"]
        relevant = {"d1", "d2", "d3"}
        assert compute_precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5, abs=0.01)

    def test_empty_retrieved(self):
        assert compute_precision_at_k([], {"d1"}, k=5) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self):
        retrieved = ["d1", "d4"]
        relevant = {"d1", "d2", "d3"}
        assert compute_recall_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 3, abs=0.01)

    def test_zero_recall(self):
        retrieved = ["d4", "d5"]
        relevant = {"d1", "d2"}
        assert compute_recall_at_k(retrieved, relevant, k=5) == 0.0


class TestAbstentionRate:
    def test_no_abstentions(self):
        responses = [
            {"response_type": "verified_answer"},
            {"response_type": "verified_answer"},
        ]
        assert compute_abstention_rate(responses) == 0.0

    def test_all_abstentions(self):
        responses = [
            {"response_type": "abstention"},
            {"response_type": "abstention"},
        ]
        assert compute_abstention_rate(responses) == 1.0

    def test_mixed(self):
        responses = [
            {"response_type": "verified_answer"},
            {"response_type": "abstention"},
            {"response_type": "verified_answer"},
            {"response_type": "abstention"},
        ]
        assert compute_abstention_rate(responses) == 0.5

    def test_empty(self):
        assert compute_abstention_rate([]) == 0.0


class TestCitationSupportRate:
    def test_all_cited(self):
        responses = [
            {"citations": [{"marker": "[E1]"}]},
            {"citations": [{"marker": "[E1]"}, {"marker": "[E2]"}]},
        ]
        assert compute_citation_support_rate(responses) == 1.0

    def test_none_cited(self):
        responses = [
            {"citations": []},
            {"citations": []},
        ]
        assert compute_citation_support_rate(responses) == 0.0


class TestLatencyMetrics:
    def test_basic_latencies(self):
        latencies = [100, 200, 300]
        result = compute_latency_metrics(latencies)
        assert result["avg_ms"] == 200.0
        assert result["median_ms"] == 200.0
        assert result["min_ms"] == 100.0
        assert result["max_ms"] == 300.0

    def test_empty_latencies(self):
        result = compute_latency_metrics([])
        assert result["avg_ms"] == 0.0


class TestEvaluateBatch:
    def test_batch_with_ground_truths(self):
        preds = [
            {"answer": "A", "processing_time_ms": 100, "response_type": "verified_answer", "citations": [{"m": "[E1]"}]},
            {"answer": "B", "processing_time_ms": 200, "response_type": "abstention", "citations": []},
        ]
        truths = [
            {"answer": "A"},
            {"answer": "B"},
        ]
        report = evaluate_batch(preds, truths)
        assert report["total_samples"] == 2
        assert report["accuracy"] == 1.0
        assert report["abstention_rate"] == 0.5
        assert report["citation_support_rate"] == 0.5

    def test_empty_batch(self):
        report = evaluate_batch([], [])
        assert report["status"] == "empty"


# ==============================================================================
# BenchmarkRunner Tests
# ==============================================================================


class TestBenchmarkRunner:
    def test_load_dataset_from_fixture(self):
        fixture_path = Path(__file__).parent.parent / "data" / "datasets" / "medqa" / "sample.jsonl"
        runner = BenchmarkRunner(dataset_path=fixture_path)
        samples = runner.load_dataset()
        assert len(samples) >= 1
        assert "question" in samples[0]
        assert "answer" in samples[0]

    def test_missing_dataset_returns_blocked(self):
        runner = BenchmarkRunner(dataset_path=Path("/nonexistent/missing.jsonl"))
        samples = runner.load_dataset()
        assert len(samples) == 0

    @pytest.mark.asyncio
    async def test_run_evaluation_over_synthetic_fixture(self):
        """Run evaluation scaffold over synthetic MedQA fixture."""

        class FakeLLM:
            def __init__(self):
                self.responses = [
                    '{"medical_specialty": "Cardiology", "confidence": 0.90}',
                    "ACE inhibitors are recommended [E1].",
                    '{"passed": true, "factual_consistency": 0.90, "citation_support": 0.90, "hallucination_risk": 0.05, "confidence": 0.90}',
                ]
                self.idx = 0

            def generate(self, prompt, **kwargs):
                if self.idx < len(self.responses):
                    r = self.responses[self.idx]
                    self.idx += 1
                else:
                    r = self.responses[0]
                    self.idx = 1
                return r, {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20, "is_mock": True}

        fixture_path = Path(__file__).parent.parent / "data" / "datasets" / "medqa" / "sample.jsonl"
        runner = BenchmarkRunner(
            dataset_path=fixture_path,
            llm_wrapper=FakeLLM(),
        )
        result = await runner.run_evaluation(max_samples=1)

        assert result["status"] == "COMPLETED_MOCK_SCAFFOLD"
        assert result["samples_evaluated"] == 1
        assert "metrics" in result
        assert result["metrics"]["total_samples"] == 1

    @pytest.mark.asyncio
    async def test_blocked_on_missing_dataset(self):
        runner = BenchmarkRunner(dataset_path=Path("/nonexistent/missing.jsonl"))
        result = await runner.run_evaluation()
        assert result["status"] == "BLOCKED_ON_DATASET"
        assert result["total_samples"] == 0
