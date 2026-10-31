"""
Phase 4a — Agent-Contribution / Critic Analysis tests.

Tests for ``backend/evaluation/agent_contribution.py``.

All analysis tests use synthetic fixtures (no real n30 files touched).
One read-only test loads the 6 canonical n30 files to assert they all load
and contain a combined 180 questions.

Evidence tier of these tests: SOFTWARE TEST.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add backend to sys.path (pytest.ini sets pythonpath = backend/app backend, but
# be explicit so the file also works standalone).
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from evaluation.agent_contribution import (  # noqa: E402
    N30_FILES,
    RESULTS_DIR,
    build_concrete_examples,
    build_report,
    cross_config_critic_effect,
    generate_caveats,
    load_result_file,
    render_markdown,
    run_analysis,
    analyse_single_config,
)


# ---------------------------------------------------------------------------
# Fixture builders (synthetic)
# ---------------------------------------------------------------------------
def _sample(question: str, response_type: str, answer: str = "", elapsed_ms: int = 1000) -> Dict[str, Any]:
    return {
        "question": question,
        "true_answer": "gold",
        "predicted_answer": answer,
        "response_type": response_type,
        "evidence_count": 1,
        "evidence_titles": ["Doc 1"],
        "elapsed_ms": elapsed_ms,
    }


def _config_data(samples: List[Dict[str, Any]], *, abstention_rate: float = None,
                 rouge_l_f1: float = None, avg_ms: float = None) -> Dict[str, Any]:
    """Build a minimal result-file dict shaped like the n30 files."""
    metrics: Dict[str, Any] = {}
    if abstention_rate is not None:
        metrics["abstention_rate"] = abstention_rate
    if rouge_l_f1 is not None:
        metrics["rouge_l_f1"] = rouge_l_f1
    if avg_ms is not None:
        metrics["latency_stats"] = {"avg_ms": avg_ms, "median_ms": avg_ms,
                                    "min_ms": avg_ms, "max_ms": avg_ms}
    return {
        "run_metadata": {"config_name": "fixture"},
        "per_sample_results": samples,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# 1. load_result_file
# ---------------------------------------------------------------------------
class TestLoadResultFile:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_result_file(tmp_path / "does_not_exist.json") is None

    def test_reads_json_content(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"per_sample_results": [1, 2, 3]}), encoding="utf-8")
        data = load_result_file(path)
        assert data == {"per_sample_results": [1, 2, 3]}

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        path.write_text(json.dumps({"x": 1}), encoding="utf-8")
        assert load_result_file(str(path)) == {"x": 1}


# ---------------------------------------------------------------------------
# 2. analyse_single_config — abstention counts
# ---------------------------------------------------------------------------
class TestAnalyseSingleConfig:
    def test_abstention_counts(self) -> None:
        samples = [
            _sample("q1", "abstention"),
            _sample("q2", "abstention"),
            _sample("q3", "abstention"),
            _sample("q4", "verified_answer", "answer 4"),
            _sample("q5", "verified_answer", "answer 5"),
        ]
        result = analyse_single_config("full-pipeline", _config_data(samples))
        assert result["total_questions"] == 5
        assert result["abstention_triggered"] == 3
        assert result["verified_answers"] == 2
        assert result["abstention_rate"] == round(3 / 5, 4)

    def test_empty_config(self) -> None:
        result = analyse_single_config("basic-rag", _config_data([]))
        assert result["total_questions"] == 0
        assert result["abstention_triggered"] == 0
        assert result["abstention_rate"] == 0.0
        assert result["avg_elapsed_ms"] == 0.0

    def test_avg_elapsed_ms(self) -> None:
        samples = [
            _sample("q1", "verified_answer", "a", elapsed_ms=1000),
            _sample("q2", "verified_answer", "a", elapsed_ms=3000),
        ]
        result = analyse_single_config("basic-rag", _config_data(samples))
        assert result["avg_elapsed_ms"] == 2000.0

    def test_not_measured_labels_present(self) -> None:
        result = analyse_single_config("no-critic", _config_data([_sample("q", "abstention")]))
        for key in ("critic_accept", "critic_reject", "critic_rewrite"):
            assert "NOT MEASURED" in result[key]
        for key in ("rouge_l_draft_vs_final", "rouge_l_moved_wording",
                    "rouge_l_moved_abstention", "rouge_l_moved_correctness"):
            assert "NOT MEASURED" in result[key]


# ---------------------------------------------------------------------------
# 3. cross_config_critic_effect — delta arithmetic
# ---------------------------------------------------------------------------
class TestCrossConfigCriticEffect:
    def test_delta_arithmetic(self) -> None:
        # full-pipeline: 3/10 abstain -> 0.3 ; no-critic: 1/10 -> 0.1 ; no-cal: 1/10 -> 0.1
        full = _config_data([_sample(f"q{i}", "abstention") for i in range(3)]
                            + [_sample(f"q{i}", "verified_answer", "a") for i in range(7)],
                            abstention_rate=0.3, rouge_l_f1=0.40, avg_ms=100.0)
        no_critic = _config_data([_sample("q0", "abstention")]
                                + [_sample(f"q{i}", "verified_answer", "a") for i in range(9)],
                                abstention_rate=0.1, rouge_l_f1=0.50, avg_ms=90.0)
        no_cal = _config_data([_sample("q0", "abstention")]
                              + [_sample(f"q{i}", "verified_answer", "a") for i in range(9)],
                              abstention_rate=0.1, rouge_l_f1=0.50, avg_ms=90.0)
        res = cross_config_critic_effect(full, no_critic, no_cal)
        assert res["evidence_tier"] == "SOFTWARE TEST"
        assert res["full_pipeline_abstention_rate"] == 0.3
        assert res["no_critic_abstention_rate"] == 0.1
        assert res["no_calibration_abstention_rate"] == 0.1
        # critic delta = full - no-critic = 0.3 - 0.1 = 0.2
        assert res["critic_net_abstention_delta"] == 0.2
        # calibration delta = no-critic - no-cal = 0.1 - 0.1 = 0.0
        assert res["calibration_net_abstention_delta"] == 0.0
        # rouge delta = 0.40 - 0.50 = -0.10
        assert res["rouge_l_delta_full_vs_no_critic"] == -0.10

    def test_empty_configs_rates_zero(self) -> None:
        res = cross_config_critic_effect(_config_data([]), _config_data([]), _config_data([]))
        assert res["full_pipeline_abstention_rate"] == 0.0
        assert res["critic_net_abstention_delta"] == 0.0
        assert res["rouge_l_delta_full_vs_no_critic"] is None  # rouge missing

    def test_has_honest_caveat(self) -> None:
        res = cross_config_critic_effect(_config_data([]), _config_data([]), _config_data([]))
        assert "n=30" in res["caveat"]
        assert "Statistical significance NOT assessed" in res["caveat"]


# ---------------------------------------------------------------------------
# 4. build_concrete_examples — matching by question text
# ---------------------------------------------------------------------------
class TestBuildConcreteExamples:
    def test_matches_by_question_text_type_change(self) -> None:
        full = _config_data([
            _sample("Shared question?", "abstention"),
            _sample("Other question", "verified_answer", "ans"),
        ])
        no_critic = _config_data([
            _sample("Shared question?", "verified_answer", "answered here"),
            _sample("Unrelated", "verified_answer", "ans"),
        ])
        examples = build_concrete_examples(full, no_critic)
        assert len(examples) >= 1
        ex = examples[0]
        assert ex["question_excerpt"].startswith("Shared question?")
        assert ex["full_pipeline_response_type"] == "abstention"
        assert ex["no_critic_response_type"] == "verified_answer"
        assert ex["full_pipeline_answer_excerpt"] == "[abstained]"
        assert "answered here" in ex["no_critic_answer_excerpt"]

    def test_disjoint_questions_no_type_change_returns_empty(self) -> None:
        full = _config_data([_sample("full-only q", "abstention")])
        no_critic = _config_data([_sample("nc-only q", "verified_answer", "a")])
        # No shared questions at all -> no type-change examples and no overlap
        # to top up from -> empty list.
        assert build_concrete_examples(full, no_critic) == []

    def test_capped_at_three(self) -> None:
        shared = [_sample(f"q{i}?", "abstention") for i in range(10)]
        nc = [_sample(f"q{i}?", "verified_answer", "a") for i in range(10)]
        examples = build_concrete_examples(_config_data(shared), _config_data(nc))
        assert len(examples) == 3


# ---------------------------------------------------------------------------
# 5. generate_caveats
# ---------------------------------------------------------------------------
class TestGenerateCaveats:
    def test_returns_nonempty_list(self) -> None:
        caveats = generate_caveats()
        assert isinstance(caveats, list)
        assert len(caveats) >= 3
        assert all(isinstance(c, str) and c for c in caveats)

    def test_caveats_mention_not_measured_and_no_automated_model_judge_fabricated(self) -> None:
        caveats = generate_caveats()
        joined = " ".join(caveats)
        # Honest: explicitly states critic accept/reject/rewrite is NOT MEASURED.
        assert "NOT MEASURED" in joined
        # Honest: explicitly states no AUTOMATED MODEL JUDGE tier was generated.
        assert "AUTOMATED MODEL JUDGE" in joined
        # And that it was *not* generated (not used as an evidence tier).
        assert "No AUTOMATED MODEL JUDGE tier evidence was generated" in joined


# ---------------------------------------------------------------------------
# 6. build_report — evidence tier + NOT MEASURED + no AUTOMATED MODEL JUDGE tier
# ---------------------------------------------------------------------------
def _collect_evidence_tiers(obj: Any, acc: List[str]) -> None:
    """Recursively collect every ``evidence_tier`` value found in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "evidence_tier" and isinstance(v, str):
                acc.append(v)
            else:
                _collect_evidence_tiers(v, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_evidence_tiers(item, acc)


class TestBuildReport:
    def test_evidence_tier_software_test(self) -> None:
        report = build_report({"full-pipeline": _config_data([_sample("q", "abstention")])}, [])
        assert report["evidence_tier"] == "SOFTWARE TEST"

    def test_top_level_not_measured_labels(self) -> None:
        report = build_report({"full-pipeline": _config_data([_sample("q", "abstention")])}, [])
        for key in ("critic_accept", "critic_reject", "critic_rewrite",
                    "rouge_l_moved_wording", "rouge_l_moved_abstention",
                    "rouge_l_moved_correctness"):
            assert "NOT MEASURED" in report[key]
        for cfg in report["per_config_analysis"].values():
            for key in ("critic_accept", "critic_reject", "critic_rewrite",
                        "rouge_l_draft_vs_final", "rouge_l_moved_wording",
                        "rouge_l_moved_abstention", "rouge_l_moved_correctness"):
                assert "NOT MEASURED" in cfg[key]

    def test_no_automated_model_judge_tier_fabricated(self) -> None:
        report = build_report(
            {
                "full-pipeline": _config_data([_sample("q", "abstention")]),
                "no-critic": _config_data([_sample("q", "verified_answer", "a")]),
                "no-calibration": _config_data([_sample("q", "verified_answer", "a")]),
            },
            [],
        )
        tiers: List[str] = []
        _collect_evidence_tiers(report, tiers)
        assert tiers, "expected at least one evidence_tier field"
        assert all(t == "SOFTWARE TEST" for t in tiers)
        assert "AUTOMATED MODEL JUDGE" not in tiers

    def test_total_questions_sums_configs(self) -> None:
        report = build_report(
            {
                "full-pipeline": _config_data([_sample("q", "a") for _ in range(3)]),
                "no-critic": _config_data([_sample("q", "a") for _ in range(7)]),
            },
            [],
        )
        assert report["total_questions"] == 10
        assert set(report["configs_analysed"]) == {"full-pipeline", "no-critic"}

    def test_missing_files_recorded(self) -> None:
        report = build_report({}, ["missing.json"])
        assert report["missing_files"] == ["missing.json"]

    def test_cross_config_populated_when_all_three_present(self) -> None:
        report = build_report(
            {
                "full-pipeline": _config_data([_sample("q", "abstention")], abstention_rate=1.0,
                                             rouge_l_f1=0.1, avg_ms=10.0),
                "no-critic": _config_data([_sample("q", "verified_answer", "a")],
                                          abstention_rate=0.0, rouge_l_f1=0.2, avg_ms=20.0),
                "no-calibration": _config_data([_sample("q", "verified_answer", "a")],
                                                abstention_rate=0.0, rouge_l_f1=0.2,
                                                avg_ms=20.0),
            },
            [],
        )
        cc = report["cross_config_critic_effect"]
        assert cc is not None
        assert cc["evidence_tier"] == "SOFTWARE TEST"
        assert cc["critic_net_abstention_delta"] == 1.0


# ---------------------------------------------------------------------------
# 7. render_markdown
# ---------------------------------------------------------------------------
class TestRenderMarkdown:
    def test_contains_required_headers(self) -> None:
        report = build_report(
            {"full-pipeline": _config_data([_sample("q1", "abstention"),
                                            _sample("q2", "verified_answer", "a")])},
            [],
        )
        md = render_markdown(report)
        assert "# Critic Analysis Report" in md
        assert "## Critical Data Limitation" in md
        assert "## Per-Config Abstention Analysis" in md
        assert "NOT MEASURABLE" in md
        assert report["evidence_tier"] in md

    def test_cross_config_table_present(self) -> None:
        report = build_report(
            {
                "full-pipeline": _config_data([_sample("q", "abstention")], abstention_rate=1.0),
                "no-critic": _config_data([_sample("q", "verified_answer", "a")],
                                          abstention_rate=0.0),
                "no-calibration": _config_data([_sample("q", "verified_answer", "a")],
                                                abstention_rate=0.0),
            },
            [],
        )
        md = render_markdown(report)
        assert "Cross-Config Critic + Calibration Effect" in md
        assert "critic_net_abstention_delta" not in md  # human label, not raw key


# ---------------------------------------------------------------------------
# 8. run_analysis — writes outputs (synthetic fixtures, tmp dirs)
# ---------------------------------------------------------------------------
class TestRunAnalysisWritesOutputs:
    def test_writes_timestamped_json_and_md(self, tmp_path: Path) -> None:
        results = tmp_path / "in"
        results.mkdir()
        for name, filename in N30_FILES.items():
            data = _config_data([_sample(f"q{i}", "abstention" if i == 0 else "verified_answer",
                                        answer="a") for i in range(3)])
            (results / filename).write_text(json.dumps(data), encoding="utf-8")

        out = tmp_path / "out"
        report = run_analysis(results_dir=results, output_dir=out)

        jsons = list(out.glob("critic_analysis_*.json"))
        mds = list(out.glob("critic_analysis_*.md"))
        assert len(jsons) == 1
        assert len(mds) == 1
        written = json.loads(jsons[0].read_text(encoding="utf-8"))
        assert written["evidence_tier"] == "SOFTWARE TEST"
        assert "NOT MEASURED" in written["critic_accept"]
        assert written["total_questions"] == 18  # 6 configs * 3 questions
        # No leftover .tmp files.
        assert not list(out.glob("*.tmp"))
        # The returned report equals what was written (minus nothing).
        assert report["evidence_tier"] == written["evidence_tier"]


# ---------------------------------------------------------------------------
# 9. Read-only test against the real n30 files
# ---------------------------------------------------------------------------
class TestRealN30Files:
    def test_all_six_load_and_total_180(self) -> None:
        loaded = {}
        for config_name, filename in N30_FILES.items():
            data = load_result_file(RESULTS_DIR / filename)
            assert data is not None, f"missing n30 file for {config_name}: {filename}"
            loaded[config_name] = data

        assert len(loaded) == 6
        total = sum(len(d.get("per_sample_results", [])) for d in loaded.values())
        assert total == 180

        # Per-config analysis on the real files: honest labels must hold.
        for config_name, data in loaded.items():
            res = analyse_single_config(config_name, data)
            assert res["total_questions"] == 30
            assert "NOT MEASURED" in res["critic_accept"]
            assert "NOT MEASURED" in res["rouge_l_draft_vs_final"]
        # evidence_tier is a report-level field; verify the full report too.
        report = build_report(loaded, [])
        assert report["evidence_tier"] == "SOFTWARE TEST"
        tiers: List[str] = []
        _collect_evidence_tiers(report, tiers)
        assert all(t == "SOFTWARE TEST" for t in tiers)
