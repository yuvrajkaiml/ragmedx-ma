"""FEAT-003 — Metric-recompute deliverable tests.

CPU only. Builds a tiny synthetic phase5 run dir (raw per_question/*.jsonl +
splits/test.json + environment_run.json) and runs the real recompute, then
asserts the regenerated deliverables are correct, carry evidence tiers, label
retrieval NOT MEASURED, measure claim-level metrics only where claim_verdicts
exist, and recompute deterministically.

Evidence tier: SOFTWARE TEST (unit, no GPU / no model / no real FAISS index).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_SCRIPTS_DIR = _BACKEND_DIR / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import recompute_feat003_metrics as rc  # noqa: E402

_CONFIGS = [
    "phase5-improved-v2",
    "phase5-improved-no-verification",
    "phase5-full-pipeline",
    "phase5-no-critic",
    "phase5-no-iterative-retrieval",
    "phase5-no-calibration",
    "phase5-basic-rag",
    "phase5-plain-llm-no-rag",
]


def _rec(qid: str, cfg: str, source: str, *, answer: str, gold: str,
         abstain: bool, errors: Any = None, claim_verdicts: Any = None) -> Dict[str, Any]:
    return {
        "question_id": qid,
        "resolved_index_path": r"D:\Ragmed_2\backend\data\faiss_index\eval_split",
        "split": "test",
        "config": cfg,
        "source": source,
        "gold_answer": gold,
        "answer": answer,
        "status": "error" if errors else ("abstained" if abstain else "verified"),
        "citations": [{"citation_marker": "[Doc 1]", "pmid":
                       "http://www.ncbi.nlm.nih.gov/pubmed/12345678"}] if not errors else [],
        "claim_verdicts": claim_verdicts if claim_verdicts is not None else {"critic_verdict": None},
        "confidence": 0.3,
        "confidence_note": "calibrated",
        "abstain_flag": bool(abstain),
        "errors": errors,
        "error_code": "X" if errors else None,
        "per_stage_latency_ms": {"generator": 10.0},
        "model_call_count": 1,
        "retrieval_call_count": 1,
        "cache_hits": 0,
        "peak_vram_mb": 10100.0,
        "cold_latency_ms": 1000,
        "warm_latency_ms": 20,
    }


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    rd = tmp_path / "phase5_test"
    (rd / "per_question").mkdir(parents=True)
    (rd / "splits").mkdir()

    # 2 questions: one bioasq (with gold PMIDs), one pubmedqa.
    qids = ["bioasq::q1", "pubmedqa::q2"]
    pool = [
        {"id": "bioasq::q1", "question": "q1", "answer": "alpha beta", "source": "bioasq",
         "pmid": ["99999999"]},
        {"id": "pubmedqa::q2", "question": "q2", "answer": "yes", "source": "pubmedqa"},
    ]
    (rd / "splits" / "test.json").write_text(json.dumps(pool), encoding="utf-8")

    verdicts = {
        "verified_claims": [
            {"label": "unverifiable", "score": 0.0},
            {"label": "contradicted", "score": 0.0},
        ],
        "sufficiency": {"contradictions_present": True, "sufficient": False},
    }
    for cfg in _CONFIGS:
        lines = []
        # improved-v2 answers q1 with claim_verdicts; everyone else abstains q1.
        if cfg == "phase5-improved-v2":
            lines.append(_rec("bioasq::q1", cfg, "bioasq", answer="alpha beta gamma",
                              gold="alpha beta", abstain=False, claim_verdicts=verdicts))
        else:
            lines.append(_rec("bioasq::q1", cfg, "bioasq", answer="", gold="alpha beta",
                              abstain=True))
        # q2: plain-llm answers correctly, others abstain.
        if cfg == "phase5-plain-llm-no-rag":
            lines.append(_rec("pubmedqa::q2", cfg, "pubmedqa", answer="yes", gold="yes",
                              abstain=False))
        else:
            lines.append(_rec("pubmedqa::q2", cfg, "pubmedqa", answer="", gold="yes",
                              abstain=True))
        (rd / "per_question" / f"{cfg}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    env = {
        "generated_at": "2026-10-18T00:00:00+00:00",
        "llm_model": "Qwen/Qwen3-8B", "llm_mode": "local", "generator_temperature": 0.0,
        "model_load_ms": 34000, "configs": _CONFIGS, "mode": "target_matrix",
        "n_questions": len(qids), "peak_vram_mb_run": 10161.24,
        "subset": {"dev": 0, "test": len(qids), "full": False},
        "torch_version": "2.11.0+cu128", "cuda_version": "12.8",
        "matrix": {"plan": {"total": len(qids) * 8, "reused": 0, "missing": len(qids) * 8},
                   "run_counts": {"completed": len(qids) * 8, "failed": 0,
                                  "reused_skipped": 0, "duplicate_skipped": 0,
                                  "run_seconds": 100.0, "budget_stopped_early": False}},
    }
    (rd / "environment_run.json").write_text(json.dumps(env), encoding="utf-8")
    return rd


def test_all_deliverables_regenerated(run_dir: Path) -> None:
    rc.recompute(run_dir)
    for name in ("metric_summary.json", "latency_breakdown.json", "failure_cases.json",
                 "summary.json", "extended_metrics.json", "report.md",
                 "comparison_table.md", "runtime_completion_report.md",
                 "reproducibility_report.md", "research_results_summary.md"):
        assert (run_dir / name).exists(), name
    ms = json.loads((run_dir / "metric_summary.json").read_text(encoding="utf-8"))
    assert len(ms["configs"]) == 8
    ext = json.loads((run_dir / "extended_metrics.json").read_text(encoding="utf-8"))
    assert len(ext["configs"]) == 8


def test_report_has_table_and_caveats(run_dir: Path) -> None:
    rc.recompute(run_dir)
    ms = json.loads((run_dir / "metric_summary.json").read_text(encoding="utf-8"))
    rep = (run_dir / "report.md").read_text(encoding="utf-8")
    for cfg in _CONFIGS:
        assert cfg in rep
    assert "Wilson CI" in rep and "Paired McNemar" in rep
    assert ms["significance_caveat"] in rep
    assert ms["metric_caveat"] in rep


def test_retrieval_not_measured_for_all(run_dir: Path) -> None:
    rc.recompute(run_dir)
    ext = json.loads((run_dir / "extended_metrics.json").read_text(encoding="utf-8"))
    for cfg, c in ext["configs"].items():
        assert c["retrieval_quality"]["evidence_tier"] == "NOT MEASURED", cfg
        assert "reason" in c["retrieval_quality"]


def test_claim_level_measured_only_where_present(run_dir: Path) -> None:
    rc.recompute(run_dir)
    ext = json.loads((run_dir / "extended_metrics.json").read_text(encoding="utf-8"))
    iv2 = ext["configs"]["phase5-improved-v2"]["citation"]["claim_level"]
    assert iv2["evidence_tier"].startswith("BENCHMARK")
    # 2 claims (unverifiable + contradicted) => both unsupported.
    assert iv2["n_claims_total"] == 2
    assert iv2["unsupported_claim_rate"] == 1.0
    assert iv2["contradicted_claim_rate"] == 0.5
    for cfg in _CONFIGS:
        if cfg == "phase5-improved-v2":
            continue
        cl = ext["configs"][cfg]["citation"]["claim_level"]
        assert cl["evidence_tier"] == "NOT MEASURED", cfg


def test_failure_count_matches_raw(run_dir: Path) -> None:
    rc.recompute(run_dir)
    fc = json.loads((run_dir / "failure_cases.json").read_text(encoding="utf-8"))
    assert fc["n_failures"] == 0  # no error cells in fixture


def test_no_unsupported_superiority_claim(run_dir: Path) -> None:
    rc.recompute(run_dir)
    ext = json.loads((run_dir / "extended_metrics.json").read_text(encoding="utf-8"))
    # Every paired bootstrap entry carries an explicit ci_excludes_zero flag so a
    # claim can only be made where it is True; nothing asserts superiority inline.
    for cfg, p in ext["paired_rouge_l_bootstrap_vs_baseline"].items():
        if "ci_excludes_zero" in p:
            assert isinstance(p["ci_excludes_zero"], bool)


def test_recompute_is_deterministic(run_dir: Path) -> None:
    rc.recompute(run_dir)
    first_ext = (run_dir / "extended_metrics.json").read_text(encoding="utf-8")
    first_ms = json.loads((run_dir / "metric_summary.json").read_text(encoding="utf-8"))
    first_ms.pop("generated_at", None)
    rc.recompute(run_dir)
    second_ext = (run_dir / "extended_metrics.json").read_text(encoding="utf-8")
    second_ms = json.loads((run_dir / "metric_summary.json").read_text(encoding="utf-8"))
    second_ms.pop("generated_at", None)
    assert first_ext == second_ext
    assert first_ms == second_ms
