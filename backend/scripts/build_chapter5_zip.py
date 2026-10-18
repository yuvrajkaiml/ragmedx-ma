#!/usr/bin/env python
"""
build_chapter5_zip.py — Package the ACTUAL RAGMedX-MA Chapter 5 evaluation
outputs into RAGMedXMA_Chapter5_Results.zip.

STRICT RULES (academic integrity):
  * Reads only artifacts that already exist on disk. Computes nothing new
    except trivial reshaping (JSONL -> CSV) and counts from real cells.
  * Never fabricates or estimates a missing value. Absent values are written
    verbatim as "NOT MEASURED" (carrying the reason from the source file where
    one exists) or "NOT RUN".
  * No secrets: .env / *.db / tokens / keys are never read or copied.

Source of truth:
  backend/results/phase5_20261018_053140/   (completed 40 x 8 real-Qwen3-8B run)
  .agents/tasks/ragmed2-pipeline-smoke-2026-10-18/   (pipeline smoke test)

Run (CPU venv is fine — no GPU, no model load):
  cd d:\\Ragmed_2\\backend
  d:\\Ragmed_2\\.venv\\Scripts\\python.exe scripts\\build_chapter5_zip.py
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root = parent of backend/)
# ---------------------------------------------------------------------------
BACKEND = Path(__file__).resolve().parent.parent          # d:\Ragmed_2\backend
REPO = BACKEND.parent                                      # d:\Ragmed_2
RUN_DIR = BACKEND / "results" / "phase5_20261018_053140"
PERQ_DIR = RUN_DIR / "per_question"
SMOKE_DIR = REPO / ".agents" / "tasks" / "ragmed2-pipeline-smoke-2026-10-18"

STAGE = BACKEND / "results" / "chapter5_package"          # staging dir
ZIP_PATH = BACKEND / "results" / "RAGMedXMA_Chapter5_Results.zip"

CONFIGS = [
    "phase5-basic-rag",
    "phase5-full-pipeline",
    "phase5-improved-no-verification",
    "phase5-improved-v2",
    "phase5-no-calibration",
    "phase5-no-critic",
    "phase5-no-iterative-retrieval",
    "phase5-plain-llm-no-rag",
]

NM = "NOT MEASURED"
NR = "NOT RUN"


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_cells(config: str):
    """Yield each per-question cell dict for a config (real JSONL rows)."""
    fp = PERQ_DIR / f"{config}.jsonl"
    if not fp.exists():
        return
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _answer_text(cell: dict) -> str:
    """plain-llm cells store answer as [text, usage_dict]; RAG cells store a str."""
    a = cell.get("answer", "")
    if isinstance(a, list):
        return str(a[0]) if a else ""
    return str(a or "")


# ---------------------------------------------------------------------------
# 1. metrics_summary.csv  — one row per (experiment, config)
# ---------------------------------------------------------------------------
def build_metrics_summary(out: Path):
    metric = _load_json(RUN_DIR / "metric_summary.json")
    ext = _load_json(RUN_DIR / "extended_metrics.json")
    env = _load_json(RUN_DIR / "environment_run.json")
    lat = _load_json(RUN_DIR / "latency_breakdown.json")

    rows = []
    for cfg in CONFIGS:
        ms = metric["configs"][cfg]
        em = ext["configs"][cfg]
        latc = lat["configs"][cfg]
        rl = em["rouge_l"]
        cite = em["citation"]
        rows.append({
            "experiment": "phase5_40x8",
            "config": cfg,
            "dataset": "eval_split (BioASQ+PubMedQA, test split)",
            "n": ms["n_total"],
            "n_answered": em["n_answered"],
            "n_abstained": em["n_abstained"],
            "n_error": em["n_error"],
            "accuracy_token_containment": ms["accuracy"],
            "accuracy_wilson_ci95_low": ms["accuracy_wilson_ci_95"][0],
            "accuracy_wilson_ci95_high": ms["accuracy_wilson_ci_95"][1],
            "rouge_l_f1": rl["mean_rouge_l_f1"],
            "abstention_rate": ms["abstention_rate"],
            "citation_support_rate_all": cite["citation_support_rate_all"],
            "answer_with_citation_rate_non_abstained": cite["answer_with_citation_rate_non_abstained"],
            "retrieval_quality": "NOT MEASURED (0/789 BioASQ gold PMIDs in eval_split index)",
            "ece": NM + " (not computed on this run)",
            "brier": NM + " (not computed on this run)",
            "latency_cold_mean_ms": latc["cold_latency_ms"]["mean"],
            "latency_cold_min_ms": latc["cold_latency_ms"]["min"],
            "latency_cold_max_ms": latc["cold_latency_ms"]["max"],
            "peak_vram_mb": em["peak_vram_mb"]["value"],
            "llm_model": env["llm_model"],
            "llm_mode": env["llm_mode"],
            "generator_temperature": env["generator_temperature"],
        })

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# 2. predictions.jsonl + predictions.csv  — every real per-question cell
# ---------------------------------------------------------------------------
def build_predictions(out_jsonl: Path, out_csv: Path):
    all_rows = []
    with out_jsonl.open("w", encoding="utf-8") as fj:
        for cfg in CONFIGS:
            for cell in _iter_cells(cfg):
                cv = cell.get("claim_verdicts", {})
                verdict = cv.get("critic_verdict") if isinstance(cv, dict) else None
                citations = cell.get("citations", []) or []
                flat = {
                    "config": cfg,
                    "question_id": cell.get("question_id"),
                    "source": cell.get("source"),
                    "split": cell.get("split"),
                    "gold_answer": cell.get("gold_answer", ""),
                    "predicted_answer": _answer_text(cell),
                    "status": cell.get("status"),
                    "abstain_flag": cell.get("abstain_flag"),
                    "confidence": cell.get("confidence"),
                    "n_citations": len(citations),
                    "evidence_doc_ids": "|".join(
                        str(c.get("document_id", "")) for c in citations
                    ),
                    "citation_markers": "|".join(
                        str(c.get("citation_marker", "")) for c in citations
                    ),
                    "critic_passed": (verdict or {}).get("passed") if verdict else NM,
                    "citation_support": (verdict or {}).get("citation_support") if verdict else NM,
                    "hallucination_risk": (verdict or {}).get("hallucination_risk") if verdict else NM,
                    "error_code": cell.get("error_code"),
                    "cold_latency_ms": cell.get("cold_latency_ms"),
                    "model_call_count": cell.get("model_call_count"),
                    "retrieval_call_count": cell.get("retrieval_call_count"),
                }
                # correctness label is NOT stored per-cell in raw data; mark honestly.
                flat["correctness_label"] = (
                    "NOT STORED PER-CELL (aggregate token-containment acc in metrics_summary.csv)"
                )
                fj.write(json.dumps(cell, ensure_ascii=False) + "\n")
                all_rows.append(flat)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    return len(all_rows)


# ---------------------------------------------------------------------------
# 3. baseline_results.csv + ablation_results.csv  (subsets of the 8 configs)
# ---------------------------------------------------------------------------
BASELINE_CONFIGS = {
    "phase5-plain-llm-no-rag": "Chain-of-Thought / no-retrieval LLM baseline",
    "phase5-basic-rag": "Vanilla RAG baseline",
    "phase5-full-pipeline": "RAGMedX-MA full pipeline (reference)",
}
ABLATION_CONFIGS = {
    "phase5-full-pipeline": "Full pipeline (reference)",
    "phase5-no-critic": "- Critic Agent",
    "phase5-no-iterative-retrieval": "- Iterative retrieval (max_retrieval_iterations=1)",
    "phase5-no-calibration": "- Confidence calibration",
    "phase5-improved-no-verification": "- Evidence-quality/claim verification",
}


def _subset_rows(mapping: dict):
    metric = _load_json(RUN_DIR / "metric_summary.json")
    ext = _load_json(RUN_DIR / "extended_metrics.json")
    paired = metric["paired_vs_baseline_mcnemar_holm"]
    rows = []
    for cfg, label in mapping.items():
        ms = metric["configs"][cfg]
        em = ext["configs"][cfg]
        p = paired.get(cfg, {})
        rows.append({
            "config": cfg,
            "role": label,
            "n": ms["n_total"],
            "accuracy_token_containment": ms["accuracy"],
            "wilson_ci95": f'[{ms["accuracy_wilson_ci_95"][0]}, {ms["accuracy_wilson_ci_95"][1]}]',
            "rouge_l_f1": em["rouge_l"]["mean_rouge_l_f1"],
            "abstention_rate": ms["abstention_rate"],
            "mcnemar_p_holm_vs_full_pipeline": p.get("p_value_holm_adjusted", "baseline (self)"),
            "significant_holm_0.05": p.get("significant_holm_0.05", "baseline (self)"),
        })
    return rows


def build_subset_csv(mapping: dict, out: Path):
    rows = _subset_rows(mapping)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# experiment_report.md  (provenance, procedure, hardware, limitations)
# ---------------------------------------------------------------------------
def build_experiment_report(out: Path):
    env = _load_json(RUN_DIR / "environment_run.json")
    report_src = (RUN_DIR / "report.md").read_text(encoding="utf-8")
    repro_src = (RUN_DIR / "reproducibility_report.md").read_text(encoding="utf-8") \
        if (RUN_DIR / "reproducibility_report.md").exists() else "(reproducibility_report.md absent)"

    md = f"""# RAGMedX-MA — Chapter 5 Experiment Report

Packaged: {datetime.now(timezone.utc).isoformat()}

## Provenance
- Primary experiment: 40 unique questions x 8 configs = 320 cells.
- Run directory: `backend/results/phase5_20261018_053140/`
- Pipeline smoke test (functional, NOT accuracy): `.agents/tasks/ragmed2-pipeline-smoke-2026-10-18/`
- These are two DISTINCT provenances; the smoke test uses mock LLM + synthetic
  evidence (cases 01-09) and one real-model structural case (case10). It is NOT
  a benchmark accuracy result and must never be merged with the 40x8 numbers.

## Model, data, hardware (verbatim from environment_run.json)
- LLM: {env['llm_model']} (mode={env['llm_mode']}, generator_temperature={env['generator_temperature']})
- torch {env['torch_version']} / CUDA {env['cuda_version']} / cuda_available={env['cuda_available']}
- Peak VRAM on run: {env['peak_vram_mb_run']} MB (8 GB RTX 5060 — offload/paging active)
- Eval index: {env['resolved_eval_split_path']}
- Matrix: total={env['matrix']['plan']['total']}, reused={env['matrix']['plan']['reused']}, newly run={env['matrix']['plan']['missing']}; failures={env['matrix']['run_counts']['failed']}
- Run wall-clock: {env['matrix']['run_counts']['run_seconds']} s

## Evaluation procedure
- Entry point: `backend/scripts/run_phase5_experiment.py` (8-config matrix runner).
- Per-sample pipeline: `backend/evaluation/mirage_runner.py::run_rag_evaluation`
  over the real RAG pipeline (`app/rag/pipeline.py`: BGE-M3 + FAISS + BM25 + RRF +
  cross-encoder rerank) with the multi-agent orchestrator (`app/agents/orchestrator.py`).
- Metrics: `backend/evaluation/metrics.py`; statistics (Wilson CI, McNemar,
  paired bootstrap, Holm): `backend/evaluation/statistics.py`.
- Leakage guard: `backend/evaluation/split_builder.py::IndexLeakageGuard`
  (production index `backend/data/faiss_index/` is PROTECTED; only eval_split used).

## Commands executed (verbatim)
Completed 40x8 run (GPU venv):
```
d:\\Ragmed_2\\.venv-cu128-pilot\\Scripts\\python.exe backend\\scripts\\run_phase5_experiment.py \\
    --output-root backend\\results\\phase5_20261018_053140
```
Pipeline smoke suite (CPU venv):
```
cd d:\\Ragmed_2\\backend
d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m pytest tests/test_pipeline_smoke_trace.py -q
```
This package (CPU venv, no GPU):
```
d:\\Ragmed_2\\.venv\\Scripts\\python.exe backend\\scripts\\build_chapter5_zip.py
```

## What is MEASURED vs NOT MEASURED / NOT RUN
MEASURED: token-containment accuracy (threshold 0.6 vs gold), ROUGE-L F1,
abstention rate, citation-support rate, cold end-to-end latency, peak VRAM,
Wilson CIs, Holm-corrected paired McNemar, paired bootstrap ROUGE-L deltas,
claim-level verdicts where the critic produced them, per-stage latency for the
two 'improved' configs only.

NOT MEASURED (reason recorded inline in the source JSONs):
- Retrieval quality (P@5/R@5/MRR/nDCG@5): 0/789 BioASQ gold PMIDs are present in
  the eval_split index universe, so retrieval-vs-gold cannot be scored.
- ECE / Brier: calibration metrics were not computed on this run (code exists in
  metrics.py; a follow-up recompute is required — not included here).
- Per-stage latency for 6 of 8 configs: record_node_timing was OFF.
- False-acceptance / answerability: split has no gold should-abstain labels.
- Warm latency for most configs.

NOT RUN:
- Full 50x8 / 500-question sweep (exceeds the 8 h GPU budget).
- Concurrency / 50-user load test.
- MedRAG / MedTrust-RAG baselines (no comparable implementation reproduced).
- MIRAGE / MedRGB / MedQA / MedMCQA / MMLU-Med accuracy runs.

## Known defect (verified, not fixed in this package)
- Citation extraction: `generator.py` CITATION_PATTERN `\\[([A-Za-z0-9_-]+)\\]`
  does not match markers containing a space (e.g. `[Doc E1]`), so answer_citations
  can be empty even when the answer cites evidence. Flagged for a separate labeled fix.

## Limitations (carried from research_results_summary.md)
- Small n (40/config): proportion CIs are wide; treat deltas as directional.
- Accuracy/ROUGE-L score abstentions as 0, penalising high-abstention RAG configs;
  always read answer quality ALONGSIDE abstention rate.
- No superiority/clinical claim without a Holm-significant paired test. Only
  plain-LLM-no-RAG is significantly different from the full pipeline on this mix.
- Single eval set, single seed (42).

---

## Appendix A — report.md (verbatim from the run)

{report_src}

---

## Appendix B — reproducibility_report.md (verbatim from the run)

{repro_src}
"""
    out.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------
def build_readme(out: Path, counts: dict):
    md = f"""# RAGMedXMA_Chapter5_Results — README

Generated: {datetime.now(timezone.utc).isoformat()}

This package contains the ACTUAL evaluation outputs for RAGMedX-MA, extracted
from on-disk artifacts. No value is fabricated or estimated. Missing values are
written as `NOT MEASURED` (with reason) or `NOT RUN`.

## File map

| File | What it is | Experiment |
|---|---|---|
| `metrics_summary.csv` | All measured metrics per config ({counts['metrics']} rows) | phase5_40x8 |
| `predictions.jsonl` | Raw per-question cells, verbatim ({counts['pred']} cells) | phase5_40x8 |
| `predictions.csv` | Flattened per-question predictions ({counts['pred']} rows) | phase5_40x8 |
| `baseline_results.csv` | Vanilla-RAG / CoT-no-RAG / full-pipeline comparison | phase5_40x8 |
| `ablation_results.csv` | Full vs -critic / -iterative / -calibration / -verification | phase5_40x8 |
| `experiment_report.md` | Commands, versions, procedure, hardware, limitations | both |
| `scripts/` | Evaluation + metric + stats source used to produce the numbers | — |
| `test_logs/` | Raw run log + smoke test results + run reports | both |
| `raw_run_artifacts/` | Verbatim metric/latency/summary JSON from the 40x8 run | phase5_40x8 |
| `screenshots/` | SEE screenshots/README.txt — none captured in this run | — |

## The eight configurations
- `phase5-plain-llm-no-rag` — Chain-of-Thought / no-retrieval baseline
- `phase5-basic-rag` — Vanilla RAG baseline
- `phase5-full-pipeline` — RAGMedX-MA full pipeline (reference for paired tests)
- `phase5-no-critic`, `phase5-no-iterative-retrieval`, `phase5-no-calibration`,
  `phase5-improved-no-verification` — ablations
- `phase5-improved-v2` — claim-verification + cache ON variant

## Honest headline (read before citing any number)
On this BioASQ/PubMedQA mix, RAG configs abstain 87-95% and score ~0 token
accuracy; plain-LLM-no-RAG answers all and scores 0.375. Only plain-LLM differs
significantly from the full pipeline (Holm p=0.000427). This is NOT evidence that
abstention is wrong — it is a measurement property. Report accuracy ALONGSIDE
abstention_rate. Do NOT claim RAGMedX-MA outperforms the baselines on this data.

## Explicitly NOT RUN / NOT MEASURED
- Retrieval quality (P@5/R@5/MRR/nDCG): NOT MEASURED — 0/789 BioASQ gold PMIDs in index.
- ECE / Brier: NOT MEASURED on this run (code exists; follow-up recompute needed).
- Full 50x8 / 500-Q sweep, concurrency/load, MedRAG/MedTrust-RAG, MIRAGE/MedRGB/MedQA: NOT RUN.
- Screenshots of the running UI: NOT CAPTURED (see screenshots/README.txt).

## Reproduce
See `experiment_report.md` for exact commands. The CSVs here are regenerated by
`scripts/build_chapter5_zip.py` from the raw cells in `raw_run_artifacts/` and the
original run's `per_question/*.jsonl`.

## Secrets
No `.env`, database, token, or key files are included. Verified by the packager's
allowlist (it copies only named source scripts and result JSON/MD/log files).
"""
    out.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    if not RUN_DIR.exists():
        raise SystemExit(f"Run dir not found: {RUN_DIR}")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    (STAGE / "scripts").mkdir()
    (STAGE / "test_logs").mkdir()
    (STAGE / "screenshots").mkdir()
    (STAGE / "raw_run_artifacts").mkdir()

    counts = {}
    counts["metrics"] = build_metrics_summary(STAGE / "metrics_summary.csv")
    counts["pred"] = build_predictions(STAGE / "predictions.jsonl", STAGE / "predictions.csv")
    build_subset_csv(BASELINE_CONFIGS, STAGE / "baseline_results.csv")
    build_subset_csv(ABLATION_CONFIGS, STAGE / "ablation_results.csv")
    build_experiment_report(STAGE / "experiment_report.md")
    build_readme(STAGE / "README.md", counts)

    # Evaluation source scripts (allowlist — no secrets)
    eval_scripts = [
        BACKEND / "evaluation" / "metrics.py",
        BACKEND / "evaluation" / "statistics.py",
        BACKEND / "evaluation" / "mirage_runner.py",
        BACKEND / "evaluation" / "split_builder.py",
        BACKEND / "evaluation" / "plain_llm_runner.py",
        BACKEND / "evaluation" / "ablation_studies.py",
        BACKEND / "evaluation" / "index_builder.py",
        BACKEND / "scripts" / "build_chapter5_zip.py",
    ]
    for s in eval_scripts:
        if s.exists():
            shutil.copy2(s, STAGE / "scripts" / s.name)

    # Raw run artifacts (JSON/MD only — no secrets)
    raw_files = [
        "metric_summary.json", "extended_metrics.json", "latency_breakdown.json",
        "summary.json", "failure_cases.json", "environment_run.json",
        "comparison_table.md", "report.md", "research_results_summary.md",
        "reproducibility_report.md", "runtime_completion_report.md",
    ]
    for name in raw_files:
        p = RUN_DIR / name
        if p.exists():
            shutil.copy2(p, STAGE / "raw_run_artifacts" / name)

    # test_logs
    run_log = RUN_DIR / "run_40x8.log"
    if run_log.exists():
        shutil.copy2(run_log, STAGE / "test_logs" / "run_40x8.log")
    for name in ["report.md", "results.json", "smoke-notes.md",
                 "smoke-review.md", "smoke-review.json"]:
        p = SMOKE_DIR / name
        if p.exists():
            shutil.copy2(p, STAGE / "test_logs" / f"smoke_{name}")

    # screenshots placeholder — honest about absence
    (STAGE / "screenshots" / "README.txt").write_text(
        "NOT CAPTURED.\n\n"
        "No UI/functional screenshots were captured during this evaluation run.\n"
        "The 40x8 experiment was a headless batch job; the FastAPI app and React\n"
        "frontend exist in the repo (backend/app/api, frontend/) but were not\n"
        "exercised with captured screenshots. To supply them, run the server +\n"
        "frontend and capture: login, query+answer, document upload, abstention,\n"
        "and conversation history. Do NOT fabricate these images.\n",
        encoding="utf-8",
    )

    # Zip
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(STAGE))

    print(f"metrics_summary rows : {counts['metrics']}")
    print(f"prediction cells     : {counts['pred']}")
    print(f"staging dir          : {STAGE}")
    print(f"ZIP written          : {ZIP_PATH}  ({ZIP_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
