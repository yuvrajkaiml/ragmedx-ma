"""
Phase 4a — Agent-Contribution / Critic Analysis (importable, testable module).

This module contains the analysis logic for the Phase 4a critic /
agent-contribution report.  It is a refactor of the logic that previously
lived inline in ``backend/scripts/generate_critic_analysis.py``; that script
is now a thin CLI wrapper that imports :func:`run_analysis` from here.

Design
------
The functions are split into *pure* analysis helpers (no I/O) and a single
:func:`run_analysis` entry point that performs file I/O and writes the
timestamped JSON + Markdown outputs.  The pure helpers are unit-tested in
``backend/tests/test_agent_contribution.py`` on synthetic fixtures so the
tests never touch the real n30 files (one read-only test does, see below).

IMPORTANT DATA LIMITATION
-------------------------
The n30 ablation result files
(``backend/data/results/step4_*_n30_20261008_010119.json``) were produced by
the pipeline's ``IncrementalResultWriter``.  That writer saves, per sample::

    question, true_answer, predicted_answer, response_type,
    evidence_count, evidence_titles, elapsed_ms

It does **NOT** save:

  - ``draft_answer``      (intermediate answer before critic review)
  - ``critic_verdict`` / ``critic_reviews`` (LLM critic output)
  - per-agent latency breakdown (only total ``elapsed_ms`` is recorded)
  - calibration intermediate state

Consequences (honesty):

  - ROUGE-L of draft vs final: NOT MEASURED (draft answer not recorded)
  - Critic accept/reject/rewrite per question: NOT MEASURED (verdict not recorded)
  - Whether a ROUGE-L change was due to wording / abstention flip / correctness:
    NOT MEASURABLE from these files.

What CAN be measured:

  - Abstention rate per config (``response_type == 'abstention'``)
  - Cross-config comparison: full-pipeline vs no-critic vs no-calibration to
    infer the net critic + calibration effect at the **population** level
  - Latency difference (total ``elapsed_ms``) between configs

Evidence tier: SOFTWARE TEST (counts / arithmetic from file parsing).
No AUTOMATED MODEL JUDGE tier is produced here — that tier would require the
critic verdict data, which is absent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths (resolved from this module's location so imports work regardless of CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "backend" / "data" / "results"
OUTPUT_DIR = REPO_ROOT / "backend" / "results"

# Canonical n30 ablation result files (read-only inputs).
N30_FILES = {
    "full-pipeline": "step4_full-pipeline_n30_20261008_010119.json",
    "no-critic": "step4_no-critic_n30_20261008_010119.json",
    "no-calibration": "step4_no-calibration_n30_20261008_010119.json",
    "basic-rag": "step4_basic-rag_n30_20261008_010119.json",
    "no-iterative-retrieval": "step4_no-iterative-retrieval_n30_20261008_010119.json",
    "plain-llm-no-rag": "step4_plain-llm-no-rag_n30_20261008_010119.json",
}

EVIDENCE_TIER = "SOFTWARE TEST"

# Honest "NOT MEASURED" labels — these fields are genuinely absent from the
# n30 files, so we surface that fact explicitly rather than zero-filling.
NOT_MEASURED_CRITIC = "NOT MEASURED: critic_verdict not recorded in output files"
NOT_MEASURED_DRAFT = "NOT MEASURED: draft_answer not recorded in output files"


# ---------------------------------------------------------------------------
# Pure analysis helpers
# ---------------------------------------------------------------------------
def load_result_file(path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a result JSON file. Returns ``None`` if the file is absent."""
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def analyse_single_config(config_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyse a single config's ``per_sample_results``.

    Since draft answers and critic verdicts are NOT recorded, we can only:

      - Count abstentions (``response_type == 'abstention'``)
      - Count answered samples (``response_type == 'verified_answer'``)
      - Record ``elapsed_ms`` statistics (per-sample average)

    Critic accept/reject/rewrite counts are NOT MEASURED.
    """
    samples = data.get("per_sample_results", [])
    abstentions = sum(1 for s in samples if s.get("response_type") == "abstention")
    verified = sum(1 for s in samples if s.get("response_type") == "verified_answer")
    total = len(samples)

    elapsed_list = [
        s.get("elapsed_ms", 0)
        for s in samples
        if s.get("elapsed_ms") is not None
    ]
    avg_elapsed = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0.0

    return {
        "config_name": config_name,
        "total_questions": total,
        "abstention_triggered": abstentions,
        "verified_answers": verified,
        "abstention_rate": round(abstentions / total, 4) if total > 0 else 0.0,
        "avg_elapsed_ms": round(avg_elapsed, 1),
        # Unavailable fields — clearly labelled.
        "critic_accept": NOT_MEASURED_CRITIC,
        "critic_reject": NOT_MEASURED_CRITIC,
        "critic_rewrite": NOT_MEASURED_CRITIC,
        "rouge_l_draft_vs_final": NOT_MEASURED_DRAFT,
        "rouge_l_moved_wording": NOT_MEASURED_DRAFT,
        "rouge_l_moved_abstention": NOT_MEASURED_DRAFT,
        "rouge_l_moved_correctness": NOT_MEASURED_DRAFT,
    }


def cross_config_critic_effect(
    full_data: Dict[str, Any],
    no_critic_data: Dict[str, Any],
    no_calibration_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Infer the net critic + calibration effect at the **population** level by
    comparing full-pipeline vs no-critic vs no-calibration configs.

    This is a population-level inference, NOT a per-question critic verdict.
    Evidence tier: SOFTWARE TEST (counts from file parsing).

    IMPORTANT: This is a cross-config delta, not a controlled experiment.
    Differences may reflect retrieval variation, generation randomness
    (temperature=0 but different question order), or other confounds.
    Inference is weak with n=30 and no statistical significance is claimed.
    """

    def _rates(data: Dict[str, Any]) -> Dict[str, Any]:
        samples = data.get("per_sample_results", [])
        n = len(samples)
        abstentions = sum(1 for s in samples if s.get("response_type") == "abstention")
        metrics = data.get("metrics", {})
        return {
            "n": n,
            "abstention_count": abstentions,
            "abstention_rate": round(abstentions / n, 4) if n > 0 else 0.0,
            "rouge_l_f1": metrics.get("rouge_l_f1", None),
            "avg_elapsed_ms": metrics.get("latency_stats", {}).get("avg_ms", None),
        }

    fp = _rates(full_data)
    nc = _rates(no_critic_data)
    ncal = _rates(no_calibration_data)

    # Net critic effect = full-pipeline abstention rate minus no-critic rate.
    # If the critic causes more abstentions, this is positive.
    critic_abstention_delta = round(fp["abstention_rate"] - nc["abstention_rate"], 4)

    # Net calibration effect = no-critic minus no-calibration.
    calibration_abstention_delta = round(nc["abstention_rate"] - ncal["abstention_rate"], 4)

    rouge_l_delta_critic = None
    if fp.get("rouge_l_f1") is not None and nc.get("rouge_l_f1") is not None:
        rouge_l_delta_critic = round(fp["rouge_l_f1"] - nc["rouge_l_f1"], 4)

    return {
        "evidence_tier": EVIDENCE_TIER,
        "full_pipeline_abstention_rate": fp["abstention_rate"],
        "no_critic_abstention_rate": nc["abstention_rate"],
        "no_calibration_abstention_rate": ncal["abstention_rate"],
        "critic_net_abstention_delta": critic_abstention_delta,
        "calibration_net_abstention_delta": calibration_abstention_delta,
        "full_pipeline_rouge_l_f1": fp.get("rouge_l_f1"),
        "no_critic_rouge_l_f1": nc.get("rouge_l_f1"),
        "rouge_l_delta_full_vs_no_critic": rouge_l_delta_critic,
        "caveat": (
            "Population-level delta across independent config runs with n=30. "
            "Not a controlled per-question comparison. "
            "Does not isolate critic effect from retrieval/generation variance. "
            "Statistical significance NOT assessed (n too small for reliable inference). "
            "ROUGE-L delta != correctness delta."
        ),
    }


def build_concrete_examples(
    full_data: Dict[str, Any],
    no_critic_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build up to 3 concrete cross-config examples by matching question text.

    Since draft answers are not recorded, examples show:

      - Whether the same question was abstained or answered in full-pipeline
        vs no-critic
      - The final answer text in each config

    These are NOT critic accept/reject examples (that data is unavailable).
    They illustrate the net pipeline effect (critic + calibration combined).

    Evidence tier: SOFTWARE TEST (direct file comparison).
    """
    full_by_q = {s["question"]: s for s in full_data.get("per_sample_results", [])}
    nc_by_q = {s["question"]: s for s in no_critic_data.get("per_sample_results", [])}

    def _excerpt(sample: Dict[str, Any]) -> str:
        answer = sample.get("predicted_answer", "")
        if answer:
            return answer[:200] + "..."
        return "[abstained]"

    examples: List[Dict[str, Any]] = []
    # Interesting first: one config abstained, the other answered.
    for q, fp_sample in full_by_q.items():
        if q not in nc_by_q:
            continue
        nc_sample = nc_by_q[q]
        fp_type = fp_sample.get("response_type")
        nc_type = nc_sample.get("response_type")
        if fp_type != nc_type and len(examples) < 3:
            examples.append(
                {
                    "question_excerpt": q[:120] + ("..." if len(q) > 120 else ""),
                    "full_pipeline_response_type": fp_type,
                    "no_critic_response_type": nc_type,
                    "full_pipeline_answer_excerpt": _excerpt(fp_sample),
                    "no_critic_answer_excerpt": _excerpt(nc_sample),
                    "elapsed_ms_full_pipeline": fp_sample.get("elapsed_ms"),
                    "elapsed_ms_no_critic": nc_sample.get("elapsed_ms"),
                    "note": (
                        "This example shows a RESPONSE TYPE CHANGE (abstention vs "
                        "answer) between full-pipeline and no-critic configs. The "
                        "cause (critic verdict, calibration, or retrieval variance) "
                        "cannot be determined from the saved output files. "
                        "Evidence tier: SOFTWARE TEST."
                    ),
                }
            )

    # Top up with identical-type examples if fewer than 3 type-change ones.
    if len(examples) < 3:
        for q, fp_sample in list(full_by_q.items())[:10]:
            if q in nc_by_q and len(examples) < 3:
                if any(e["question_excerpt"].startswith(q[:50]) for e in examples):
                    continue
                nc_sample = nc_by_q[q]
                examples.append(
                    {
                        "question_excerpt": q[:120] + ("..." if len(q) > 120 else ""),
                        "full_pipeline_response_type": fp_sample.get("response_type"),
                        "no_critic_response_type": nc_sample.get("response_type"),
                        "full_pipeline_answer_excerpt": _excerpt(fp_sample),
                        "no_critic_answer_excerpt": _excerpt(nc_sample),
                        "elapsed_ms_full_pipeline": fp_sample.get("elapsed_ms"),
                        "elapsed_ms_no_critic": nc_sample.get("elapsed_ms"),
                        "note": (
                            "Response type identical between configs for this "
                            "question. Evidence tier: SOFTWARE TEST."
                        ),
                    }
                )

    return examples[:3]


def generate_caveats() -> List[str]:
    """Return the honest limitation caveats for the report."""
    return [
        (
            "CRITICAL LIMITATION: The n30 ablation result files do not record "
            "draft_answer, critic_verdict, or critic_reviews. Per-question critic "
            "accept/reject/rewrite counts are therefore NOT MEASURED. What data "
            "would be needed: IncrementalResultWriter must be extended to save "
            "draft_answer (pre-critic), critic_verdict dict, and critic_round_count "
            "per sample."
        ),
        (
            "ROUGE-L of draft vs final is NOT MEASURED for the same reason. Even if "
            "it were measured, ROUGE-L change != correctness change. A rewrite can "
            "lower ROUGE-L (more concise) while being more factually correct, or "
            "raise ROUGE-L (more verbose) while still being wrong. ROUGE-L is a "
            "surface-similarity metric, not a safety or correctness metric."
        ),
        (
            "Cross-config abstention delta (full-pipeline vs no-critic) is a "
            "population-level inference with n=30. It does NOT isolate critic effect "
            "from calibration, retrieval variance, or generation variance. With n=30 "
            "the delta is not statistically reliable. A McNemar test cannot be applied "
            "because per-question critic decision data is absent."
        ),
        (
            "Evidence tier for abstention counts: SOFTWARE TEST (direct file parsing). "
            "Evidence tier for cross-config delta inference: SOFTWARE TEST (arithmetic "
            "on parsed counts — no model judgment involved). No AUTOMATED MODEL JUDGE "
            "tier evidence was generated in this analysis because the critic verdict "
            "data needed for that tier is absent."
        ),
        (
            "The no-critic config ran with disable_critic=True (inferred from config "
            "name and abstention rate difference). The no-calibration config ran with "
            "disable_calibration=True. These flags were set in run_step4_matrix.py but "
            "are not explicitly recorded in the output JSON run_metadata."
        ),
    ]


def build_report(
    loaded: Dict[str, Dict[str, Any]],
    missing: List[str],
) -> Dict[str, Any]:
    """
    Build the full machine-readable report dict from already-loaded configs.

    Pure function (no I/O).  ``loaded`` maps config name -> parsed result dict;
    ``missing`` lists filenames that could not be loaded.
    """
    per_config: Dict[str, Any] = {}
    total_questions_across_configs = 0
    for config_name, data in loaded.items():
        result = analyse_single_config(config_name, data)
        per_config[config_name] = result
        total_questions_across_configs += result["total_questions"]

    cross_config: Optional[Dict[str, Any]] = None
    if all(k in loaded for k in ("full-pipeline", "no-critic", "no-calibration")):
        cross_config = cross_config_critic_effect(
            loaded["full-pipeline"],
            loaded["no-critic"],
            loaded["no-calibration"],
        )

    examples: List[Dict[str, Any]] = []
    if "full-pipeline" in loaded and "no-critic" in loaded:
        examples = build_concrete_examples(loaded["full-pipeline"], loaded["no-critic"])

    caveats = generate_caveats()
    fp_stats = per_config.get("full-pipeline", {})

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_tier": EVIDENCE_TIER,
        "total_questions": total_questions_across_configs,
        "configs_analysed": list(per_config.keys()),
        "missing_files": missing,
        # Top-level fields (full-pipeline config values where applicable).
        "critic_accept": NOT_MEASURED_CRITIC,
        "critic_reject": NOT_MEASURED_CRITIC,
        "critic_rewrite": NOT_MEASURED_CRITIC,
        "abstention_triggered": fp_stats.get("abstention_triggered", "N/A"),
        "rouge_l_moved_wording": NOT_MEASURED_DRAFT,
        "rouge_l_moved_abstention": NOT_MEASURED_DRAFT,
        "rouge_l_moved_correctness": NOT_MEASURED_DRAFT,
        "per_config_analysis": per_config,
        "cross_config_critic_effect": cross_config,
        "examples": examples,
        "caveats": caveats,
        "what_data_is_needed": (
            "To enable per-question critic analysis, IncrementalResultWriter in "
            "backend/scripts/run_step4_matrix.py must be extended to save: "
            "(1) draft_answer (answer before first critic review), "
            "(2) critic_verdict dict (passed, factual_consistency, citation_support, "
            "hallucination_risk, revision_instructions, verdict), "
            "(3) critic_round_count, "
            "(4) per-agent duration_ms from the agent_trace."
        ),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the human-readable Markdown report from a report dict (pure)."""
    per_config = report.get("per_config_analysis", {})
    cross_config = report.get("cross_config_critic_effect")
    examples = report.get("examples", [])
    caveats = report.get("caveats", [])

    md_lines = [
        "# Critic Analysis Report",
        "",
        f"**Generated at:** {report['generated_at']}  ",
        f"**Evidence tier:** {report['evidence_tier']}  ",
        f"**Configs analysed:** {', '.join(report['configs_analysed'])}  ",
        f"**Total questions across configs:** {report['total_questions']}  ",
        "",
        "---",
        "",
        "## Critical Data Limitation",
        "",
        "The n30 ablation result files (`backend/data/results/step4_*_n30_*.json`) "
        "record **only**:",
        "",
        "| Field | Present? |",
        "|-------|----------|",
        "| `question` | yes |",
        "| `true_answer` | yes |",
        "| `predicted_answer` (final) | yes |",
        "| `response_type` (verified_answer / abstention) | yes |",
        "| `evidence_count` | yes |",
        "| `evidence_titles` | yes |",
        "| `elapsed_ms` (total) | yes |",
        "| `draft_answer` (pre-critic) | no — NOT RECORDED |",
        "| `critic_verdict` (accept/reject/rewrite) | no — NOT RECORDED |",
        "| `critic_round_count` | no — NOT RECORDED |",
        "| per-agent latency breakdown | no — NOT RECORDED |",
        "",
        "**Consequence:** Per-question critic accept/reject/rewrite analysis and "
        "ROUGE-L draft-vs-final comparison are **NOT MEASURABLE** from these files.",
        "",
        "---",
        "",
        "## Per-Config Abstention Analysis",
        "",
        "Evidence tier: SOFTWARE TEST (direct file parsing)",
        "",
        "| Config | n | Abstentions | Answered | Abstention Rate | Avg Elapsed (ms) |",
        "|--------|---|-------------|----------|-----------------|-----------------|",
    ]

    for config_name, stats in per_config.items():
        md_lines.append(
            f"| {config_name} | {stats['total_questions']} | "
            f"{stats['abstention_triggered']} | {stats['verified_answers']} | "
            f"{stats['abstention_rate']:.4f} | {stats['avg_elapsed_ms']:.0f} |"
        )

    md_lines += [
        "",
        "---",
        "",
        "## Cross-Config Critic + Calibration Effect (Population Level)",
        "",
        "Evidence tier: SOFTWARE TEST",
        "",
    ]

    if cross_config:
        md_lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| full-pipeline abstention rate | {cross_config['full_pipeline_abstention_rate']} |",
            f"| no-critic abstention rate | {cross_config['no_critic_abstention_rate']} |",
            f"| no-calibration abstention rate | {cross_config['no_calibration_abstention_rate']} |",
            f"| Critic net abstention delta (full - no-critic) | {cross_config['critic_net_abstention_delta']} |",
            f"| Calibration net abstention delta (no-critic - no-calibration) | {cross_config['calibration_net_abstention_delta']} |",
            f"| full-pipeline ROUGE-L F1 | {cross_config['full_pipeline_rouge_l_f1']} |",
            f"| no-critic ROUGE-L F1 | {cross_config['no_critic_rouge_l_f1']} |",
            f"| ROUGE-L delta (full - no-critic) | {cross_config['rouge_l_delta_full_vs_no_critic']} |",
            "",
            f"**Caveat:** {cross_config['caveat']}",
        ]
    else:
        md_lines.append("_Cross-config comparison not available (missing result files)._")

    md_lines += [
        "",
        "---",
        "",
        "## Concrete Examples (Cross-Config Comparison)",
        "",
        "Evidence tier: SOFTWARE TEST",
        "",
        "The following examples compare the final output of **full-pipeline** vs "
        "**no-critic** configs for the same question. These are NOT critic "
        "accept/reject examples; they show the net pipeline effect (critic + "
        "calibration combined).",
        "",
    ]

    for i, ex in enumerate(examples, 1):
        md_lines += [
            f"### Example {i}",
            "",
            f"**Question (excerpt):** {ex['question_excerpt']}",
            "",
            "| | full-pipeline | no-critic |",
            "|---|---|---|",
            f"| response_type | {ex['full_pipeline_response_type']} | {ex['no_critic_response_type']} |",
            f"| answer (excerpt) | {ex['full_pipeline_answer_excerpt'][:120]}... | "
            f"{ex['no_critic_answer_excerpt'][:120]}... |",
            f"| elapsed_ms | {ex['elapsed_ms_full_pipeline']} | {ex['elapsed_ms_no_critic']} |",
            "",
            f"**Note:** {ex['note']}",
            "",
        ]

    md_lines += [
        "---",
        "",
        "## What Data Is Needed for Full Critic Analysis",
        "",
        f"{report['what_data_is_needed']}",
        "",
        "---",
        "",
        "## Caveats",
        "",
    ]

    for i, caveat in enumerate(caveats, 1):
        md_lines.append(f"{i}. {caveat}")
        md_lines.append("")

    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# I/O entry point
# ---------------------------------------------------------------------------
def run_analysis(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load all 6 canonical n30 result files, run the analysis, and write the
    timestamped JSON + Markdown outputs.

    Parameters
    ----------
    results_dir
        Directory holding the n30 input files (defaults to the canonical
        ``backend/data/results``).
    output_dir
        Directory to write outputs into (defaults to ``backend/results``).

    Returns
    -------
    dict
        The full machine-readable report dict that was written to JSON.
    """
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all 6 result files ---
    loaded: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for config_name, filename in N30_FILES.items():
        path = results_dir / filename
        data = load_result_file(path)
        if data is None:
            missing.append(filename)
        else:
            loaded[config_name] = data

    print(f"Loaded {len(loaded)}/6 result files. Missing: {missing or 'none'}")

    # --- Build the report ---
    report = build_report(loaded, missing)

    # --- Write outputs to unique timestamped paths (never overwrite existing) ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_out = output_dir / f"critic_analysis_{ts}.json"
    md_out = output_dir / f"critic_analysis_{ts}.md"

    # Atomic write: .tmp + os.replace.
    tmp_json = json_out.with_suffix(json_out.suffix + ".tmp")
    with open(tmp_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    os.replace(tmp_json, json_out)
    print(f"JSON written: {json_out}")

    tmp_md = md_out.with_suffix(md_out.suffix + ".tmp")
    with open(tmp_md, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    os.replace(tmp_md, md_out)
    print(f"Markdown written: {md_out}")

    # --- Console summary ---
    print("\nSummary:")
    for config_name, stats in report["per_config_analysis"].items():
        print(
            f"  {config_name}: {stats['total_questions']} questions, "
            f"{stats['abstention_triggered']} abstentions "
            f"({stats['abstention_rate']:.1%})"
        )

    return report


def main() -> None:
    """CLI entry point (thin wrapper around :func:`run_analysis`)."""
    run_analysis()


if __name__ == "__main__":
    main()
