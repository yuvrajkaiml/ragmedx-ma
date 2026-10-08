"""Compute ablation metrics from phase-5 result files and write summary report."""
import json
import re
import statistics
import sys
from pathlib import Path
from datetime import datetime

# Use the project's own ROUGE-L implementation (no external dependencies)
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import compute_rouge_l

BASE = Path(__file__).parent.parent
RESULTS = BASE / "data" / "results"

FILES = {
    "full-pipeline":          RESULTS / "step4_full-pipeline_n30_20261008_010119.json",
    "no-critic":              RESULTS / "step4_no-critic_n30_20261008_010119.json",
    "no-calibration":         RESULTS / "step4_no-calibration_n30_20261008_010119.json",
    "no-iterative-retrieval": RESULTS / "step4_no-iterative-retrieval_n30_20261008_010119.json",
    "basic-rag":              RESULTS / "step4_basic-rag_n30_20261008_010119.json",
    "plain-llm-no-rag":       RESULTS / "step4_plain-llm-no-rag_n30_20261008_010119.json",
}

CITATION_RE = re.compile(r"\[E\d+\]|\[Doc \d+\]|\[doc\d+\]", re.IGNORECASE)


def get_text(pa):
    """Extract text from predicted_answer (may be str or [str, meta])."""
    if isinstance(pa, list):
        return pa[0] if pa else ""
    return pa or ""


def is_abstained(sample):
    rt = sample.get("response_type", "")
    if rt in ("abstention", "abstained"):
        return True
    return get_text(sample.get("predicted_answer", "")).strip() == ""


def compute_metrics(config, data):
    samples = data["per_sample_results"]
    n = len(samples)

    abstained_samples = [s for s in samples if is_abstained(s)]
    answered_samples = [s for s in samples if not is_abstained(s)]
    n_abs = len(abstained_samples)
    n_ans = len(answered_samples)
    abstention_rate = n_abs / n

    # --- ROUGE-L from stored metrics block ---
    metrics = data.get("metrics", {})
    rouge_all = metrics.get("rouge_l_f1", None)

    # For plain-llm, compute ROUGE from scratch using project implementation
    if rouge_all is None:
        scores = []
        for s in samples:
            pa = get_text(s.get("predicted_answer", ""))
            ta = s.get("true_answer", "")
            scores.append(compute_rouge_l(pa, ta)["rouge_l_f1"])
        rouge_all = round(statistics.mean(scores), 4) if scores else 0.0

    # --- ROUGE-L non-abstained only (using project ROUGE-L) ---
    scores_ans = []
    for s in answered_samples:
        pa = get_text(s.get("predicted_answer", ""))
        ta = s.get("true_answer", "")
        scores_ans.append(compute_rouge_l(pa, ta)["rouge_l_f1"])
    rouge_nonabst = round(statistics.mean(scores_ans), 4) if scores_ans else 0.0

    # --- Citation rates ---
    cited_answered = sum(1 for s in answered_samples if CITATION_RE.search(get_text(s.get("predicted_answer", ""))))
    citation_support_rate = cited_answered / n_ans if n_ans > 0 else 0.0
    answer_with_citation_rate = cited_answered / n

    # --- Latency ---
    latencies = [s["elapsed_ms"] for s in samples if "elapsed_ms" in s]
    if "latency_stats" in metrics:
        ls = metrics["latency_stats"]
        lat_mean   = ls["avg_ms"]
        lat_median = ls["median_ms"]
        lat_min    = ls["min_ms"]
        lat_max    = ls["max_ms"]
    elif latencies:
        lat_mean   = round(statistics.mean(latencies), 1)
        lat_median = round(statistics.median(latencies), 1)
        lat_min    = min(latencies)
        lat_max    = max(latencies)
    else:
        lat_mean = lat_median = lat_min = lat_max = 0

    return {
        "config": config,
        "rouge_l_f1_all": rouge_all,
        "rouge_l_f1_noabst": rouge_nonabst,
        "abstention_rate": round(abstention_rate, 4),
        "n_abstained": n_abs,
        "n_answered": n_ans,
        "citation_support_rate": round(citation_support_rate, 4),
        "answer_with_citation_rate": round(answer_with_citation_rate, 4),
        "lat_mean": lat_mean,
        "lat_median": lat_median,
        "lat_min": lat_min,
        "lat_max": lat_max,
    }


def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"


def main():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    dt_str = now.strftime("%Y-%m-%d %H:%M:%S")

    all_metrics = {}
    for config, path in FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = compute_metrics(config, data)
        all_metrics[config] = m
        print(f"[{config}] rouge_all={fmt(m['rouge_l_f1_all'])} rouge_noabst={fmt(m['rouge_l_f1_noabst'])} "
              f"abst={pct(m['abstention_rate'])} ({m['n_abstained']}/30) "
              f"cit_supp={pct(m['citation_support_rate'])} ans_w_cit={pct(m['answer_with_citation_rate'])} "
              f"lat_mean={m['lat_mean']}ms")

    # --- Analysis flags ---
    fp = all_metrics["full-pipeline"]
    nc = all_metrics["no-critic"]
    nir = all_metrics["no-iterative-retrieval"]
    br = all_metrics["basic-rag"]
    pl = all_metrics["plain-llm-no-rag"]

    # (a) no-critic vs full-pipeline
    rfl_fp = fp["rouge_l_f1_all"]
    rfl_nc = nc["rouge_l_f1_all"]
    if rfl_fp and rfl_nc:
        rel_diff_nc = abs(rfl_nc - rfl_fp) / rfl_fp
        if rel_diff_nc <= 0.10:
            critic_flag = "**Critic may not be earning its latency cost**"
        elif rfl_nc > rfl_fp:
            # no-critic scores higher — critic is increasing abstention rate
            critic_flag = ("**Critic increases abstention rate, reducing ROUGE-L (all); "
                           "removing critic raises answered-question count but may lower answer quality**")
        else:
            critic_flag = "**Critic provides measurable quality improvement**"
        critic_detail = (f"no-critic ROUGE-L (all) = {fmt(rfl_nc)} vs full-pipeline = {fmt(rfl_fp)} "
                         f"(relative diff {rel_diff_nc*100:.1f}%); "
                         f"no-critic abstention rate = {pct(nc['abstention_rate'])} vs "
                         f"full-pipeline = {pct(fp['abstention_rate'])}")
    else:
        critic_flag = "N/A"
        critic_detail = ""

    # (b) no-iterative-retrieval vs full-pipeline
    rfl_nir = nir["rouge_l_f1_all"]
    if rfl_fp and rfl_nir:
        drop_nir = (rfl_fp - rfl_nir) / rfl_fp
        iter_flag = ("**MGA loop earns its complexity**"
                     if drop_nir > 0.15
                     else "**Iterative retrieval provides minimal uplift**")
        iter_detail = (f"no-iterative-retrieval ROUGE-L (all) = {fmt(rfl_nir)} vs full-pipeline = {fmt(rfl_fp)} "
                       f"(drop {drop_nir*100:.1f}%)")
    else:
        iter_flag = "N/A"
        iter_detail = ""

    # (c) basic-rag vs plain-llm abstention rate difference
    abs_br = br["abstention_rate"]
    abs_pl = pl["abstention_rate"]
    abs_diff = abs_br - abs_pl
    if abs(abs_diff) > 0.10:
        abst_flag = (f"Retrieval-based configs detect retrieval failure and abstain "
                     f"(basic-rag abstention {pct(abs_br)} vs plain-llm {pct(abs_pl)}, diff {abs_diff*100:+.1f} ppts) — "
                     f"plain LLM always generates an answer even without grounding.")
    else:
        abst_flag = (f"basic-rag abstention rate ({pct(abs_br)}) and plain-llm-no-rag ({pct(abs_pl)}) "
                     f"do not differ significantly (diff {abs_diff*100:+.1f} ppts).")

    # --- Write report ---
    out_path = RESULTS / f"step4_ablation_summary_{timestamp}.md"

    # Table rows
    configs_order = [
        "full-pipeline",
        "no-critic",
        "no-calibration",
        "no-iterative-retrieval",
        "basic-rag",
        "plain-llm-no-rag",
    ]

    table_rows = []
    for cfg in configs_order:
        m = all_metrics[cfg]
        row = (
            f"| {cfg} "
            f"| {fmt(m['rouge_l_f1_all'])} "
            f"| {fmt(m['rouge_l_f1_noabst'])} "
            f"| {pct(m['abstention_rate'])} "
            f"| {pct(m['citation_support_rate'])} "
            f"| {pct(m['answer_with_citation_rate'])} "
            f"| {m['lat_mean']} |"
        )
        table_rows.append(row)

    table = "\n".join([
        "| Config | ROUGE-L F1 (all) | ROUGE-L F1 (non-abstained) | Abstention Rate | Citation Support Rate | Ans w/ Citation Rate | Mean Latency (ms) |",
        "|--------|-----------------|---------------------------|-----------------|----------------------|----------------------|-------------------|",
    ] + table_rows)

    report = f"""# Step 4 Ablation Evaluation — RAGMedX-MA

**Date/Time:** {dt_str}
**Question subset:** First 30 questions from `eval_questions.json`
**generator_temperature:** 0.0
**eval_split index path:** `d:\\Ragmed_2\\backend\\data\\faiss_index\\eval_split`
**Model:** Qwen/Qwen3-8B, 4-bit NF4, RTX 5060 Laptop 8GB

---

## Summary Table

{table}

> ROUGE-L F1 (non-abstained): computed over answered questions only (abstentions excluded from denominator).
> Citation Support Rate: fraction of non-abstained answers containing a citation marker ([Doc N] or [EN]).
> Ans w/ Citation Rate: citations over all 30 questions.

---

## Analysis

### (a) Critic component contribution

{critic_flag}

{critic_detail}. The no-critic config skips the answer-verification critic pass. The full-pipeline's higher abstention rate ({pct(fp['abstention_rate'])}) versus no-critic ({pct(nc['abstention_rate'])}) indicates the critic is withholding answers the no-critic variant would have generated, which depresses ROUGE-L (all) even if per-answer quality is similar. The ROUGE-L (non-abstained) comparison — full-pipeline {fmt(fp['rouge_l_f1_noabst'])} vs no-critic {fmt(nc['rouge_l_f1_noabst'])} — is a fairer signal for per-answer quality. The critic step adds latency ({fp['lat_mean']} ms mean vs {nc['lat_mean']} ms without critic); whether its abstention behaviour represents correct caution or over-conservatism requires qualitative inspection of the withheld answers.

### (b) Iterative retrieval (MGA loop) contribution

{iter_flag}

{iter_detail}. The no-iterative-retrieval config executes a single retrieval pass without the multi-gap-analysis loop. A drop greater than 15% relative signals that the MGA loop is genuinely improving retrieval coverage for harder questions. A smaller drop suggests the initial retrieval pass already captures the relevant evidence and the extra round-trips add latency without proportionate quality gains. Note that no-iterative-retrieval also has the highest abstention rate ({pct(nir['abstention_rate'])}), suggesting a single retrieval pass more often fails to satisfy the sufficiency check.

### (c) Abstention behaviour — retrieval-based vs plain LLM

{abst_flag}

RAG-based pipelines can inspect retrieved evidence and withhold an answer when the evidence is insufficient (abstention). Plain LLM inference has no such grounding check and will always produce a response, which can result in confident but incorrect answers on knowledge-boundary questions. This behavioural difference is a key quality-of-service property independent of ROUGE-L.

### (d) Known limitation — gap-analysis / reranking hallucination

**Affected configs: full-pipeline, no-critic, no-calibration** (any config that retains gap-analysis and reranking agents).

The Q3 pilot evaluation identified a systematic failure mode: the gap-analysis agent declared `sufficient=True` on irrelevant retrieved evidence, and the reranking agent subsequently invented the statement "Doc 1 states trade name of sildenafil as Prolia" — a hallucinated attribution to a real document. This hallucination propagated to the generated answer in the no-calibration and no-critic configs (where the answer-calibration or verification step was absent to catch it), while the full-pipeline config correctly abstained on the sildenafil question. The finding highlights that gap-analysis sufficiency judgement and reranker attribution are both under-constrained and can amplify rather than correct retrieval errors. Until these agents are re-prompted or otherwise hardened, ROUGE-L and citation rates for the affected configs are inflated by confidently wrong answers.

---

## Per-Config File Paths

| Config | File |
|--------|------|
| full-pipeline | `{FILES["full-pipeline"]}` |
| no-critic | `{FILES["no-critic"]}` |
| no-calibration | `{FILES["no-calibration"]}` |
| no-iterative-retrieval | `{FILES["no-iterative-retrieval"]}` |
| basic-rag | `{FILES["basic-rag"]}` |
| plain-llm-no-rag | `{FILES["plain-llm-no-rag"]}` |
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to: {out_path}")
    print("\n--- TABLE ---")
    print(table)
    return str(out_path)


if __name__ == "__main__":
    main()
