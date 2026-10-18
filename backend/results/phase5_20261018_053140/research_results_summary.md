# Research-Results Summary — Phase 5 40×8 Evaluation

**Scope:** 40 unique questions (27 PubMedQA + 13 BioASQ, seed 42, source-stratified) × 8 configurations
= 320 comparable cells, all on the identical question set. 128 cells reused from the validated
16-question pilot; 192 run fresh. 0 failures. Run wall-clock ≈ 5.25 h on an 8 GB RTX 5060.

> **Scope honesty.** The original task described a 500-question / 50×8 ambition. A fresh runtime
> benchmark projected 50×8 at ~8.73 h, exceeding the hard 8 h budget, so the target was reduced to a
> complete 40×8 (a strict subset of the frozen 50, itself a superset of the pilot 16). **This is a
> complete 40×8 experiment, not a partial 50×8 or 500-question run.**

## Headline observations (directional; small n = 40)

1. **The RAG configurations abstain heavily on this question mix.** Abstention rates: full-pipeline,
   improved-v2, improved-no-verification, no-iterative-retrieval = **0.95**; basic-rag, no-critic,
   no-calibration = **0.875**; plain-LLM-no-RAG = **0.0** (always answers).

2. **Token-containment accuracy (threshold 0.6, vs gold):** plain-LLM 0.375 (Wilson 95% CI
   0.242–0.530); basic-rag and no-critic 0.025; all other RAG configs 0.0. **This gap is driven by
   abstention, not by better reasoning:** the accuracy metric scores an abstention as incorrect, so an
   always-answering config is mechanically favored. Paired McNemar vs full-pipeline is Holm-significant
   **only** for plain-LLM (p_adj ≈ 4.3e-4) — and that cell table (n01=15, n10=0) is entirely the
   abstention asymmetry.

3. **ROUGE-L F1 (answer quality vs gold):** plain-LLM 0.060; basic-rag / no-critic 0.016; all other RAG
   configs ~0.010. Paired bootstrap ROUGE-L delta vs full-pipeline excludes zero **only** for plain-LLM
   (+0.050, 95% CI 0.023–0.082, n=40). All other configs' CIs include zero — no distinguishable
   difference at this sample size.

4. **Citation support:** every RAG config attached evidence to 100% of its (few) non-abstained answers;
   plain-LLM attaches none (by design, no retrieval).

## What is NOT measured (and why)

- **Retrieval quality (recall@K / MRR / nDCG@K): NOT MEASURED.** Verified directly: **0 of 789 BioASQ
  gold PMIDs appear in the eval_split index universe** (90,074 distinct PMID-like ids in `chunks.pkl`).
  The index was built from a BioASQ-derived corpus, but the questions' gold-standard PMIDs are not the
  indexed documents, so retrieval-vs-gold cannot be scored. This **corrects** the manifest's earlier
  "BioASQ MEASURABLE" assumption with evidence. PubMedQA retrieval is also NOT MEASURED (gold PMIDs
  not in the index).
- **Unsupported-claim rate / claim-level evidence sufficiency / contradiction:** NOT MEASURED as an
  aggregate — only configs with claim-verification/critic ON populate `claim_verdicts`, and the heavy
  abstention leaves too few answered cells to aggregate meaningfully. Raw `claim_verdicts` are preserved
  per cell for inspection.

## Limitations

- **Small sample (n = 40 per config).** Proportion CIs are wide; all between-RAG-config differences are
  statistically indistinguishable here. Treat every non-significant delta as directional only.
- **Accuracy/ROUGE-L penalize abstention.** High-abstention configs look worse on answer quality but may
  be behaving conservatively/safely. Read answer quality **only** alongside abstention rate.
- **No clinical-safety or superiority claim is made.** The one significant result (plain-LLM > RAG on
  containment accuracy) reflects an always-answer vs mostly-abstain artifact on this question mix, **not**
  evidence that skipping retrieval is clinically better or safer. If anything, a 0.875–0.95 abstention
  rate warrants investigating whether the RAG abstention threshold is miscalibrated for this eval set —
  that is a follow-up, not a conclusion of this run.
- **Single eval set, single seed.** Results are specific to this 40-question seed-42 sample of the
  250-question test pool.

## Artifacts

- Raw: `per_question/*.jsonl` (320 cells, source of truth).
- Metrics: `metric_summary.json` (accuracy, Wilson CIs, McNemar+Holm), `extended_metrics.json`
  (ROUGE-L, citation, retrieval NOT MEASURED, paired bootstrap), `comparison_table.md`,
  `latency_breakdown.json`, `failure_cases.json`, `summary.json`.
- Reports: `runtime_completion_report.md`, `reproducibility_report.md`, this file.
