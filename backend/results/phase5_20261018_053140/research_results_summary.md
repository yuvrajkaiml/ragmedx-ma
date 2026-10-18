# Research-Results Summary — Phase 5 Evaluation (FEAT-003)

**Scope:** 40 unique questions (seed 42, source-stratified) × 8 configurations = 320 comparable cells, all on the identical question set. Reused validated pilot cells + newly executed cells; see runtime report. Peak VRAM (run) ≈ 10161.24 MB on an 8 GB RTX 5060.

> **Scope honesty.** The original task described a 500-question / 50×8 ambition. A fresh runtime benchmark projected 50×8 over the hard 8 h budget, so the target was reduced to a complete 40×8 run (a strict subset of the frozen 50, itself a superset of the pilot 16). **This is a complete 40×8 experiment, not a partial 50×8 or 500-question run.**

## Headline observations (directional; small n)

1. **Abstention dominates the RAG configs on this question mix.** Abstention rates: phase5-basic-rag=0.875, phase5-full-pipeline=0.95, phase5-improved-no-verification=0.95, phase5-improved-v2=0.95, phase5-no-calibration=0.875, phase5-no-critic=0.875, phase5-no-iterative-retrieval=0.95, phase5-plain-llm-no-rag=0.0.

2. **Token-containment accuracy (threshold 0.6, vs gold):** see `metric_summary.json`. Paired McNemar vs `phase5-full-pipeline` is Holm-significant only for: `phase5-plain-llm-no-rag`.

3. **ROUGE-L F1 (primary answer quality vs gold):** paired bootstrap ROUGE-L delta vs baseline excludes zero only for: `phase5-plain-llm-no-rag`.

4. **Citation support:** reported per config in `comparison_table.md`; plain-LLM-no-RAG attaches no citations by design (no retrieval).

5. **Claim-level evidence (unsupported-claim / contradiction / sufficiency):** MEASURED only for configs that produced `claim_verdicts` (see per-config `citation.claim_level`); NOT MEASURED elsewhere with reason. Where measured, claims were overwhelmingly `unverifiable` against the retrieved evidence with no detected contradictions — read with the small denominators shown.

## What is NOT measured (and why)

- **Retrieval quality (recall@K / MRR / nDCG@K): NOT MEASURED.** Verified directly in this recompute: 0/789 BioASQ gold PMIDs appear in the eval_split index universe. The gold PMIDs are not the indexed documents, so retrieval-vs-gold cannot be scored. This corrects the manifest's earlier 'BioASQ MEASURABLE' assumption with evidence. PubMedQA retrieval is also NOT MEASURED.
- **False-acceptance / answerability on unanswerable questions:** NOT MEASURED — the frozen split has no gold `should-abstain` labels, so false_abstention_rate == abstention_rate and false_acceptance_rate is undefined here.

## Limitations

- **Small sample (n = 40 per config).** Proportion CIs are wide; between-RAG-config differences are statistically indistinguishable. Treat every non-significant delta as directional.
- **Accuracy / ROUGE-L penalize abstention.** High-abstention configs look worse on answer quality but may be behaving conservatively. Read answer quality only alongside abstention rate.
- **No clinical-safety or superiority claim is made.** Any significant result reflects an always-answer vs mostly-abstain artifact on this mix, not evidence that a config is clinically better. A 0.875–0.95 RAG abstention rate warrants investigating threshold calibration — a follow-up, not a conclusion.
- **Single eval set, single seed.** Specific to this seed-42 sample of the 250-question test pool.

## Artifacts

- Raw: `per_question/*.jsonl` (source of truth).
- Metrics: `metric_summary.json`, `extended_metrics.json`, `comparison_table.md`, `latency_breakdown.json`, `failure_cases.json`, `summary.json`.
- Reports: `report.md`, `runtime_completion_report.md`, `reproducibility_report.md`, this file.

