# Phase 5 — 8-Config Experiment Report

Generated: 2026-10-18T05:48:00.273029+00:00
Tier: **REDUCED_VALIDATION_SUBSET**  (dev=0, test=40)

- Resolved eval_split: `D:\Ragmed_2\backend\data\faiss_index\eval_split`
- LLM: `Qwen/Qwen3-8B` mode=`local` temp=0.0
- torch 2.11.0+cu128 / CUDA 12.8

## Answer correctness (BENCHMARK GROUND TRUTH, token-containment ≥0.6)

| Config | n | correct | accuracy | 95% Wilson CI | abstention |
|---|---|---|---|---|---|
| phase5-basic-rag | 40 | 1 | 0.025 | [0.004427, 0.128814] | 0.875 |
| phase5-full-pipeline | 40 | 0 | 0.0 | [0.0, 0.087622] | 0.95 |
| phase5-improved-no-verification | 40 | 0 | 0.0 | [0.0, 0.087622] | 0.95 |
| phase5-improved-v2 | 40 | 0 | 0.0 | [0.0, 0.087622] | 0.95 |
| phase5-no-calibration | 40 | 0 | 0.0 | [0.0, 0.087622] | 0.875 |
| phase5-no-critic | 40 | 1 | 0.025 | [0.004427, 0.128814] | 0.875 |
| phase5-no-iterative-retrieval | 40 | 0 | 0.0 | [0.0, 0.087622] | 0.95 |
| phase5-plain-llm-no-rag | 40 | 15 | 0.375 | [0.24223, 0.529676] | 0.0 |

## Paired McNemar vs `phase5-full-pipeline` (Holm-corrected)

| Config | shared n | p (raw) | p (Holm) | significant? | method |
|---|---|---|---|---|---|
| phase5-basic-rag | 40 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-no-verification | 40 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-v2 | 40 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-calibration | 40 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-critic | 40 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-iterative-retrieval | 40 | 1.0 | 1.0 | False | exact_binomial |
| phase5-plain-llm-no-rag | 40 | 6.1e-05 | 0.000427 | True | exact_binomial |

## Latency (SOFTWARE TEST) vs PRD targets

PRD: single-pass <5s; iterative ≤3 rounds <15s; retrieval <500ms; rerank <2s (cold vs warm separate).

| Config | cold mean (ms) | warm mean (ms) |
|---|---|---|
| phase5-basic-rag | 60592.3 | NOT MEASURED |
| phase5-full-pipeline | 137212.88 | NOT MEASURED |
| phase5-improved-no-verification | 121484.75 | 0.23 |
| phase5-improved-v2 | 135499.77 | 18.2 |
| phase5-no-calibration | 136585.02 | NOT MEASURED |
| phase5-no-critic | 130628.23 | NOT MEASURED |
| phase5-no-iterative-retrieval | 71410.12 | NOT MEASURED |
| phase5-plain-llm-no-rag | 11167.52 | NOT MEASURED |

## Honesty / limitations

- Every metric is tagged with an evidence tier; `NOT MEASURED` is used where data is genuinely absent (e.g. per-stage latency on historical baselines where record_node_timing is OFF; PubMedQA retrieval-quality not in the eval index).
- No superiority claim is made without a Holm-adjusted significant paired test. Small pilot n yields wide CIs; treat deltas as directional only.
- Answer-correctness here is token-containment vs gold (threshold 0.6). It scores an abstention as incorrect, so configs with high abstention rates (the RAG baselines on this BioASQ/PubMedQA mix) are penalised relative to always-answering plain-LLM. This is a measurement property, NOT evidence that abstention is wrong — abstention can be the safe behaviour. Report abstention_rate alongside accuracy; do not read accuracy alone.

## Remaining full 500×8 sweep — NOT RUN (exceeds 4 h GPU budget)

The full 500-question × 8-config sweep is estimated at ~40+ hours of GPU wall-clock and was deliberately NOT run (it exceeds the 4-hour budget). To run it later:

```
d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe backend\scripts\run_phase5_experiment.py --full --output-root D:\Ragmed_2\backend\results\phase5_20261018_053140
```
