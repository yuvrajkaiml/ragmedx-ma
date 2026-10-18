# Phase 5 — 8-Config Experiment Report

Generated: 2026-10-17T23:55:51.409030+00:00
Tier: **BUDGET_BOUNDED_PILOT**  (dev=0, test=3)

Wall-clock: 0.68 h (3 questions × 8 configs; mean 815.5 s/question-all-configs). Peak VRAM (run): 10163.91 MB.
Pilot: n=3 test questions (seed=42, stratified by source, budget=None h, stopped_early=False).

- Resolved eval_split: `D:\Ragmed_2\backend\data\faiss_index\eval_split`
- LLM: `Qwen/Qwen3-8B` mode=`local` temp=0.0
- torch 2.11.0+cu128 / CUDA 12.8

## Answer correctness (BENCHMARK GROUND TRUTH, token-containment ≥0.6)

| Config | n | correct | accuracy | 95% Wilson CI | abstention |
|---|---|---|---|---|---|
| phase5-basic-rag | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-full-pipeline | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-improved-no-verification | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-improved-v2 | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-no-calibration | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-no-critic | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-no-iterative-retrieval | 3 | 0 | 0.0 | [0.0, 0.561497] | 1.0 |
| phase5-plain-llm-no-rag | 3 | 0 | 0.0 | [0.0, 0.561497] | 0.0 |

## Paired McNemar vs `phase5-full-pipeline` (Holm-corrected)

| Config | shared n | p (raw) | p (Holm) | significant? | method |
|---|---|---|---|---|---|
| phase5-basic-rag | 3 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-no-verification | 3 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-v2 | 3 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-calibration | 3 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-critic | 3 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-iterative-retrieval | 3 | 1.0 | 1.0 | False | exact_binomial |
| phase5-plain-llm-no-rag | 3 | 1.0 | 1.0 | False | exact_binomial |

## Latency (SOFTWARE TEST) vs PRD targets

PRD: single-pass <5s; iterative ≤3 rounds <15s; retrieval <500ms; rerank <2s (cold vs warm separate).

| Config | cold mean (ms) | warm mean (ms) |
|---|---|---|
| phase5-basic-rag | 62457.0 | NOT MEASURED |
| phase5-full-pipeline | 141265.0 | NOT MEASURED |
| phase5-improved-no-verification | 122701.33 | 0.33 |
| phase5-improved-v2 | 143816.67 | 17.67 |
| phase5-no-calibration | 137106.33 | NOT MEASURED |
| phase5-no-critic | 137088.33 | NOT MEASURED |
| phase5-no-iterative-retrieval | 57888.0 | NOT MEASURED |
| phase5-plain-llm-no-rag | 13177.67 | NOT MEASURED |

## Honesty / limitations

- Every metric is tagged with an evidence tier; `NOT MEASURED` is used where data is genuinely absent (e.g. per-stage latency on historical baselines where record_node_timing is OFF; PubMedQA retrieval-quality not in the eval index).
- No superiority claim is made without a Holm-adjusted significant paired test. Small pilot n yields wide CIs; treat deltas as directional only.
- Answer-correctness here is token-containment vs gold (threshold 0.6). It scores an abstention as incorrect, so configs with high abstention rates (the RAG baselines on this BioASQ/PubMedQA mix) are penalised relative to always-answering plain-LLM. This is a measurement property, NOT evidence that abstention is wrong — abstention can be the safe behaviour. Report abstention_rate alongside accuracy; do not read accuracy alone.

## Remaining full 500×8 sweep — NOT RUN (exceeds 4 h GPU budget)

The full 500-question × 8-config sweep is estimated at ~40+ hours of GPU wall-clock and was deliberately NOT run (it exceeds the 4-hour budget). To run it later:

```
d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe backend\scripts\run_phase5_experiment.py --full --output-root d:\Ragmed_2\backend\results\phase5_benchmark
```
