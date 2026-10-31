# Phase 5 — 8-Config Experiment Report

Generated: 2026-10-17T12:46:42.334424+00:00
Tier: **BUDGET_BOUNDED_PILOT**  (dev=0, test=18)

Wall-clock: 3.691 h (16 questions × 8 configs; mean 830.4 s/question-all-configs). Peak VRAM (run): 10157.1 MB.
Pilot: n=16 test questions (seed=42, stratified by source, budget=3.8 h, stopped_early=True).

- Resolved eval_split: `D:\Ragmed_2\backend\data\faiss_index\eval_split`
- LLM: `Qwen/Qwen3-8B` mode=`local` temp=0.0
- torch 2.11.0+cu128 / CUDA 12.8

## Answer correctness (BENCHMARK GROUND TRUTH, token-containment ≥0.6)

| Config | n | correct | accuracy | 95% Wilson CI | abstention |
|---|---|---|---|---|---|
| phase5-basic-rag | 16 | 1 | 0.0625 | [0.011119, 0.283287] | 0.75 |
| phase5-full-pipeline | 16 | 0 | 0.0 | [0.0, 0.193608] | 0.9375 |
| phase5-improved-no-verification | 16 | 0 | 0.0 | [0.0, 0.193608] | 0.9375 |
| phase5-improved-v2 | 16 | 0 | 0.0 | [0.0, 0.193608] | 0.9375 |
| phase5-no-calibration | 16 | 0 | 0.0 | [0.0, 0.193608] | 0.75 |
| phase5-no-critic | 16 | 1 | 0.0625 | [0.011119, 0.283287] | 0.75 |
| phase5-no-iterative-retrieval | 16 | 0 | 0.0 | [0.0, 0.193608] | 0.9375 |
| phase5-plain-llm-no-rag | 16 | 6 | 0.375 | [0.184812, 0.61359] | 0.0 |

## Paired McNemar vs `phase5-full-pipeline` (Holm-corrected)

| Config | shared n | p (raw) | p (Holm) | significant? | method |
|---|---|---|---|---|---|
| phase5-basic-rag | 16 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-no-verification | 16 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-v2 | 16 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-calibration | 16 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-critic | 16 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-iterative-retrieval | 16 | 1.0 | 1.0 | False | exact_binomial |
| phase5-plain-llm-no-rag | 16 | 0.03125 | 0.21875 | False | exact_binomial |

## Latency (SOFTWARE TEST) vs PRD targets

PRD: single-pass <5s; iterative ≤3 rounds <15s; retrieval <500ms; rerank <2s (cold vs warm separate).

| Config | cold mean (ms) | warm mean (ms) |
|---|---|---|
| phase5-basic-rag | 63742.81 | NOT MEASURED |
| phase5-full-pipeline | 140713.12 | NOT MEASURED |
| phase5-improved-no-verification | 125241.81 | 0.12 |
| phase5-improved-v2 | 137773.44 | 19.62 |
| phase5-no-calibration | 140146.5 | NOT MEASURED |
| phase5-no-critic | 123891.25 | NOT MEASURED |
| phase5-no-iterative-retrieval | 89239.38 | NOT MEASURED |
| phase5-plain-llm-no-rag | 9624.5 | NOT MEASURED |

## Honesty / limitations

- Every metric is tagged with an evidence tier; `NOT MEASURED` is used where data is genuinely absent (e.g. per-stage latency on historical baselines where record_node_timing is OFF; PubMedQA retrieval-quality not in the eval index).
- No superiority claim is made without a Holm-adjusted significant paired test. Small pilot n yields wide CIs; treat deltas as directional only.
- Answer-correctness here is token-containment vs gold (threshold 0.6). It scores an abstention as incorrect, so configs with high abstention rates (the RAG baselines on this BioASQ/PubMedQA mix) are penalised relative to always-answering plain-LLM. This is a measurement property, NOT evidence that abstention is wrong — abstention can be the safe behaviour. Report abstention_rate alongside accuracy; do not read accuracy alone.

## Remaining full 500×8 sweep — NOT RUN (exceeds 4 h GPU budget)

The full 500-question × 8-config sweep is estimated at ~40+ hours of GPU wall-clock and was deliberately NOT run (it exceeds the 4-hour budget). To run it later:

```
d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe backend\scripts\run_phase5_experiment.py --full --output-root results\phase5_20261017_131123
```
