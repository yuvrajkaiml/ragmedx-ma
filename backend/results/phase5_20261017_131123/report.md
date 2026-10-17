# Phase 5 — 8-Config Experiment Report

Generated: 2026-10-17T08:35:00.224753+00:00
Tier: **REDUCED_VALIDATION_SUBSET**  (dev=1, test=0)

- Resolved eval_split: `D:\Ragmed_2\backend\data\faiss_index\eval_split`
- LLM: `Qwen/Qwen3-8B` mode=`local` temp=0.0
- torch 2.11.0+cu128 / CUDA 12.8

## Answer correctness (BENCHMARK GROUND TRUTH, token-containment ≥0.6)

| Config | n | correct | accuracy | 95% Wilson CI | abstention |
|---|---|---|---|---|---|
| phase5-basic-rag | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-full-pipeline | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-improved-no-verification | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-improved-v2 | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-no-calibration | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-no-critic | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-no-iterative-retrieval | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |
| phase5-plain-llm-no-rag | 1 | 1 | 1.0 | [0.206549, 1.0] | 0.0 |

## Paired McNemar vs `phase5-full-pipeline` (Holm-corrected)

| Config | shared n | p (raw) | p (Holm) | significant? | method |
|---|---|---|---|---|---|
| phase5-basic-rag | 1 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-no-verification | 1 | 1.0 | 1.0 | False | exact_binomial |
| phase5-improved-v2 | 1 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-calibration | 1 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-critic | 1 | 1.0 | 1.0 | False | exact_binomial |
| phase5-no-iterative-retrieval | 1 | 1.0 | 1.0 | False | exact_binomial |
| phase5-plain-llm-no-rag | 1 | 1.0 | 1.0 | False | exact_binomial |

## Latency (SOFTWARE TEST) vs PRD targets

PRD: single-pass <5s; iterative ≤3 rounds <15s; retrieval <500ms; rerank <2s (cold vs warm separate).

| Config | cold mean (ms) | warm mean (ms) |
|---|---|---|
| phase5-basic-rag | 79535.0 | NOT MEASURED |
| phase5-full-pipeline | 120210.0 | NOT MEASURED |
| phase5-improved-no-verification | 101353.0 | 0.0 |
| phase5-improved-v2 | 116902.0 | 0.0 |
| phase5-no-calibration | 112145.0 | NOT MEASURED |
| phase5-no-critic | 84542.0 | NOT MEASURED |
| phase5-no-iterative-retrieval | 103940.0 | NOT MEASURED |
| phase5-plain-llm-no-rag | 5677.0 | NOT MEASURED |

## Honesty / limitations

- Every metric is tagged with an evidence tier; `NOT MEASURED` is used where data is genuinely absent (e.g. per-stage latency on historical baselines where record_node_timing is OFF; PubMedQA retrieval-quality not in the eval index).
- No superiority claim is made without a Holm-adjusted significant paired test. Small reduced-subset n yields wide CIs; treat deltas as directional only.
