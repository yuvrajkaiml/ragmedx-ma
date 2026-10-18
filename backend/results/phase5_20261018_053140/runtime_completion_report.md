# Runtime & Completion Report — Phase 5 40×8 Evaluation

**Run dir:** `backend/results/phase5_20261018_053140/`
**Date:** 2026-10-18 · **Branch:** `yuvraj/llm` · **GPU:** NVIDIA RTX 5060 Laptop (8 GB)

## Scope actually completed

| Item | Value |
|---|---|
| Executed matrix | **40 questions × 8 configs = 320 cells** |
| Cells completed | **320 / 320** |
| Cells failed | **0** |
| Reused pilot cells (not re-run) | **128** (16 pilot questions × 8 configs, byte-faithful) |
| New cells run this session | **192** |
| Duplicate writes prevented | 0 (none occurred; guard active) |
| Budget stop triggered | No |

**Tier label:** `BUDGET_BOUNDED → COMPLETE 40×8`. This is a *complete* 40×8 experiment on a fixed
question set across all 8 configs — not a partial slice of a larger run.

> The original task goal referenced a 500-question / 50×8 ambition (and an older ~40 GPU-hour estimate).
> The **actual completed scope is 40×8 = 320 cells**, chosen after a fresh benchmark showed 50×8 would
> exceed the hard 8 h budget. See `benchmark-decision.json`. Do not read this report as a 50×8 or
> 500-question result.

## Runtime

| Metric | Value |
|---|---|
| Fresh benchmark (3 q × 8 configs) | **815.5 s/question-across-all-8-configs**, 101.9 s/cell |
| Projected 40×8 total | ~6.24 h (incl. overhead) |
| **Actual run wall-clock (192 new cells)** | **18,903.8 s ≈ 5.25 h** |
| Model load (once) | ~34.4 s |
| Peak VRAM | **10,161 MB** (offload/paging on 8 GB card; no OOM) |
| Mean per new cell | ~98.5 s (18,903.8 / 192) — within ~3% of benchmark projection |

The actual per-cell runtime (~98.5 s) was slightly faster than the benchmark projection (101.9 s),
confirming the projection was sound and the old 40 GPU-hour estimate was correctly discarded.

## Reliability

- **0 execution errors / 0 OOM** across all 320 cells.
- Checkpoint/resume verified: a dry-run against the completed dir reports `reused=320 missing=0`
  (restart repeats zero valid completed cells).
- Genuine abstentions (161 of the 192 new cells) are recorded distinct from errors
  (`abstain_flag=True`, `errors=None`); no abstention or failure was silently dropped.

## Safety

- eval_split IndexLeakageGuard **PASS**; resolved index = `backend/data/faiss_index/eval_split`.
- Production index `backend/data/faiss_index/` SHA-256 **unchanged** pre/post run.
- eval_split `index.faiss` / `chunks.pkl` SHA-256 **unchanged** pre/post run.
- Temperature 0.0, v1 prompts, `top_k_retrieval=10` / `top_k_final=3`, Qwen3-8B 4-bit NF4,
  precision and batch size all unchanged; `llm_mode=local`, no mock fallback.
