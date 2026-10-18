# Runtime & Completion Report — Phase 5 Evaluation (FEAT-003 recompute)

**Run dir:** `phase5_20261018_053140/`
**Date:** 2026-10-18T05:18:31.042114+00:00 · **Branch:** `yuvraj/llm` · **GPU:** NVIDIA RTX 5060 Laptop (8 GB)

## Scope actually completed

| Item | Value |
|---|---|
| Executed matrix | **40 questions × 8 configs = 320 cells** |
| Cells completed (new this session) | **192** |
| Cells failed | **0** |
| Reused pilot cells (not re-run) | **128** |
| Duplicate writes prevented | 0 |
| Budget stop triggered | False |

**Tier label:** `REDUCED_VALIDATION_SUBSET`. This is a *complete* 40×8 experiment on a fixed question set across all configs — NOT a partial slice of a larger run, and NOT reported as a 50×8 or 500-question result.

## Runtime

| Metric | Value |
|---|---|
| Actual run wall-clock (new cells) | 18903.8 s ≈ 5.25 h |
| Model load (once) | 34.4 s |
| Peak VRAM (run) | 10161.24 MB (offload/paging on 8 GB card; no OOM) |
| Mean per new cell | 98.5 s |

## Reliability

- 0 execution errors across the matrix (see `failure_cases.json`).
- Genuine abstentions are recorded distinct from errors (`abstain_flag=True`, `errors=None`); no abstention or failure was silently dropped.

## Safety

- eval_split IndexLeakageGuard **PASS**; resolved index = `backend/data/faiss_index/eval_split`.
- Production index `backend/data/faiss_index/` left untouched; eval_split index files unchanged.
- Temperature 0.0, v1 prompts, top_k_retrieval=10 / top_k_final=3, Qwen3-8B 4-bit NF4; precision and batch size unchanged; `llm_mode=``local`, no mock fallback.

