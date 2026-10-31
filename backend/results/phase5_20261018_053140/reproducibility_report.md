# Reproducibility Report — Phase 5 Evaluation (FEAT-003 recompute)

**Run dir:** `phase5_20261018_053140/`

## Dataset & split (immutable)

- **Pool:** `backend/results/phase5_20261017_131123/splits/test.json` — 250 questions (167 PubMedQA + 83 BioASQ), seed 42, stratified by `source`. Test-split manifest hash `8e2b2d925afd86b8d5f2567594bcf40f64ed3e0cab6c3e5b169b524569621eee`.
- **eval split reused verbatim, never rebuilt.** eval_split index files (SHA-256):
  - `index.faiss` `fc18d20d30ea92e6d4b7581786bd1ba640e18e658d994466d47d924c847fc258`
  - `chunks.pkl`  `6370a1b47b2d0e799788f3406f320dc29057c008db4f4d8ce19f761456a5f5d1`
  - 58,173 chunks. Production index `backend/data/faiss_index/` left untouched.

## Question selection (frozen, nested)

- Function: `run_phase5_experiment._stratified_sample(test_250, n, seed=42, stratify_by="source")`.
- `question_ids_40` (this run) ⊂ `question_ids_50`; `pilot_reused_ids_16` ⊂ `question_ids_40`.
- Frozen lists in `backend/results/phase5_manifest/experiment_manifest.json`.
- 40 evaluated question IDs are reproduced in this run's `splits/test.json`.

## Models & generation (unchanged)

- LLM: **Qwen/Qwen3-8B**, 4-bit NF4, mode `local`, mock fallback disabled.
- Generation: **temperature 0.0** (deterministic / greedy).
- Prompts: `backend/prompts/v1/*.txt`, unchanged (per-file SHA-256 recorded in the manifest).
- Retrieval: `top_k_retrieval=10`, `top_k_final=3`.

## Code & environment

- Branch `yuvraj/llm`.
- Runner: `backend/scripts/run_phase5_experiment.py` (`--target 40`, per-cell checkpoint + pilot reuse).
- Metric recompute (this report): `backend/scripts/recompute_feat003_metrics.py` (CPU; reuses `evaluation/metrics.py` + `evaluation/statistics.py` + `aggregate_outputs` — no metric redefined).
- torch 2.11.0+cu128, CUDA 12.8, GPU RTX 5060 Laptop 8 GB.

## How to regenerate (no GPU)

```
d:\Ragmed_2\.venv\Scripts\python.exe backend\scripts\recompute_feat003_metrics.py \\
    --run-dir backend\results\phase5_20261018_053140
```
All metrics are regenerable from the raw `per_question/*.jsonl` alone; those raw files are the source of truth. The recompute is deterministic (bootstrap uses fixed seed 42).

