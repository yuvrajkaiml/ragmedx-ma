# Reproducibility Report — Phase 5 40×8 Evaluation

**Run dir:** `backend/results/phase5_20261018_053140/`

## Dataset & split (immutable)

- **Pool:** `backend/results/phase5_20261017_131123/splits/test.json` — 250 questions
  (167 PubMedQA + 83 BioASQ), seed 42, stratified by `source`.
  Test-split manifest hash `8e2b2d925afd86b8d5f2567594bcf40f64ed3e0cab6c3e5b169b524569621eee`.
- **The eval split was reused verbatim, never rebuilt.** eval_split index files (SHA-256, verified
  unchanged pre/post run):
  - `index.faiss` `fc18d20d30ea92e6d4b7581786bd1ba640e18e658d994466d47d924c847fc258`
  - `chunks.pkl`  `6370a1b47b2d0e799788f3406f320dc29057c008db4f4d8ce19f761456a5f5d1`
  - 58,173 chunks. Production index `backend/data/faiss_index/` left untouched.

## Question selection (frozen, nested)

- Function: `run_phase5_experiment._stratified_sample(test_250, n, seed=42, stratify_by="source")`.
- `question_ids_40` (this run) ⊂ `question_ids_50`; `pilot_reused_ids_16` ⊂ `question_ids_40`.
- Frozen lists live in `backend/results/phase5_manifest/experiment_manifest.json` and
  `question_ids_50.json`. The 40 executed IDs are reproduced in this run's `splits/test.json`.
- 128 of the 320 cells were reused from the validated 16-question pilot
  (`backend/results/phase5_20261017_131123/`); the 16 pilot IDs are a strict subset of the 40.

## Models & generation (unchanged)

- LLM: **Qwen/Qwen3-8B**, 4-bit NF4, local mode, mock fallback disabled, `hard_assert_llm_mode_local`.
- Generation: **temperature 0.0** (deterministic / greedy; `do_sample=False` confirmed at runtime).
- Prompts: `backend/prompts/v1/*.txt`, unchanged (per-file SHA-256 recorded in the manifest).
- Retrieval: `top_k_retrieval=10`, `top_k_final=3`. Embedding `BAAI/bge-m3` (cuda),
  reranker `BAAI/bge-reranker-v2-m3` (cpu).

## Code & environment

- Code revision: branch `yuvraj/llm`, benchmark committed at `055103be…`.
- Runner: `backend/scripts/run_phase5_experiment.py` (`--target 40`, per-cell checkpoint + pilot reuse).
- Extended-metrics recompute: `backend/scripts/recompute_phase5_extended_metrics.py` (CPU, reuses
  `evaluation/metrics.py` + `evaluation/statistics.py` — no metric redefined).
- torch 2.11.0+cu128, CUDA 12.8, GPU RTX 5060 Laptop 8 GB.
- Metric/statistics module SHA-256 and library versions: see `experiment_manifest.json`.

## How to reproduce / regenerate

GPU re-run (idempotent — reuses all completed cells):
```
d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe backend\scripts\run_phase5_experiment.py \
    --target 40 --output-root backend\results\phase5_20261018_053140 --budget-hours 7.5
```
Metrics-only recompute from saved raw outputs (no GPU):
```
d:\Ragmed_2\.venv\Scripts\python.exe backend\scripts\recompute_phase5_extended_metrics.py \
    --run-dir backend\results\phase5_20261018_053140
```
All metrics in `metric_summary.json` and `extended_metrics.json` are regenerable from the raw
`per_question/*.jsonl` files alone; those raw files are the source of truth.
