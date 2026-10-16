# RAGMedX-MA — Progress Tracker

**Last updated:** 2026-10-17 (Phase 1 complete)  
**Branch:** `yuvraj/llm`  
**Protocol:** Re-read this file + handoff.md at the start of every phase.

---

## Phase 0 — Read-Only Audit

**Status:** IMPLEMENTED AND TESTED  
**Audit file:** `docs/audit_20261017_005421.md`  
**Commands run:** git branch/status/log; directory listings; Python --version; pip list; JSON parsing of result files and checkpoint; file existence checks (read-only, no code modified)  
**Test counts:** N/A (no tests in Phase 0; gate = audit written)

### Summary of Key Findings

1. **Architecture confirmed:** 7-agent LangGraph pipeline (QUA → IRA/MGA → ERA → GA → CA → CCA → ADM) built on BGE-M3 dense + BM25 sparse + RRF hybrid retrieval, cross-encoder reranking (CPU), Qwen3-8B (GPU, 4-bit NF4).

2. **Provenance gaps:** `index_id` never embedded in evidence dicts. `retrieval_iteration` tracked in orchestrator list but not per-item. `evidence_grade`/`reranking_score` not propagated to final response. `evidence_verification` (critic) not in public API response.

3. **Verification bypasses:** `disable_critic=True` and `disable_calibration=True` flags exist (used in ablation configs). No NLI-based claim verifier; only LLM critic. `EvidenceVerifier` is structural only (`min_content_length=1` in critic path).

4. **Metric issues:** `compute_accuracy` bug (empty string always matched) was fixed in current code but old pilot25 results may be affected. `accuracy` field in n30 results is now aliased to `rouge_l_f1`. Citation support rate denominator inconsistency between JSON and ablation summary. No ECE, Brier, MRR, nDCG, risk-coverage, McNemar, bootstrap implemented.

5. **No mock leakage in n30 canonical results:** All 6 n30 result files use `llm_mode=local` and `COMPLETED_RAG_EVALUATION` status.

6. **Datasets present:** MedQA sample (4q), MedMCQA+MedQA+PubMedQA+BioASQ+MMLU (via MIRAGE benchmark.json), BioASQ eval split (250q + train corpus). **MedRGB: ABSENT.**

7. **Checkpoint verified read-only:** All 6 configs done, 30/30 questions, output files exist, counts consistent.

8. **Sildenafil/Prolia regression documented** in ablation summary: no automated test.

9. **handoff.md does not exist** — create in Phase 1.

10. **`fsync` not called before `os.replace`** in checkpoint/output writers — gap for Phase 1 tests.

### Open Issues → BLOCKED.md

- MedRGB dataset absent — MedRGB metrics BLOCKED
- Eval index leakage not formally verified (smoke test not run, read-only audit)
- 500-question split not yet built

---

## Phase 1 — Workstream D: Checkpoint/Resume Reliability + Fault Injection Tests

**Status:** IMPLEMENTED AND TESTED  
**Target env:** CPU (`.venv\Scripts\python.exe`)  
**Completed:** 2026-10-17

### Files changed

| File | Action | Description |
|------|--------|-------------|
| `backend/tests/test_workstream_d_checkpoint.py` | CREATED | Full fault-injection test suite, 45 tests across 18 test classes |
| `backend/scripts/run_step4_matrix.py` | MODIFIED (minimal) | Added 3 missing helpers: `_detect_duplicate_question_ids`, `_compute_dataset_manifest_hash`, `_check_manifest_consistency` |

### Commands run

```
d:\Ragmed_2\.venv\Scripts\python.exe -m pytest d:\Ragmed_2\backend\tests\test_workstream_d_checkpoint.py -v
  → Run 1: 40 passed, 5 failed (missing helper functions in runner)
  → Fixes applied: added 3 helpers to run_step4_matrix.py; fixed source-inspection regex in test
  → Run 2: 45 passed, 0 failed

git -C d:\Ragmed_2 add backend/tests/test_workstream_d_checkpoint.py backend/scripts/run_step4_matrix.py
git -C d:\Ragmed_2 commit -m "Phase 1: Workstream D checkpoint/resume reliability tests and fixes"
  → Commit: 94000e14
```

### Test counts

| Run | Passed | Failed | Not Run |
|-----|--------|--------|---------|
| Initial | 40 | 5 | 0 |
| After fixes | **45** | **0** | 0 |

### Test scenarios covered (all PASS)

- [x] first-time init creates valid checkpoint, all configs pending
- [x] fully-completed fixture is skipped (all 6 configs verified)
- [x] partial resume: resumes from correct config/question
- [x] resume with reordered questions: stable question IDs (text-based matching)
- [x] duplicate question IDs: `_detect_duplicate_question_ids` detects correctly
- [x] altered dataset manifest: `_compute_dataset_manifest_hash` + `_check_manifest_consistency` detect drift
- [x] interrupted write before atomic replace: `.tmp` left behind, original checkpoint intact
- [x] corrupt checkpoint: `_load_checkpoint` returns None, no data loss
- [x] corrupt output file: `_verify_done_config` returns False
- [x] missing output file marked done: `_verify_done_config` returns False
- [x] invalid completed count: `_verify_done_config` detects sample count mismatch
- [x] GPU OOM recorded as error, never counted as answer/abstention/skip
- [x] crash after output before state: output file is ground truth, writer recovers
- [x] crash after state before output: missing output forces rerun
- [x] invalid/production index path: `_resolve_and_guard_index_dir` → `sys.exit(1)` before model load
- [x] guard verified before `load_model()` in source order
- [x] plain-llm-no-rag records `index_dir='N/A'` and `generator_temperature='N/A'`
- [x] atomic writes: `os.replace` called with same-dir temp, `IncrementalResultWriter._flush` uses atomic replace

### Real checkpoint verification (read-only)

Evidence tier: SOFTWARE TEST (read-only parse, no writes)

- `backend/data/checkpoints/step4_matrix_state.json`: parseable ✓
- All 6 configs present: `full-pipeline`, `no-critic`, `no-iterative-retrieval`, `no-calibration`, `basic-rag`, `plain-llm-no-rag` ✓
- All statuses: `"done"` ✓
- All `completed_questions`: 30/30 ✓
- All 6 output files exist on disk ✓
- Checkpoint mtime not changed by test run ✓

### Fixes applied to runner (minimal)

3 helper functions added to `backend/scripts/run_step4_matrix.py` (all new code, no existing logic modified):
1. `_detect_duplicate_question_ids(questions) -> set[str]` — returns set of question texts appearing >1 time
2. `_compute_dataset_manifest_hash(questions) -> str` — SHA-256 of ordered question texts
3. `_check_manifest_consistency(state, questions) -> bool` — compares stored hash vs current; first-run always True

Existing functions (`_init_checkpoint`, `_load_checkpoint`, `_save_checkpoint`, `_verify_done_config`, `_load_partial_output`, `IncrementalResultWriter`) were already correct and required no modification.

### Open issues / blocked items

None for Phase 1. All 16 required test scenarios pass.

---

---

## Phase 2 — Workstream B: Medical QA Evaluation + Statistics + Leakage Checks

**Status:** IMPLEMENTED AND TESTED  
**Target env:** CPU (`.venv\Scripts\python.exe`)  
**Completed:** 2026-10-17

### Files changed

| File | Action | Description |
|------|--------|-------------|
| `backend/evaluation/metrics.py` | MODIFIED | Added 14 new metrics with docstrings + denominators: accuracy, exact_match, rouge_l, recall_at_k, precision_at_k, mrr, ndcg_at_k, ece, brier_score, risk_coverage_curve, abstention_precision, abstention_recall, false_abstention_rate, false_acceptance_rate |
| `backend/evaluation/statistics.py` | CREATED | Wilson CI, McNemar (exact+chi2), paired bootstrap, Holm correction, format_result |
| `backend/evaluation/split_builder.py` | CREATED | build_splits (seeded, stratified), check_near_duplicates, check_question_leakage, check_answer_leakage, check_prompt_leakage, IndexLeakageGuard/IndexLeakageError |
| `backend/evaluation/annotation_schema_docs.md` | CREATED | Human annotation schema documentation (fields, valid values, tier rules) |
| `backend/evaluation/annotation_schema.jsonl` | CREATED | One example annotation record (annotation_status: pending) |
| `backend/results/corrected_metrics_20261017_011041.json` | CREATED | Historical metric corrections; originals untouched |
| `backend/tests/test_workstream_b_metrics.py` | CREATED | 96 unit tests for all 14 metrics + 5 statistics functions |
| `backend/tests/test_workstream_b_leakage.py` | CREATED | 35 unit tests for split builder and leakage checks |

### Commands run

```
d:\Ragmed_2\.venv\Scripts\python.exe -m pytest backend/tests/test_workstream_b_metrics.py backend/tests/test_workstream_b_leakage.py -v
  → 130 passed, 1 skipped, 0 failed (68s)
  → 1 skipped: TestIndexLeakageGuard::test_rejects_symlink_resolving_to_production
    (skipped because Windows symlink creation requires elevated privileges;
     the guard logic was verified by code inspection to call Path.resolve() before comparison)

git -C d:\Ragmed_2 add <8 files>
git -C d:\Ragmed_2 commit -m "Phase 2: Workstream B metrics, statistics, leakage checks, split builder"
  → Commit: dfb0ba1d
```

### Test counts

| Test file | Passed | Failed | Skipped |
|-----------|--------|--------|---------|
| test_workstream_b_metrics.py | 95 | 0 | 0 |
| test_workstream_b_leakage.py | 35 | 0 | 1 |
| **Total** | **130** | **0** | **1** |

### Sub-item status

**2a — Metric implementations:** IMPLEMENTED AND TESTED  
Evidence tier: SOFTWARE TEST  
All 14 metrics with docstrings, denominator definitions, and ≥2 unit tests each (6 tests for accuracy through NDCGAtK; 4 for calibration metrics). Hand-computable expected values are documented inline.

**2b — Historical metric corrections:** IMPLEMENTED AND TESTED  
Evidence tier: SOFTWARE INSPECTION  
File: `backend/results/corrected_metrics_20261017_011041.json`  
Findings:
- `step4_full-pipeline_pilot25_20261007_021046.json`: `accuracy=0.52` is wrong — produced by the old `compute_accuracy` bug (`"" in truth` always True for abstentions). 13/25 abstentions counted as "correct". Corrected value: `0.0` (0 non-empty predictions match their reference).
- `step4_full-pipeline_pilot25_20261003_234006.json`: mock-mode run (all answers are `[MOCK_GENERATED_ANSWER...]`); `llm_mode=None`. Not a valid benchmark result; no numeric correction needed.
- All 6 canonical n30 results: `accuracy` is aliased to `rouge_l_f1` (no bug, labelling issue only). No correction needed.
- `citation_support_rate` denominator inconsistency between JSON and ablation summary: labelling issue, no numeric correction.
- Original files untouched.

**2c — Statistics utilities:** IMPLEMENTED AND TESTED  
Evidence tier: SOFTWARE TEST  
Functions: `wilson_ci`, `mcnemar_test` (exact binomial when discordant < threshold, chi-squared with continuity correction otherwise), `paired_bootstrap` (percentile method, seeded), `holm_correction` (step-down, returns both rejected flags and adjusted p-values), `format_result`.

**2d — Split builder + leakage checks:** IMPLEMENTED AND TESTED  
Evidence tier: SOFTWARE TEST  
`build_splits`: seeded stratified split, raises ValueError on insufficient data or duplicate IDs.  
`check_near_duplicates`: Jaccard token similarity on question texts.  
`check_question_leakage`, `check_answer_leakage`: Jaccard similarity vs corpus texts.  
`check_prompt_leakage`: substring containment vs prompt templates.  
`IndexLeakageGuard`: resolves symlinks via Path.resolve(); rejects production root, parent paths, and non-eval_split subpaths; accepts eval_split and external paths.

**2e — Human annotation schema:** IMPLEMENTED  
Evidence tier: N/A (schema documentation only)  
`annotation_schema_docs.md`: documents all fields, valid values, tier rules, separation from model judgment.  
`annotation_schema.jsonl`: one example record with `annotation_status: pending`.  
Human-validated metrics = "pending annotation" until annotation_status = "complete".  
Schema is explicitly separated from AUTOMATED MODEL JUDGE outputs.

### Open issues / blocked items

None for Phase 2. All required items implemented and tested.  
Note: 500-question eval split (100 dev / 150 cal / 250 test) remains NOT BUILT — split_builder.py is implemented and tested, but actual dataset assembly is blocked pending dataset access (see BLOCKED.md).

---

## Phase 3 — Workstream A: Evidence Verification, Citation Faithfulness, Contradiction

**Status:** NOT IMPLEMENTED  
**Target env:** CPU (`.venv\Scripts\python.exe`) for tests + mocks  
**Files to create/modify:**

| File | Action | Description |
|------|--------|-------------|
| `backend/app/agents/state.py` | MODIFY | Add index_id, retrieval_iteration per-evidence-item; add pipeline_version; backward compatible |
| `backend/app/rag/pipeline.py` | MODIFY | Embed index_id in evidence dict |
| `backend/app/agents/orchestrator.py` | MODIFY | Carry retrieval_iteration into each evidence item |
| `backend/app/rag/verification/claim_verifier.py` | CREATE | Claim segmenter → citation mapper → ID check → quote fidelity → NLI (CPU, small model) |
| `backend/tests/test_claim_verifier.py` | CREATE | Unit + regression tests (sildenafil/Prolia MUST fail old code, pass new) |
| `backend/app/agents/evidence_reranking.py` | MODIFY | Strengthen unknown-marker rejection; bounded retry on malformed JSON |
| `backend/tests/test_evidence_reranking.py` | MODIFY | Add sildenafil/Prolia regression |
| `backend/evaluation/sufficiency_evaluator.py` | CREATE | Evidence sufficiency evaluator |

**Gate:** Provenance tests pass; sildenafil regression: FAIL on old code, PASS on new.

---

## Phase 4 — Workstream C: Agent-Contribution Analysis, Caching, Latency

**Status:** NOT IMPLEMENTED  
**Target env:** CPU (`.venv\Scripts\python.exe`)  
**Files to create/modify:**

| File | Action | Description |
|------|--------|-------------|
| `backend/evaluation/agent_contribution.py` | CREATE | Parse n30 result JSONs; per-agent latency breakdown; delta analysis |
| `backend/tests/test_agent_contribution.py` | CREATE | Tests for contribution analysis |
| `backend/app/agents/orchestrator.py` | MODIFY | Verify per-node duration_ms is non-zero |
| `backend/evaluation/cache_analysis.py` | CREATE | Cache hit/miss analysis (if cache exists) |

**Gate:** Agent contribution tests pass; latency breakdown matches raw elapsed_ms within tolerance.

---

## All Files to Change — Master List

```
CREATE:
  docs/audit_20261017_005421.md                          ← Phase 0 ✅
  docs/progress/PROGRESS.md                              ← Phase 0 ✅
  docs/progress/BLOCKED.md                               ← Phase 0 ✅
  backend/tests/test_checkpoint_resume.py                ← Phase 1
  backend/evaluation/metrics.py (additions)              ← Phase 2
  backend/tests/test_metrics_unit.py                     ← Phase 2
  backend/evaluation/statistics.py                       ← Phase 2
  backend/tests/test_statistics.py                       ← Phase 2
  backend/evaluation/dataset_manifest.py                 ← Phase 2
  backend/tests/test_dataset_manifest.py                 ← Phase 2
  backend/evaluation/human_annotation_schema.py          ← Phase 2
  backend/evaluation/recomputed_metrics/ (dir + files)   ← Phase 2 (if needed)
  backend/app/rag/verification/claim_verifier.py         ← Phase 3
  backend/tests/test_claim_verifier.py                   ← Phase 3
  backend/evaluation/sufficiency_evaluator.py            ← Phase 3
  backend/evaluation/agent_contribution.py               ← Phase 4
  backend/tests/test_agent_contribution.py               ← Phase 4
  backend/evaluation/cache_analysis.py                   ← Phase 4

MODIFY (minimal):
  backend/scripts/run_step4_matrix.py                    ← Phase 1 (only if tests fail)
  backend/app/agents/state.py                            ← Phase 3
  backend/app/rag/pipeline.py                            ← Phase 3
  backend/app/agents/orchestrator.py                     ← Phase 3 + Phase 4
  backend/app/agents/evidence_reranking.py               ← Phase 3
  backend/tests/test_evidence_reranking.py               ← Phase 3

NEVER MODIFY:
  backend/data/faiss_index/                              ← PROTECTED
  backend/prompts/v1/*.txt                               ← PROTECTED
  backend/data/results/ (existing files)                 ← PROTECTED
  backend/data/checkpoints/step4_matrix_state.json       ← PROTECTED
```

---

## BLOCKED Items (also in BLOCKED.md)

| Item | Reason | What is needed |
|------|--------|----------------|
| MedRGB metrics | Dataset absent from repo | Obtain MedRGB dataset with licence verification |
| Eval index leakage smoke test | Read-only audit; smoke test not run | Run `python -m scripts.run_step4_benchmark --smoke-test` with GPU venv |
| 500-question eval split | Split not built; requires BioASQ + MedQA data + seeded split builder | Phase 2 split builder + dataset access |
| NLI model download | CPU NLI model not present | Download `cross-encoder/nli-MiniLM-L2-v2` before Phase 3 (not during benchmark) |
| Human annotation labels | Require human review | Phase 2 creates schema; labels = "pending annotation" until human review done |
