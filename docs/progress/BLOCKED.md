# RAGMedX-MA — Blocked Items

**Last updated:** 2026-10-17 (Phase 6 reconciliation)
**Branch:** `yuvraj/llm`

This file and the root `BLOCKED.md` are kept consistent. Each item carries a status:
RESOLVED, CARRY-FORWARD, or NOT MEASURABLE.

---

## BLOCKED-001: MedRGB Dataset Absent — NOT MEASURABLE (carry-forward)

**What:** MedRGB evaluation dataset is required for the MedRGB-specific metrics.
**Status:** NOT MEASURABLE. No MedRGB file, loader, or reference exists anywhere in
the repo; MIRAGE `benchmark.json` does not include MedRGB. All MedRGB metrics are
therefore omitted from every report and labelled NOT MEASURED.
**What is needed:** Obtain the MedRGB dataset, verify its licence (academic use only
per arXiv:2411.09213), add a loader. Until then MedRGB metrics stay NOT MEASURABLE.

---

## BLOCKED-002: Eval-Index Leakage Smoke Test — RESOLVED (Phase 5)

**What:** The benchmark canary must verify the first retrieved PMID is NOT in the
eval-question PMID set (i.e. the eval_split index does not contain eval-question
abstracts).
**Status:** RESOLVED. Ran on GPU in Phase 5:
`.venv-cu128-pilot\Scripts\python.exe -m scripts.run_step4_benchmark --smoke-test`
→ **CANARY RESULT: PASS**. First retrieved PMID `17128671` is NOT in the
eval-question set `{18039527, 31709696, 31722348}`. Recorded in
`backend/results/phase5_20261017_131123/environment.json`.

---

## BLOCKED-003: 500-Question Eval Split — RESOLVED (built); FULL SWEEP CARRY-FORWARD

**What:** A 500-question stratified seeded split (100 dev / 150 cal / 250 test).
**Status:** RESOLVED for the split build. `backend/scripts/build_phase5_split.py`
built a 750-question pool (BioASQ-eval 250 + MIRAGE PubMedQA 500) and produced
100/150/250 via `split_builder.build_splits(seed=42, stratify_by='source')`.
Leakage checks: near-duplicate counts (0.9) all 0; corpus question/answer leakage
(0.95) 0/0; IndexLeakageGuard PASS. Manifest + splits under
`backend/results/phase5_20261017_131123/`.
**CARRY-FORWARD:** the **full 500×8 GPU sweep** (~40+ h) exceeds the GPU budget and
was deliberately NOT run; a budget-bounded pilot (16 Q × 8 configs) was run instead.
Full-run command recorded in PROGRESS.md and `report.md`.
**Measurability note:** PubMedQA retrieval-quality (R@K/MRR/nDCG) is NOT MEASURED
(PubMed abstracts not in the BioASQ-built eval index); answer-quality only.

---

## BLOCKED-004: NLI Model Not Present — CARRY-FORWARD

**What:** Claim verification (Workstream A) needs a small CPU NLI model.
**Status:** CARRY-FORWARD. No NLI model is downloaded in the CPU venv; all NLI tests
use mocks and `enable_claim_verification` is default OFF. The code path is
implemented and tested against mocks (Phase 3).
**What is needed:** Download `cross-encoder/nli-MiniLM-L2-v2` (not during a benchmark
GPU run), confirm it fits in CPU RAM without sharing VRAM, then enable the flag.
Until then claim support is NOT MEASURED (would be AUTOMATED MODEL JUDGE tier once
the model is present).

---

## BLOCKED-005: Human Annotation Labels Pending — CARRY-FORWARD

**What:** Abstention precision/recall, false-abstention/acceptance rates, and
citation faithfulness need human ground-truth labels.
**Status:** CARRY-FORWARD. No human annotation performed. The schema exists
(`backend/evaluation/annotation_schema_docs.md` + `annotation_schema.jsonl`,
`annotation_status: pending`, Phase 2). These metrics are NOT MEASURED (would be
HUMAN-VALIDATED tier once labels exist); they are kept separate from any AUTOMATED
MODEL JUDGE output.
