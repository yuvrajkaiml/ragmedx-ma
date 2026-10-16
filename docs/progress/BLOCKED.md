# RAGMedX-MA — Blocked Items

**Last updated:** 2026-10-17 00:54:21

---

## BLOCKED-001: MedRGB Dataset Absent

**What:** MedRGB evaluation dataset is required for Workstream B metrics.  
**Why blocked:** No MedRGB file, loader, or reference exists anywhere in the repo. The MIRAGE benchmark.json does not include MedRGB. `MedRGB metrics = NOT MEASURABLE`.  
**What is needed:** Obtain MedRGB dataset, verify licence (academic use only per arXiv:2411.09213), add loader. Until then, all MedRGB metrics are omitted from reports.

---

## BLOCKED-002: Eval Index Leakage Smoke Test Not Run

**What:** The benchmark smoke test (`python -m scripts.run_step4_benchmark --smoke-test`) should verify that the first retrieved PMID is NOT in the eval-question PMID set (i.e., the eval_split FAISS index does not contain eval-question abstracts).  
**Why blocked:** This audit was read-only; no GPU/LLM process was started. The eval_split index exists but leakage cannot be ruled out without running the canary.  
**What is needed:** Run smoke test on GPU venv before any Phase 5 benchmark run: `d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe -m scripts.run_step4_benchmark --smoke-test`

---

## BLOCKED-003: 500-Question Eval Split Not Built

**What:** Task parameters specify 500 questions (100 dev / 150 calibration / 250 test, stratified, fixed seed) from PubMedQA + MedQA-US test.  
**Why blocked:** Only BioASQ 250-question eval split currently exists. PubMedQA and MedQA-US are in MIRAGE benchmark.json (MCQ format) but not as a combined seeded split.  
**What is needed:** Phase 2 split builder with seeded stratified sampling, dedup/near-dup checks across splits, and index guard verification.

---

## BLOCKED-004: NLI Model Not Present

**What:** Claim verification (Workstream A) requires a small NLI model on CPU.  
**Why blocked:** No NLI model is downloaded in the current repo. `sentence-transformers` 5.7.0 is installed in both venvs and can load `cross-encoder/nli-MiniLM-L2-v2`.  
**What is needed:** Download `cross-encoder/nli-MiniLM-L2-v2` before Phase 3 begins. Must NOT be downloaded during a benchmark GPU run. Verify the model fits in CPU RAM and is not sharing VRAM with the generator.

---

## BLOCKED-005: Human Annotation Labels Pending

**What:** Abstention precision/recall, false abstention rate, false acceptance rate, and citation faithfulness metrics require human annotation (ground truth for whether each abstention was correct, whether each citation was faithful).  
**Why blocked:** No human annotation has been performed. Schema will be created in Phase 2.  
**What is needed:** Human reviewer to label a sample of model outputs using the annotation schema created in Phase 2.
