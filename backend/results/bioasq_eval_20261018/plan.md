# Implementation Plan — Fix BioASQ Retrieval Evaluation and Confidence Calibration (RAGMedX-MA)

Derived from the **APPROVED** design at `d:\Ragmed_2\backend\results\bioasq_eval_20261018\design.md`
(review verdict APPROVED, 0 HIGH / 0 MEDIUM / 3 NITs in `design-review.md`). This plan sequences the
work faithfully from that design; it does not re-decide the architecture. Every decision point the
design already resolved is treated as fixed.

## Absolute paths (locked)

- Repo root: `d:\Ragmed_2`
- Backend dir: `d:\Ragmed_2\backend`
- Python: `d:\Ragmed_2\.venv\Scripts\python.exe`
- Isolated retrieval-eval index root: `d:\Ragmed_2\backend\data\indexes\bioasq_retrieval_eval`
- Deliverables root: `d:\Ragmed_2\backend\results\bioasq_eval_20261018`
- Env facts: `d:\Ragmed_2\backend\results\bioasq_eval_20261018\env_smoke.json`
- Integrity baseline: `d:\Ragmed_2\backend\results\bioasq_eval_20261018\integrity_baseline.json`

## Verified facts that shape this plan (grounded in code read during planning)

- `evaluation/metrics.py` has the exact functions the design reuses: `precision_at_k(retrieved_ids,
  relevant_ids, k)` (denominator always `k`), `recall_at_k(...)` (denominator `|relevant|`),
  `mrr(list[list[str]], list[list[str]])`, `ndcg_at_k(...)` (binary relevance, returns 0.0 when no
  relevant), `ece(confidences, correct_flags, n_bins=10)`, `brier_score(...)`,
  `risk_coverage_curve(...)`, `exact_match`, `compute_rouge_l`, `abstention_precision/recall`. **Reuse;
  do not reimplement.**
- `evaluation/split_builder.py` exposes `build_splits(dataset, dev_n, cal_n, test_n, seed=42,
  stratify_by=None)` → `(dev_split, cal_split, test_split)`, raising `ValueError` on insufficient data,
  duplicate ids, and sparse strata (stratified path runs an internal `_adjust` pass). It also exposes
  `IndexLeakageGuard` / `IndexLeakageError` which the eval scripts should use when opening any index path.
- `app/rag/embeddings/bge_embedding.py`: `BGEEmbedding(model_name="BAAI/bge-m3", normalize=True)`,
  `embed(texts, batch_size=8)` **hard-codes `max_length=512`**, mean pooling, L2-normalized, 1024-dim.
  Alias `BGEEmbedder = BGEEmbedding`. **Instantiate `BGEEmbedding()` directly; do NOT read the stale
  `bge-small-en-v1.5` id from `app/config.py`.**
- `app/rag/vectorstore/faiss_store.py`: `FAISSVectorStore(dimension=1024)` (`IndexFlatIP`),
  `add(embeddings, chunks)`, `search(query_embedding, top_k)` → `list[(MedicalChunk, score)]` high→low,
  `save(dir)`, classmethod `load(dir)` (derives `d` from the index, asserts `index.ntotal ==
  len(chunks)`). Position in `store.chunks` == FAISS vector id. Re-exported at
  `app/retrieval/vector_store.py`.
- `app/rag/retrievers/dense_retriever.py`: `DenseRetriever(embedder, vector_store)` and classmethod
  `DenseRetriever.load(directory, embedder)`; `retrieve(query, top_k)` raises `TypeError` on non-str,
  `ValueError` on empty query or `top_k<=0`. **Construct with the `vector_store=` keyword (NOT `store=`).**
- `app/rag/chunking/models.py`: `MedicalChunk` is a pydantic model with `id`, `document_id`, `content`,
  `title`, `section`, `source` (default `"PMC"`), `journal`, `publication_year`, `metadata: dict`.
- `app/agents/confidence_calibration.py`: `CONFIDENCE_THRESHOLD = 0.65`, weights `0.25 / 0.30 / 0.25 /
  0.20`. **Never modify this file.**
- Inputs confirmed on disk: `backend/data/datasets/bioasq_eval_split/eval_questions.json`;
  `backend/data/faiss_index/eval_split/chunks.pkl`;
  `RAGMedXMA_Chapter5_Validated_Results/{predictions.jsonl, confidence_abstention.jsonl,
  retrieval_results.jsonl}` at the **repo root**;
  `backend/results/phase5_benchmark/per_question/phase5-full-pipeline.jsonl` (in-backend mirror).
- **Correction to a step-prompt detail:** `training13b.json` is at
  `d:\Ragmed_2\backend\data\datasets\bioasq\training13b.json` (NOT `backend/data/datasets/`). It is only
  needed for the **default-OFF** `--add-training-distractors` flag; the primary corpus is
  `eval_questions.json` snippets only (design §6.1), so the main path does not read it. The scripts must
  anchor this path on `BACKEND_DIR / "data" / "datasets" / "bioasq" / "training13b.json"` and treat a
  missing file as "distractors unavailable", never fatal.
- `backend/evaluation/bioasq_eval/` does **not** exist yet (create it). `backend/data/indexes/` exists.
- Test config: `backend/pytest.ini` has `pythonpath = app .`, `testpaths = tests`, `asyncio_mode =
  auto`, and an `integration` marker. New tests live under `backend/tests/` so `testpaths` picks them up;
  model-loading tests are marked `@pytest.mark.integration`.

## Cross-cutting conventions every new script must follow (design §2, §9)

- First executable lines of every module under `backend/evaluation/bioasq_eval/`:
  `BACKEND_DIR = Path(__file__).resolve().parents[2]` (→ `D:\Ragmed_2\backend`),
  `REPO_ROOT = Path(__file__).resolve().parents[3]` (→ `D:\Ragmed_2`),
  then `sys.path.insert(0, str(BACKEND_DIR))` before any `app.*` / `evaluation.*` import. **Never resolve
  a path against the process cwd.** Run modules from `backend/` as `python -m evaluation.bioasq_eval.<x>`.
- Seed = `42` everywhere (`random`, `numpy`, split builder, bootstrap).
- Atomic writes: write `*.tmp` then `os.replace`. JSONL outputs append per record, keyed so a resumed run
  skips completed items. Each phase writes a `manifests/<phase>.manifest.json` with seed, config dict +
  its SHA-256 `config_hash`, input file SHA-256s, embedding model id, package versions, UTC start/end,
  completed-item counts.
- One `BGEEmbedding()` load per process. CPU-only; no CUDA assumptions.
- Logging via `app.utils.logger.logger`.
- **Never** modify/rebuild/overwrite/delete/re-embed `backend/data/faiss_index/` (production) or its
  `eval_split/` subdir, and **never** modify `backend/data/datasets/bioasq_eval_split/`. All writes go to
  `backend/data/indexes/bioasq_retrieval_eval/` and `backend/results/bioasq_eval_20261018/` only.
- **Never fabricate** metrics, labels, relevant docs, confidence values, or test outcomes. Where data is
  insufficient, report "insufficient n" / "no positive class" / "provisional / degenerate" and report
  only what is measurable.

---

# Implementation Plan

- [ ] 1. Create the package scaffold and the integrity module.
      Create `backend/evaluation/bioasq_eval/__init__.py` (empty) and
      `backend/evaluation/bioasq_eval/integrity.py` exposing `compute_sha256(path)`,
      `load_baseline()` (reads `integrity_baseline.json`), `check_pre_run()` and `check_post_run()` that
      recompute SHA-256 of the four protected files and compare to baseline, plus
      `check_eval_split_membership()` (asserts eval_split distinct-PMID set + chunk count == 58,173
      unchanged — load via `FAISSVectorStore.load` read-only). Results written to
      `results/bioasq_eval_20261018/integrity_check.json` with `pre_run_ok`/`post_run_ok`; any mismatch
      raises `RuntimeError` and marks the run INVALID.
      Files: `backend/evaluation/bioasq_eval/__init__.py`,
      `backend/evaluation/bioasq_eval/integrity.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.integrity
      --check-only` prints `pre_run_ok: true` for all four files (hashes already match baseline) and does
      not alter any protected file. Re-run the SHA-256 of the four protected files before and after;
      confirm identical.

- [ ] 2. Create shared helpers for PMID normalization, chunk→article dedup, and atomic/resumable IO.
      Add `backend/evaluation/bioasq_eval/common.py` with: `normalize_pmid(value)` →
      `re.search(r"(\d+)$", str(value)).group(1)` or `None` (counts/returns unparsable); `canonical_pmid
      (chunk)` using `chunk.metadata["pmid"]` with `source`-parse fallback (and a `pmid_fallback_used`
      counter); `dedup_articles(ranked_chunks)` collapsing a ranked `MedicalChunk` list to a rank-ordered
      unique article (PMID) list preserving first occurrence; `atomic_write_json(path, obj)` and
      `JsonlCheckpoint` (append-with-flush, load completed keys, skip duplicates); `config_hash(cfg)`
      (SHA-256 of canonical JSON); `write_manifest(phase, cfg, inputs, extra)`.
      Files: `backend/evaluation/bioasq_eval/common.py`
      Verify: unit tests in step 11 (`test_common.py`) exercise `normalize_pmid`, `dedup_articles` (incl.
      multi-chunk-per-article collapse and rank preservation), and `JsonlCheckpoint` resume semantics;
      run `d:\Ragmed_2\.venv\Scripts\python.exe -m pytest tests/test_common.py -q` from `backend/` — all
      pass.

- [ ] 3. Implement Phase A diagnostic (no generation LLM).
      Add `backend/evaluation/bioasq_eval/diagnose.py` producing
      `results/bioasq_eval_20261018/diagnostics/diagnostic_report.json` and `.md`. Read-only inputs:
      `eval_questions.json`, `faiss_index/eval_split/chunks.pkl` (via `FAISSVectorStore.load`), optionally
      production `chunks.pkl` and the prior `RAGMedXMA_Chapter5_Validated_Results/retrieval_results.jsonl`.
      Report: `n_queries` (expect 250), `n_queries_with_gold`, `n_gold_pmids_raw`,
      `n_gold_pmids_unique_normalized` (expect 2,394), `unparsable_pmids` sample, `eval_split_chunks`
      (expect 58,173) and `eval_split_distinct_pmids`, `gold_pmids_matched_in_eval_split` (**expect 0**) +
      pct (confirming the deliberate exclusion, citing the prior run's `metric_reason`), optional
      production cross-check pct, the §4 figure reconciliation (name the four candidate populations and
      the count each yields: 2,394/250 full, 172/40 phase5 subset, 172 Σn_gold, 145 BioASQ-13; state 789
      is not reproducible), `missing_pmid_examples` (≤25, each tagged `not_in_local_snippets` /
      `normalization_variant` / `fetch_skipped`), an explicit **"missing gold doc (coverage gap)" vs
      "retrieval failure"** distinction section, and a 20-sample PMID→chunk round-trip traceability check.
      The per-corpus availability counts (`n_queries_with_at_least_one_available_gold`,
      `n_queries_with_all_gold_available`) against the NEW retrieval-eval corpus are filled in by step 5
      after the index exists (diagnose exposes a `--corpus <index_dir>` mode to compute them). Load the
      embedding model only when `--roundtrip` is passed.
      Files: `backend/evaluation/bioasq_eval/diagnose.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.diagnose`
      exits 0 and writes `diagnostics/diagnostic_report.json` with `n_queries==250`,
      `n_gold_pmids_unique_normalized==2394`, and `gold_pmids_matched_in_eval_split==0`; confirm those
      three fields in the JSON.

- [ ] 4. Implement Phase B isolated index builder.
      Add `backend/evaluation/bioasq_eval/build_eval_index.py` building the versioned index at
      `backend/data/indexes/bioasq_retrieval_eval/v1/{index.faiss, chunks.pkl, manifest.json}` from
      `eval_questions.json` **snippets only**. Build `MedicalChunk`s with
      `id=f"bioasq_{qid}_s{si}_chunk_{j}"`, `document_id=f"bioasq_{qid}_s{si}"`,
      `content = "[{beginSection}] " + snippet.text` (section prefix ON, `endSection` ignored — recorded
      verbatim in manifest), `source=f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"`, `metadata={pmid,
      pmid_url, begin_section, end_section, offset_begin, offset_end, question_id}`. Dedup raw snippets by
      `(pmid, normalized_text)` before embedding. One `BGEEmbedding()` load; embed at `batch_size=16`
      (override the embedder default of 8; record 16 in manifest); `FAISSVectorStore(dimension=1024)`;
      `add`; `save` to `v1/`. Default-OFF flags: `--add-training-distractors N` (reads
      `backend/data/datasets/bioasq/training13b.json`; missing file → "distractors unavailable", not
      fatal), `--fetch-missing` (NCBI E-utilities ≤3 req/s, cached under `pubmed_cache/`, provenance per
      record), `--reranker`, `--force-rebuild` (writes `v2/`, never overwrites `v1`). Idempotent: skip
      rebuild if `v1/manifest.json` has a matching `config_hash`. Verification (fatal on failure, written
      to `build_report.json`): `index.ntotal == len(chunks)`, `index.d == 1024`, distinct-PMID and chunk
      counts recorded, 10-query gold round-trip (own gold PMID appears in top-5), 10-sample
      `metadata["pmid"]` matches `source`-parsed PMID. Dense-only. Use `IndexLeakageGuard` is not needed
      here (writing a new dir), but the integrity pre-check (step 1) MUST pass before building.
      Files: `backend/evaluation/bioasq_eval/build_eval_index.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m
      evaluation.bioasq_eval.build_eval_index` creates `v1/index.faiss`, `v1/chunks.pkl`,
      `v1/manifest.json`, and a `build_report.json` whose `ntotal_equals_len_chunks` is true and `d==1024`;
      confirm the four protected-file SHA-256s are unchanged afterward (step 1 post-check).

- [ ] 5. Implement Phase C retrieval evaluation (reuse `evaluation/metrics.py`).
      Add `backend/evaluation/bioasq_eval/run_retrieval_eval.py`. One `BGEEmbedding()` load; load the `v1`
      index via `DenseRetriever.load(index_dir, embedder=BGEEmbedding())`; for each of the 250 questions
      `retriever.retrieve(question["body"], top_k=5)`, map chunks→articles via `dedup_articles` (step 2),
      compute per-query `precision_at_k/recall_at_k/ndcg_at_k` against the full normalized `gold(q)` and
      collect pairs for batch `mrr`. Compute `available(q) = gold(q) ∩ corpus_pmids`. Emit the **three
      views** exactly as design §7.3 (one metric computation, denominator differs): corpus-coverage
      (`coverage@≥1`, `coverage@all` over 250), end-to-end (denominator all 250; uncovered-gold queries
      score 0), retrieval-effectiveness (denominator = covered queries only). Print counts of
      uncovered-gold and partially-covered queries. **Mere PMID presence is never a hit — only top-5
      article-rank membership counts.** Resume key = the bare eval `id` (written under key `question_id`
      for output uniformity). Outputs: `results/bioasq_eval_20261018/retrieval/retrieval_metrics.json`
      (all three views + per-query table with `gold`, `available`, `covered`, retrieved articles, each
      metric) and `.../retrieval/per_query.jsonl` (atomic incremental append, resumable). Also feed the
      corpus-availability counts back into the diagnostic (invoke `diagnose --corpus v1` or write the two
      counts into a side file the diagnostic reads).
      Files: `backend/evaluation/bioasq_eval/run_retrieval_eval.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m
      evaluation.bioasq_eval.run_retrieval_eval` writes `retrieval/retrieval_metrics.json` with the three
      named views and 250 per-query rows in `per_query.jsonl`; re-running skips completed queries
      (resume) and produces identical metrics. Unit coverage in step 11.

- [ ] 6. Implement Phase D calibration source loader + label/join logic.
      Add `backend/evaluation/bioasq_eval/calibrate_confidence.py` (part 1: data). Resolve the three
      calibration-source candidates against `REPO_ROOT`/`BACKEND_DIR` in fixed precedence
      (`REPO_ROOT/RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` →
      `.../confidence_abstention.jsonl` →
      `BACKEND_DIR/results/phase5_benchmark/per_question/phase5-full-pipeline.jsonl`), logging each
      probed absolute path (exists/not-found). Missing all → exit non-zero `CALIBRATION_SOURCE_MISSING`
      (never a silent LLM run). Select the first that resolves; cross-check every other resolved
      candidate (restricted to `phase5-full-pipeline`) for identical `n_rows`, `set(question_id)`, and
      per-id `correctness`/`abstained`; mismatch → fatal `CALIBRATION_SOURCE_MISMATCH`; record
      `source_consistency` in the manifest. Primary correctness label = dataset `correctness`
      (`token_containment>=0.6`). Exclude `source=="pubmedqa"` rows and abstentions. Graded secondary
      join: strip `<source>::` from `question_id` and look up `exact_answer`/`ideal_answer`/`type` in
      `eval_questions.json`. Note the §8.2 NIT corrections verbatim in the manifest: the timestamped
      `phase5_20261017_131123` dir and `backend/results/chapter5_package/predictions.jsonl` both exist but
      are **intentionally excluded** from the precedence list (non-canonical/timestamped mirror).
      Files: `backend/evaluation/bioasq_eval/calibrate_confidence.py`
      Verify: from `backend/`, running the loader step prints the resolved primary source and the realized
      counts (expect for full-pipeline: 40 rows → 13 BioASQ → 2 non-abstained → 0 correct); confirm
      `source_consistency` recorded. Unit coverage of the id-join + pubmedqa exclusion in step 12.

- [ ] 7. Implement Phase D calibration metrics, degenerate-data policy, and the graded-evaluator gate.
      Extend `calibrate_confidence.py` (part 2). Treat raw Agent-7 confidence as **uncalibrated**. Compute
      the §8.4 branch deterministically on `(N, P)`: if `P==0` or `N-P==0` (the verified current state,
      N≈2 P=0) → DEGENERATE path: report raw-heuristic ECE/Brier over labelled rows, confidence histogram,
      released/withheld at 0.65, empirical selective accuracy at 0.65, and an explicit statement that 0
      correct answers means 0.65 can be neither confirmed nor refuted — all labelled
      "DEGENERATE — single-class / insufficient data, provisional, not a calibration claim" with exact N,
      P, source config named. If `0<N<15`: fit map on all rows (both classes present), report raw-vs-mapped
      ECE/Brier flagged provisional, no threshold claim, no CIs. The per-type graded evaluator (§8.3,
      yes/no polarity; factoid/list exact-match/containment with ≥50% list threshold; summary ROUGE-L F1 ≥
      0.30 via `metrics.compute_rouge_l`) and its acceptance gate (automated-vs-manual agreement ≥ 0.85 on
      a seed-42 sample written to `calibration/manual_review_sample.jsonl`; `--manual-labels` optional;
      otherwise all numbers provisional) are wired but expected to resolve to "provisional by default" on
      this data. ECE uses `metrics.ece` (10 equal-width bins); add bin counts + acc-vs-conf to the report.
      Finite/in-[0,1] checks on all confidences (clamp+warn out-of-range). Outputs:
      `calibration/calibration_report.json` + `.md`. **Never fabricate a positive label.**
      Files: `backend/evaluation/bioasq_eval/calibrate_confidence.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m
      evaluation.bioasq_eval.calibrate_confidence` writes `calibration/calibration_report.json` whose
      `data_regime` is `"DEGENERATE"` with `N`/`P` recorded and raw ECE/Brier present; confirm no
      calibration/threshold claim is emitted. Unit coverage of the branch selector on synthetic (N,P) in
      step 12.

- [ ] 8. Implement Phase D/E split + method selection + threshold protocol (reached only when N≥15 and
      both classes present).
      Extend `calibrate_confidence.py` (part 3) and write
      `results/bioasq_eval_20261018/calibration/partitions.json` and
      `results/bioasq_eval_20261018/threshold_evaluation.json`. Partitions via `build_splits(dataset,
      dev_n=round(0.6N), cal_n=round(0.2N), test_n=N-dev_n-cal_n, seed=42, stratify_by="type")` unpacked as
      `fit_split, select_split, test_split`; assert the three id-sets are pairwise disjoint and union ==
      labelled set; record realized sizes. Per-stratum sparsity guard: pre-check `max(1, round(frac*n))`
      per type per split and fall back to `stratify_by=None` if any stratum too sparse (record
      `fallback_reason`); the pre-check is advisory — the authoritative guard is a `try/except ValueError`
      wrapper around `build_splits` that routes to the §8.4 provisional path. Fit **both** Platt (logistic)
      and isotonic on `fit` only; select lower fit-ECE (tie-break lower fit-Brier) — sklearn if importable
      else NumPy logistic fallback (record which). Report ECE/Brier **before (raw) and after (mapped)** on
      fit/select/test. Per-partition single-class guard: if `select` single-class → no data-driven
      threshold, evaluate only 0.65 on test, provisional; if `test` single-class → no verdict, §8.4
      labelling. Threshold grid 0.30–0.90 step 0.05 on calibrated confidence; objective = max coverage s.t.
      selective accuracy ≥ 0.80 on `select`, smallest-threshold tie-break; evaluate selected + incumbent
      0.65 **once** on untouched `test`; report coverage, selective accuracy, error-rate-among-answered,
      abstention rate, risk-coverage (`metrics.risk_coverage_curve`), ECE/Brier on test; 0.65 "validated"
      iff test sel.acc ≥ 0.80 & coverage ≥ 0.50; bootstrap 95% CIs (1,000 resamples, seed 42) only if test
      n ≥ 30. A "test opened once" guard enforces the single final evaluation.
      Files: `backend/evaluation/bioasq_eval/calibrate_confidence.py`
      Verify: on the current data this path is NOT reached (DEGENERATE, N≈2) — confirm
      `threshold_evaluation.json` records `status: "not_evaluable"` with the reason and the exact N/P, and
      that no split is produced. The split/select/threshold **logic** is validated by the synthetic unit
      tests in step 12 (disjoint/union assertion, sparse-stratum fallback, method selection,
      threshold-objective, single-class guards, "test opened once").

- [ ] 9. Implement the Phase F resumable execution driver.
      Add `backend/evaluation/bioasq_eval/run_all.py` orchestrating: integrity pre-check → diagnose →
      build index → retrieval eval → calibration → integrity post-check, with stable per-query ids +
      config hashes, skip-valid-completed-results, reject incompatible/incomplete artifacts (config-hash
      mismatch → treat as not-done), atomic writes, bounded-retry (e.g. 3×) for transient failures
      (PubMed fetch, embed batch), never silently drop a failed query (record it with a failure reason),
      and `--diagnostic-only` plus `--no-model` modes that run without the expensive generation model
      (diagnose + reuse-based calibration). Writes `results/bioasq_eval_20261018/manifest.json`
      (top-level: phases run, config hashes, input versions, timestamps, completed counts, integrity
      pre/post status) and a reproducible rerun command string.
      Files: `backend/evaluation/bioasq_eval/run_all.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.run_all
      --no-model` runs diagnose + reuse-calibration end-to-end, writes `manifest.json` with
      `integrity.pre_run_ok` and `integrity.post_run_ok` true; a second run reports all phases skipped as
      already-complete (resume) and leaves outputs byte-identical for deterministic phases.

- [ ] 10. Protected-index + eval-split integrity tests.
      Add `backend/tests/test_bioasq_integrity.py`: production index + eval_split `index.faiss`/`chunks.pkl`
      SHA-256 match `integrity_baseline.json` (checksum unchanged); eval_split membership (distinct-PMID
      set + chunk count == 58,173) unchanged; `IndexLeakageGuard` rejects the production root and accepts
      the eval_split / isolated-index paths. These run read-only and must never write to protected dirs.
      Files: `backend/tests/test_bioasq_integrity.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m pytest tests/test_bioasq_integrity.py
      -q` passes; confirm the four protected files' SHA-256 are unchanged after the test run.

- [ ] 11. Retrieval-metric + PMID-mapping + common-helper unit tests (small, hand-verifiable).
      Add `backend/tests/test_bioasq_retrieval_metrics.py` and `backend/tests/test_bioasq_common.py`
      covering: all-5-irrelevant (P=R=MRR=nDCG=0); first-result relevant; relevant at rank 5; multiple
      relevant PMIDs; **multiple chunks mapping to one PMID collapse to one article** (dedup invariant);
      gold absent from corpus (conditional vs end-to-end divergence); dedup preserves rank order; PMID
      normalization incl. unparsable; `JsonlCheckpoint` resume (no duplication/corruption, same-config
      rerun consistent). These do not load the model.
      Files: `backend/tests/test_bioasq_retrieval_metrics.py`, `backend/tests/test_bioasq_common.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m pytest
      tests/test_bioasq_retrieval_metrics.py tests/test_bioasq_common.py -q` — all pass.

- [ ] 12. Calibration-logic unit tests (hand-computed, no model, leakage guards).
      Add `backend/tests/test_bioasq_calibration.py` covering: ECE/Brier vs hand-computed small arrays;
      invalid-score (out-of-[0,1]) and missing-label handling; the §8.4 branch selector on synthetic
      `(N,P)` (DEGENERATE / <15 / 15–40 / ≥40); phase5-id→eval-id join incl. pubmedqa exclusion; per-type
      correctness logic (table-driven); `build_splits` slot-mapping + disjoint/union assertion +
      sparse-stratum fallback on synthetic typed data; method-selection rule on synthetic score/label
      arrays with a known-better map; threshold-objective + smallest-threshold tie-break on a synthetic
      `select` set; **calibration never uses test labels** and **threshold selection never uses final
      test outcomes** (assert the test partition is untouched until the single final evaluation).
      Files: `backend/tests/test_bioasq_calibration.py`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m pytest
      tests/test_bioasq_calibration.py -q` — all pass.

- [ ] 13. Run the full test suite and capture the test report.
      Run the new BioASQ tests and confirm the pre-existing suite still passes (no regressions in reused
      modules). Capture output to `results/bioasq_eval_20261018/test_report.txt`.
      Files: `backend/results/bioasq_eval_20261018/test_report.txt`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m pytest tests/test_bioasq_*.py -q`
      passes with 0 failures; the captured `test_report.txt` shows the pass summary.

- [ ] 14. Execute the pipeline and produce the consolidated honest report + manifest.
      Run `run_all.py --no-model` (reuse-based; the data is DEGENERATE so no LLM is needed) to populate
      `diagnostics/`, `retrieval/`, `calibration/`, `threshold_evaluation.json`, `integrity_check.json`,
      and `manifest.json`, then write `results/bioasq_eval_20261018/evaluation_report.md` consolidating:
      Phase-A diagnosis (789/2,394/172/250 reconciliation; coverage-gap vs retrieval-failure), Phase-B
      index coverage (honest < 100% snippet-only coverage + forced bge-m3 / batch_size=16 / section-prefix
      deviations), Phase-C three retrieval views, Phase-D DEGENERATE calibration outcome (raw ECE/Brier +
      why 0.65 cannot be validated from this data), Phase-E threshold `not_evaluable` status, all
      limitations stated plainly, and the exact reproducible rerun command. Confirm integrity post-check
      passed (run marked VALID only if pre and post both OK).
      Files: `backend/results/bioasq_eval_20261018/evaluation_report.md`,
      `backend/results/bioasq_eval_20261018/manifest.json`
      Verify: from `backend/`, `d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.run_all
      --no-model` exits 0; `integrity_check.json` has `pre_run_ok` and `post_run_ok` true; the four
      protected-file SHA-256s equal `integrity_baseline.json`; `evaluation_report.md` and `manifest.json`
      exist and reference the measured diagnostic/retrieval/calibration numbers (no fabricated values).

## Dependency ordering notes

- Items 1–2 are foundational (integrity + shared helpers) and must come first.
- Item 3 (diagnose) depends on 1–2; its corpus-availability counts are completed after item 4/5.
- Item 4 (build index) depends on 1–2 and must pass the integrity pre-check before building.
- Item 5 (retrieval) depends on 4 (the `v1` index) and reuses `evaluation/metrics.py`.
- Items 6→7→8 (calibration) build the same `calibrate_confidence.py` incrementally; each leaves the file
  importable and runnable. On the current data the pipeline stops at the DEGENERATE branch (item 7), and
  item 8's split/threshold logic is exercised only by unit tests.
- Item 9 (driver) depends on 3–8 existing.
- Items 10–12 (tests) can be written alongside the modules they cover; item 13 runs them all.
- Item 14 executes the pipeline and writes the final deliverables; it depends on everything above.

## Hard constraints (restated — must hold at every step)

1. Never modify/rebuild/overwrite/delete/re-embed `backend/data/faiss_index/` or
   `backend/data/datasets/bioasq_eval_split/`. Enforced by SHA-256 pre/post integrity checks (items 1,
   10, 14).
2. Separate partitions for retrieval-eval, calibration fitting, and final threshold validation; a
   threshold is never tuned on the examples used to report its final performance (items 8, 12).
3. Seed 42 everywhere; all artifacts isolated under the two write roots; reproducible rerun command.
4. Never fabricate metrics, labels, relevant docs, confidence values, or test outcomes; report
   insufficiency honestly (items 7, 8, 14).
