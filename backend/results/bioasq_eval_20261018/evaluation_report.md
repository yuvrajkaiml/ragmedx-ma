# BioASQ Retrieval Evaluation & Confidence Calibration — Consolidated Report

Workflow `wf_f9d894c9ec5422e5`. Seed **42**. Embedding model **BAAI/bge-m3** (1024-dim).
Run status: **VALID** (integrity pre-check and post-check both passed; the four protected
files are byte-unchanged). All numbers below are measured from the artifacts in this
directory — none are fabricated. Where the available data cannot support a claim, that is
stated plainly rather than filled in.

Reproduce end to end from `d:\Ragmed_2\backend`:

```
d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.run_all --no-model
```

Individual phases (all run from `d:\Ragmed_2\backend`, no generation LLM required):

```
d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.diagnose --roundtrip --corpus data\indexes\bioasq_retrieval_eval\v1
d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.build_eval_index
d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.run_retrieval_eval
d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.calibrate_confidence
```

---

## Phase A — Diagnosis of the missing gold PMIDs

- Eval questions: **250** (all carry gold `documents[]`).
- Raw gold PMID entries normalized; **distinct normalized gold PMIDs = 2,394**.
- `eval_split` index: **58,173** chunks; **gold PMIDs matched in eval_split = 0 (0.0%)**.

The zero match is **by design**, not a bug. The `eval_split` FAISS index was built with a
deliberate cross-PMID leakage exclusion, so none of the gold documents appear in it. The
prior run's `retrieval_results.jsonl` records the same symptom for every query
(`metric_reason: NOT MEASURABLE … gold PMIDs not present / not indexed in eval_split`).
The fix is therefore **not** to touch `eval_split` but to build a separate, isolated
retrieval-eval index that contains the gold passages (Phase B).

### 789 / 2,394 / 172 / 250 reconciliation

The quoted "789" figure could **not** be reproduced from any artifact on disk. The four
candidate populations and the count each yields:

| Candidate population | Count |
|---|---|
| Distinct gold PMIDs over all 250 eval questions | **2,394** |
| Distinct gold PMIDs over the 40-question phase5 subset | **172** |
| Sum of per-query `n_gold` over the 40 phase5 queries | **172** |
| Distinct gold PMIDs over the BioASQ-only phase5 subset | **145** |

None is 789. The canonical measured figures are **2,394/250** (full) and **172/40**
(phase5 subset). "789" most plausibly came from a run configuration or question subset not
retained in the repository.

### Coverage gap vs retrieval failure

A **missing gold document** means the corpus does not contain the PMID at all (a coverage
gap, not a ranking failure). A **retrieval failure** means the PMID is in the corpus but is
not returned in the top-k. Every `eval_split` miss is a coverage gap by construction.

### PMID→chunk traceability

Model-free round-trip over 20 sampled `eval_split` chunks: **20/20** had
`metadata["pmid"]` equal to the `source`-parsed PMID; `document_id` groups chunks of one
article. A retrieved chunk is traceable to its PubMed article.

---

## Phase B — Isolated retrieval-eval index

Built at `backend/data/indexes/bioasq_retrieval_eval/v1/` from the **eval question snippets
only** (local; no bulk download). The production index and `eval_split` are never touched.

- Vectors (`index.ntotal`) = **2,947**; chunks = **2,947** → `ntotal == len(chunks)`.
- Dimension = **1024** (BGE-M3).
- Distinct PMIDs = **2,010**.
- Verification: gold round-trip **10/10** (each sampled query's own gold PMID appears in its
  top-5); `metadata["pmid"]` matches `source`-parsed PMID **10/10**.

**Recorded deviations** (for comparability / honesty, all in `v1/manifest.json`):
- Embedding model forced to `BAAI/bge-m3` (1024-dim), **not** the stale
  `bge-small-en-v1.5` id in `app/config.py` — required so similarity is comparable to the
  1024-dim production/eval indexes.
- `batch_size=16` (overrides the embedder default of 8).
- Content is section-prefixed `"[{beginSection}] "` (`endSection` ignored), matching the
  live eval_split/phase5 citation convention.
- Snippets deduped by `(pmid, normalized_text)` before embedding.

**Coverage (honest, < 100%).** Snippet-only means a gold PMID is present only if some eval
question produced a snippet for it. Corpus coverage of the 2,394 distinct gold PMIDs =
**83.96%** (2,010 covered). This is the "corpus coverage" view below and is reported, not
hidden. Optional `--fetch-missing` / `--add-training-distractors` flags exist (default OFF)
to raise coverage; they were not used in this run.

---

## Phase C — Retrieval metrics (reusing `evaluation/metrics.py`)

Actual top-5 from the retrieval-eval retriever, chunks collapsed to articles by
rank-preserving dedup (multiple chunks of one PMID count once), compared against the full
normalized gold set per query. Macro-averaged. Mere PMID presence is never a hit — only
top-5 article-rank membership counts.

### Corpus coverage view (over all 250)
- `coverage@≥1` = **1.000** (all 250 queries have at least one gold PMID in the corpus).
- `coverage@all` = **0.752** (188/250 queries have *all* their gold PMIDs covered).
- Uncovered-gold queries = **0**; partially-covered queries = **62**.
- Corpus distinct PMIDs = 2,010.

### End-to-end view (denominator = all 250; uncovered-gold queries score 0)
| Metric | Value |
|---|---|
| Precision@5 | **0.564** |
| Recall@5 | **0.5611** |
| nDCG@5 | **0.8328** |
| MRR | **0.961** |

### Retrieval effectiveness view (denominator = covered queries)
Because `coverage@≥1 == 1.0`, every query is covered, so the conditional view equals the
end-to-end view exactly: P@5 **0.564**, R@5 **0.5611**, nDCG@5 **0.8328**, MRR **0.961**.
Recall is still measured against the *full* gold set, so the 62 partially-covered queries
are correctly penalized for their uncovered gold portion (this is why R@5 < coverage@all).

**Interpretation.** MRR 0.961 and nDCG@5 0.833 show the dense retriever ranks a relevant
article at or near the top for almost every query. P@5/R@5 near 0.56 reflect that most
queries have more gold articles than fit (or are reachable) in 5 slots, plus the 83.96%
snippet-only coverage ceiling — a corpus-coverage limitation, not a ranking failure.

---

## Phase D — Confidence calibration (reused phase5 artifacts, no LLM)

Source selected by fixed precedence and verified:
`RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` (first comparable candidate).
`confidence_abstention.jsonl` was cross-checked and **agrees** (n_rows, id-set, per-id
correctness/abstained) → `source_consistency: "ok"`. The in-backend
`phase5-full-pipeline.jsonl` mirror carries a **reduced schema with no `correctness` field**
(3 rows, `config`/`confidence`/`abstain_flag`); it is correctly classified **incomparable**
and excluded from the equivalence check rather than treated as a data mismatch. This does
not affect the selected source or the realized N/P.

Realized counts for `phase5-full-pipeline`:
`40 config rows → 13 BioASQ (27 pubmedqa excluded) → 2 non-abstained (11 abstained excluded) → **N=2, P=0**`.

**Data regime: DEGENERATE — single-class / insufficient data, provisional, not a
calibration claim.** With **0 correct** answers among 2 non-abstained BioASQ questions there
is no positive class, so a calibration map and a selective-accuracy threshold cannot be
meaningfully fit. The raw Agent-7 confidence is treated as uncalibrated; the well-defined
raw statistics over the labelled rows are reported:

- Raw ECE (10 equal-width bins) = **0.8493**
- Raw Brier = **0.7213**
- At incumbent threshold 0.65: coverage **1.0**, selective accuracy **0.0**, 2 answered.

Both high ECE and high Brier simply reflect that both answers were wrong while confidence
was not near 0 — they are **not** a calibration verdict. The graded per-type evaluator and
its ≥0.85 automated-vs-manual acceptance gate are wired but resolve to **provisional by
default** (no `--manual-labels` supplied, tiny sample).

The split + method-selection (Platt vs isotonic, lower fit-ECE) + threshold protocol
(§8.5–8.6) is implemented and unit-tested, but is **not reached** on this data (requires
N ≥ 15 and both classes). More labels are obtainable only via the explicit, default-OFF
`--generate-labels` path.

---

## Phase E — Threshold 0.65 validation

`threshold_evaluation.json` records **status: `not_evaluable`**. With N=2 and 0 correct, the
0.65 threshold **can be neither confirmed nor refuted** from this data. No held-out test
partition is produced (there is nothing to hold out). The incumbent 0.65 is left untouched
in app code. The selection protocol — predeclared grid 0.30–0.90 step 0.05, objective "max
coverage s.t. selective accuracy ≥ 0.80 on the select partition, smallest-threshold
tie-break", select-then-evaluate-once on an untouched test partition, acceptance iff test
selective accuracy ≥ 0.80 and coverage ≥ 0.50, bootstrap CIs only when test n ≥ 30 — is
implemented and exercised by `tests/test_bioasq_calibration.py`.

---

## Phase F — Resumable driver

`run_all.py` orchestrates integrity-pre → build → diagnose → retrieval → calibration →
integrity-post with stable per-query ids, config-hash idempotence (the index build was
**skipped** on the second run via config-hash match), atomic incremental JSONL writes
(resumable by `question_id`), bounded 3× retry for transient failures, and never silently
drops a failed phase. `--no-model` / `--diagnostic-only` run without the generation LLM.

---

## Phase G — Tests

`tests/test_bioasq_*.py`, run with `d:\Ragmed_2\.venv\Scripts\python.exe` — **33 passed, 0
failed** (see `test_report.txt`). Coverage includes: production-index + eval_split checksum
unchanged vs `integrity_baseline.json`; eval_split membership (58,173 chunks) unchanged;
`IndexLeakageGuard` behaviour; PMID normalization/mapping; retrieval metrics vs
hand-verified examples (all-5-irrelevant, first-relevant, relevant-at-rank-5,
multiple-relevant, multiple-chunks-same-PMID collapse, gold-absent conditional-vs-end-to-end
divergence, rank-order preservation); ECE/Brier vs hand-computed; invalid-score /
missing-label handling; the degenerate branch selector; schema classification + pubmedqa
exclusion; `build_splits` disjoint/union + sparse-stratum fallback; method selection;
threshold objective + smallest-threshold tie-break; and the invariant that the test
partition is untouched until the single final evaluation.

Note: the pre-existing `tests/test_workstream_b_metrics.py` imports via
`from backend.evaluation.metrics …`, which only resolves when pytest is invoked from the
repo root (not from `backend/`). This is a pre-existing cwd/import-convention artifact,
unaffected by this work — `evaluation/metrics.py` and `evaluation/split_builder.py` were
reused unmodified.

---

## Integrity

SHA-256 of the four protected files (production `index.faiss`/`chunks.pkl`, `eval_split`
`index.faiss`/`chunks.pkl`) matched `integrity_baseline.json` **before and after** the run,
and eval_split membership (58,173 chunks, 0 missing PMIDs) was unchanged. See
`integrity_check.json`. The run is marked VALID only because both checks passed.

## What succeeded and what remains blocked

- **Succeeded:** diagnosis + figure reconciliation; isolated index build with verified ID
  mappings and round-trips; real retrieval metrics (P@5 0.564, R@5 0.561, nDCG@5 0.833,
  MRR 0.961) with three honest views; calibration data loading with verified source
  consistency; raw ECE/Brier; full test suite; integrity preserved.
- **Blocked by data, not by code:** a defensible calibration map, ECE/Brier *after*
  calibration, and validation of the 0.65 threshold. The sole on-disk labelled set
  (`phase5-full-pipeline`) yields N=2 non-abstained BioASQ questions with 0 correct — a
  single-class, insufficient sample. The precise missing prerequisite is **more labelled,
  non-abstained, correctly-answered BioASQ questions**, obtainable only via the explicit
  `--generate-labels` generation path (default OFF, not run here).
