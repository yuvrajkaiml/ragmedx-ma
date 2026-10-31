# Code Review — BioASQ retrieval-eval + confidence-calibration fix

The change adds an isolated evaluation pipeline under `backend/evaluation/bioasq_eval/` that
diagnoses why none of the BioASQ gold PMIDs appeared in the `eval_split` index, builds a
separate versioned retrieval-eval index from the eval questions' gold snippets, computes
article-level retrieval metrics (P@5/R@5/MRR/nDCG@5) in three views, and assesses confidence
calibration against reused phase5 artifacts. It is strictly additive: production and `eval_split`
indexes are hash-guarded and left untouched, and all outputs land under the two sanctioned roots.
The work is methodologically honest about its hard limitation — the only on-disk labelled
confidence data yields N=2 non-abstained BioASQ questions with 0 correct, so calibration and the
0.65 threshold are reported as `not_evaluable`/DEGENERATE rather than fabricated.

Watch for: the retrieval corpus is built entirely from the eval questions' own gold snippets with
zero distractors (default-OFF), so corpus coverage is 100% by construction and the "retrieval
effectiveness" view is numerically identical to the "end-to-end" view (confirmed) — this makes the
ranking metrics optimistic and collapses the two-view distinction, though it is disclosed in the
report. Two build flags advertised as coverage levers (`--add-training-distractors`,
`--fetch-missing`) are not actually wired to ingest/fetch (confirmed) — minor, since both are
default-OFF and out of the gated scope.

**Verdict**: APPROVED

## High-level view

Integrity is airtight. The four protected files' SHA-256 match `integrity_baseline.json` before and
after the run, eval_split membership (58,173 chunks) is unchanged, and on-disk timestamps confirm no
writes to `backend/data/faiss_index/`. The new index lives only under
`backend/data/indexes/bioasq_retrieval_eval/v1/`, and all reports under
`backend/results/bioasq_eval_20261018/`.

Partition separation is real, not prose. The calibration module fits the calibration map on a `fit`
split only, selects the threshold on a `select` split, and evaluates the incumbent 0.65 and any
selected threshold exactly once on an untouched `test` split, guarded by a `test_opened_once` flag
and a disjoint/union assertion. A unit test drives the full N=60 both-class path and asserts the
test ids never touch fit/select. On the current data this protocol is never reached (DEGENERATE,
N=2, P=0), and the pipeline correctly records `not_evaluable` instead of inventing a split.

Retrieval metrics are computed from the actual top-5, deduped to articles by a rank-preserving
collapse, and compared to the full normalized gold set. A hand-check of the first per-query record
reproduces its P@5/R@5/MRR/nDCG exactly. Mere PMID presence is never counted as a hit.

The calibration honesty holds up under spot-check: the source predictions file genuinely contains
two non-abstained BioASQ rows in `phase5-full-pipeline`, both incorrect, with confidences 0.8472 and
0.8514 — matching the reported N=2/P=0, raw ECE 0.8493, and Brier 0.7213. Nothing is fabricated.

The one substantive methodological caveat is corpus construction: the retrieval-eval index is the
gold snippets themselves with no distractors, so every query is "covered" and the two reported views
coincide. The report states this plainly, so it is a disclosed validity limitation rather than a
hidden coverage gap.

<details>
<summary>Issues (3)</summary>

1. **Snippet-only corpus inflates retrieval metrics (non-blocking)** — the corpus is built from the
   eval questions' own gold snippets with zero distractors, so coverage@≥1 = 1.0 and the
   "retrieval effectiveness" view equals the "end-to-end" view exactly. Metrics (MRR 0.961, nDCG
   0.833) are optimistic. Disclosed in the report; a later pass could enable `--add-training-distractors`
   to add negatives, but the current result is honest about the ceiling.
2. **Advertised coverage flags are inert (non-blocking)** — `--add-training-distractors N` only warns
   when the file is missing and `--fetch-missing` prints "not auto-enabled"; neither actually ingests
   or fetches. The report says these flags "exist (default OFF) to raise coverage," which slightly
   overstates them. Harmless given both are out of gated scope, but the report wording should match.
3. **Artifact filenames differ from the review brief (non-blocking)** — the gate prompt referenced
   `diagnostics.json`, `retrieval_metrics.json`, `calibration_metrics.json` at the top level; the real
   files are under `diagnostics/`, `retrieval/`, and `calibration/`. All content is present and
   consistent; only the paths differ.

</details>

<details>
<summary>Details</summary>

### Production index and eval_split are provably untouched

`integrity.py` freezes the four protected files' SHA-256 against `integrity_baseline.json` and
re-checks pre- and post-run; a mismatch raises `RuntimeError` and marks the run INVALID.
`integrity_check.json` shows `pre_run_ok` and `post_run_ok` true with every `actual == expected`,
and eval_split membership of 58,173 chunks with 0 missing PMIDs unchanged on both sides. The
production `faiss_index/` files carry September-2026 timestamps, predating the October-18 run, and
the isolated index exists only at `backend/data/indexes/bioasq_retrieval_eval/v1/`. The module reads
the protected files to hash them and never opens them for write. This satisfies the
non-negotiable constraint.

### Partition separation holds in code, not just in the report

`split_and_validate` builds `fit/select/test` via the reused `build_splits`, asserts the three id
sets are pairwise disjoint, fits Platt and isotonic on `fit` only, selects the lower-fit-ECE map,
runs the threshold grid against the `select` partition, and evaluates both the selected threshold
and the incumbent 0.65 exactly once on `test`, tagging the result `test_opened_once: True`. ECE and
Brier are reported raw-and-mapped on all three partitions, so the "before vs after a separately-fit
calibration layer" requirement is met by construction. `test_test_partition_untouched_until_final_eval`
exercises this with N=60/P=30 (both classes present, real protocol reached) and asserts the test ids
are disjoint from fit and select. Per-partition single-class guards short-circuit to `not_evaluable`
rather than producing a spurious verdict.

On the data that actually exists this protocol is deliberately not reached: the branch selector sees
`P == 0` and routes to the DEGENERATE path, writing `threshold_evaluation.json` with
`status: not_evaluable` and `threshold_0_65_validated: null`. No split is fabricated to force a
number. That is the correct behavior under the no-fabrication constraint.

### Retrieval metrics are real top-5, deduped to articles

`run_retrieval_eval.py` loads the v1 index through `DenseRetriever.load`, retrieves the top-5 per
query, maps chunks to PMIDs via `dedup_articles` (rank-preserving first-occurrence collapse, so
multiple chunks of one article count once), and scores against the full normalized gold set using
the reused `evaluation/metrics.py`. Hand-verification of the first per-query record
(`604900391cb411341a000161`, gold `{18039527, 31709696, 31722348}`, retrieved
`[31709696, 18039527, 31722348, 22222968, 1473624]`) reproduces P@5 = 0.6, R@5 = 1.0, nDCG = 1.0,
reciprocal rank = 1.0 exactly. The dedup invariant and rank preservation are covered by dedicated
unit tests. Presence of a PMID in the corpus is never treated as a hit — only top-5 article-rank
membership counts.

The two reported views are computed with genuinely different denominators (`n_total` vs
`n_covered`) in code; they coincide only because `coverage@≥1 == 1.0` makes `n_covered == n_total ==
250`. That 100% coverage is a direct consequence of the snippet-only corpus choice (the corpus *is*
the gold passages), which the design made deliberately with distractors default-OFF. The report
states the 83.96% distinct-PMID coverage ceiling, the 62 partially-covered queries, and that R@5 is
measured against the full gold so partial coverage is penalized. Nothing about coverage is hidden; the
caveat is that the "effectiveness" view adds no information here and the ranking numbers are
optimistic relative to a corpus seeded with real distractors.

### Calibration honesty is backed by the source data

A narrow spot-check of `RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` confirms the reported
funnel: 40 `phase5-full-pipeline` rows → 13 BioASQ → 2 non-abstained, both `correctness=False`,
confidences 0.8472 and 0.8514. These reproduce the calibration report's N=2, P=0, the 0.8–0.9
histogram bin, raw ECE 0.8493 (mean confidence against zero accuracy in a single occupied bin), and
Brier 0.7213 (mean of the two squared confidences). The raw heuristic is treated as uncalibrated and
every degenerate statistic is labelled "provisional, not a calibration claim." Source selection
probes three candidates in fixed precedence, cross-checks comparable candidates for row/id/label
agreement (`source_consistency: ok`), and correctly classifies the reduced in-backend mirror as
`incomparable` rather than flagging a false mismatch.

### Diagnostic and figure reconciliation

The diagnostic confirms the root cause: 0/2,394 gold PMIDs appear in `eval_split`, by deliberate
leakage exclusion, matching the prior run's `metric_reason`. The "789" figure is shown to be
irreproducible from any on-disk artifact, with four candidate populations enumerated (2,394/250,
172/40, 172 Σn_gold, 145 BioASQ-13), and the canonical figures stated. A 20-sample PMID→chunk
round-trip passes 20/20.

### Tests actually ran

`test_report.txt` records 33 passed, 0 failed, with test names that map to the plan's required cases
(integrity checksum match, eval_split membership, leakage guard, PMID normalization, the retrieval
edge cases including multi-chunk collapse and conditional-vs-end-to-end divergence, hand-computed
ECE/Brier, the DEGENERATE branch selector, pubmedqa exclusion, build_splits disjoint/union with
sparse-stratum fallback, method selection, threshold objective with tie-break, and the
test-untouched-until-final-eval invariant). All four referenced test files exist on disk. The
reused `evaluation/metrics.py` and `split_builder.py` were not modified.

</details>

<details>
<summary>File map</summary>

- `backend/evaluation/bioasq_eval/integrity.py` — SHA-256 pre/post guards + eval_split membership check.
- `backend/evaluation/bioasq_eval/common.py` — PMID normalization, chunk→article dedup, atomic/resumable IO, manifests.
- `backend/evaluation/bioasq_eval/diagnose.py` — Phase A diagnostic + figure reconciliation.
- `backend/evaluation/bioasq_eval/build_eval_index.py` — Phase B isolated snippet-only index build (v1).
- `backend/evaluation/bioasq_eval/run_retrieval_eval.py` — Phase C three-view article-level metrics.
- `backend/evaluation/bioasq_eval/calibrate_confidence.py` — Phase D/E source load, DEGENERATE policy, split/threshold protocol.
- `backend/evaluation/bioasq_eval/run_all.py` — Phase F resumable driver with integrity pre/post.
- `backend/tests/test_bioasq_{integrity,common,retrieval_metrics,calibration}.py` — unit tests (33 passed).
- Artifacts under `backend/results/bioasq_eval_20261018/` (diagnostics/, retrieval/, calibration/, manifests/, reports) and the index under `backend/data/indexes/bioasq_retrieval_eval/v1/`.

</details>
