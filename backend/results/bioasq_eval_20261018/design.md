# Design — Fix BioASQ Retrieval Evaluation and Confidence Calibration in RAGMedX-MA

Status: **REVISION 3** (addresses the latest `design-review.json`, verdict CHANGES_REQUESTED,
findings 1–7 of that review). Revision 2's responses to the earlier review are retained in **§14**;
Revision 3's responses to the current review are in **§15**.
Workflow: `wf_f9d894c9ec5422e5`. Deliverables root: `d:\Ragmed_2\backend\results\bioasq_eval_20261018`.
Isolated retrieval-eval index root: `d:\Ragmed_2\backend\data\indexes\bioasq_retrieval_eval`.
Python: `d:\Ragmed_2\.venv\Scripts\python.exe`. Backend dir: `d:\Ragmed_2\backend`.

> Reviewer responses are collected in **§14** (prior review) and **§15** (current review). Every
> HIGH/MEDIUM finding is addressed in-line (the relevant sections are rewritten) and summarized there.
> The current review raised 0 HIGH, 4 MEDIUM (path resolution, source-equivalence check, per-partition
> single-class guard, unified per-view metric-denominator formula) and 3 NIT (evaluator-gate honesty,
> section-prefix documentation, retrieval resume-key id-space). Revision 3 resolves all seven; the
> MEDIUMs change which code path runs or what number is reported, so each is pinned to an explicit,
> cwd-independent, deterministic rule. During this revision the authoring agent **re-verified on disk**
> that `RAGMedXMA_Chapter5_Validated_Results/` lives at the repo root (`d:\Ragmed_2\`) while scripts run
> from `backend/` — confirming finding 1 — and that the `backend/RAGMedXMA_Chapter5_Validated_Results/`
> and `backend/backend/results/...` paths do **not** exist.

---

## 1. Overview

The current RAGMedX-MA evaluation reports retrieval metrics (Precision@5, Recall@5, MRR, nDCG@5) as
"unavailable because none of the … BioASQ gold PMIDs were found in the evaluation index," never
measures confidence calibration (ECE, Brier), and never validates the hard-coded
`CONFIDENCE_THRESHOLD = 0.65`. The root cause is **confirmed and must not be "fixed" by touching the
eval index**: the `eval_split` FAISS index at `backend/data/faiss_index/eval_split/` was built with a
*deliberate* cross-PMID exclusion as a leakage safeguard, so **zero** gold PMIDs appear in it by design.
This design verified the symptom directly: the prior retrieval run
`RAGMedXMA_Chapter5_Validated_Results/retrieval_results.jsonl` has, for **all 40** evaluated queries,
`gold_pmids_in_index: []`, `relevant_docs_in_index: false`, metric values `null`, and
`metric_reason: "NOT MEASURABLE: gold PMIDs not present in eval_split index (dataset/corpus mismatch)"`.

The real work is therefore not to repair `eval_split` but to **build a separate, versioned,
retrieval-only BioASQ evaluation index** that *does* contain the gold documents, compute article-level
retrieval metrics against it, and separately **assess calibration and validate the threshold** of the
Agent-7 confidence heuristic using disjoint data partitions **to the extent the available labelled data
supports it** (§8 shows the available data is severely limited; the design states exactly what can and
cannot be claimed as a result, and never fabricates numbers to fill the gap).

All new artifacts are isolated under the two roots above; the production index, the `eval_split` index,
and the `eval_split` dataset are never modified. The design reuses the existing embedding model
(BGE-M3, 1024-dim — verified below), chunk schema (`MedicalChunk`), FAISS store, dense retriever, the
metric implementations in `evaluation/metrics.py`, and the seeded split builder in
`evaluation/split_builder.py`.

This document resolves every ambiguity with a concrete decision; there are no open architectural
questions for the implementer.

---

## 2. Technology stack (locked)

- **Language/runtime:** Python 3.14 via `d:\Ragmed_2\.venv\Scripts\python.exe` (per `env_smoke.json`).
- **Import strategy:** every script inserts the backend dir at `sys.path[0]` before importing
  `app.*` / `evaluation.*` — the `syspath_insert_backend` strategy is the **only** one that resolves
  these modules (verified in `env_smoke.json`; `default_syspath` raises `ModuleNotFoundError`). Scripts
  under `backend/evaluation/bioasq_eval/` begin with
  `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))` (which resolves to `backend/`), or are
  run as `python -m evaluation.bioasq_eval.<module>` from `backend/`.
- **Path-resolution rule (locked — findings 1, 2):** `cwd = D:\Ragmed_2\backend` at run time
  (`env_smoke.json`), but some read-only source artifacts live at the **repo root** (`D:\Ragmed_2\`) and
  some under `backend/`. Relative paths are therefore ambiguous and MUST NOT be used. Every script pins
  two cwd-independent anchors derived from its own location
  (`backend/evaluation/bioasq_eval/<module>.py`):
  `BACKEND_DIR = Path(__file__).resolve().parents[2]` (→ `D:\Ragmed_2\backend`) and
  `REPO_ROOT = Path(__file__).resolve().parents[3]` (→ `D:\Ragmed_2`). **All** input/output paths are
  built by joining onto one of these anchors; no path is ever resolved against the process cwd. The three
  calibration source candidates resolve to the following absolute forms (verified on disk this revision —
  the first and third exist; the `backend/RAGMedXMA_...` and `backend/backend/results/...` forms do
  **not** and are explicitly rejected):
  `REPO_ROOT/"RAGMedXMA_Chapter5_Validated_Results"/"predictions.jsonl"`,
  `REPO_ROOT/"RAGMedXMA_Chapter5_Validated_Results"/"confidence_abstention.jsonl"`,
  `BACKEND_DIR/"results"/"phase5_benchmark"/"per_question"/"phase5-full-pipeline.jsonl"`. Each loader
  **logs the fully-resolved absolute path of every candidate it probes** (exists / not-found) so a miss is
  diagnosable rather than silent.
- **Embedding model (locked):** `BAAI/bge-m3` via `app.rag.embeddings.bge_embedding.BGEEmbedding`,
  `normalize=True`, `max_length=512` (hard-coded in `embed`), mean pooling, L2-normalized, 1024-dim.
  **Rationale:** both the production index and `eval_split` are 1024-dim, which is BGE-M3, not the
  `BAAI/bge-small-en-v1.5` string in `app/config.py` (`Settings.embedding_model_name`). That config
  value is **stale for this purpose**; the new index MUST instantiate `BGEEmbedding()` (default
  `BAAI/bge-m3`) so similarity is comparable to the production/eval indexes. **Do NOT read the model id
  from `app/config.py`.** This forced deviation is recorded in every manifest. The compatibility alias
  `BGEEmbedder` now also points at `BGEEmbedding` (verified in `app/rag/embeddings/bge_embedding.py`),
  so either name yields the 1024-dim model.
- **Vector store (locked):** `app.rag.vectorstore.faiss_store.FAISSVectorStore(dimension=1024)`
  (`IndexFlatIP`, cosine via normalized IP). Position in `store.chunks` == FAISS vector id (verified).
  `add(embeddings, chunks)`, `search(query_embedding, top_k)`, `save(dir)`; `load(dir)` derives `d` from
  the index file and asserts `index.ntotal == len(chunks)`. Re-exported as
  `app.retrieval.vector_store.FAISSVectorStore`.
- **Retriever (locked, construction pinned — see finding 5):** `DenseRetriever`
  (`app.rag.retrievers.dense_retriever`, re-exported at `app.retrieval.dense_retriever`).
  **Construct via the verified classmethod** `DenseRetriever.load(index_dir, embedder=BGEEmbedding())`,
  or equivalently `DenseRetriever(embedder=emb, vector_store=FAISSVectorStore.load(index_dir))` using the
  keyword **`vector_store=`**. **Do NOT copy the `store=` form shown in the
  `app/retrieval/dense_retriever.py` re-export docstring — the canonical `__init__` parameter is
  `vector_store`, so `store=` raises `TypeError`.** `retrieve(query, top_k)` returns
  `list[(MedicalChunk, score)]` ordered high→low; it raises `TypeError` on non-str query, `ValueError`
  on empty query or `top_k <= 0`. Dense-only for the retrieval-eval index (hybrid/reranker excluded by
  default — §6.4).
- **Metrics (locked, REUSE):** `evaluation/metrics.py`. Verified signatures/semantics:
  `precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k)` (denominator always `k`),
  `recall_at_k(retrieved_ids, relevant_ids, k)` (denominator `|relevant|`),
  `mrr(retrieved_ids_list: list[list[str]], relevant_ids_list: list[list[str]])`,
  `ndcg_at_k(retrieved_ids, relevant_ids, k)` (**binary** relevance; returns 0.0 when no relevant),
  `ece(confidences, correct_flags, n_bins=10)` (10 equal-width bins), `brier_score(confidences,
  correct_flags)`, `risk_coverage_curve(confidences, correct_flags)`, plus `exact_match(pred, ref)`,
  `compute_rouge_l(pred, ref)`, `abstention_precision/recall`. No reimplementation unless a bug is proven
  and documented.
- **Splits (locked, REUSE):** `evaluation/split_builder.build_splits(dataset, dev_n, cal_n, test_n,
  seed=42, stratify_by=None|"type")` → returns `(dev_split, cal_split, test_split)`. Verified behaviour:
  raises `ValueError` if `len(dataset) < dev_n+cal_n+test_n`, on duplicate ids, and — when
  `stratify_by` is set — if **any stratum** has fewer items than its allocated `d+c+t`
  (`max(1, round(frac*n))` per split per stratum). The design's slot mapping and sparsity guard are in
  §8.5 (findings 3, 4).
- **Calibration maths:** `scikit-learn` if importable (`LogisticRegression` for Platt,
  `IsotonicRegression` for isotonic); else a self-contained NumPy logistic fit (documented fallback).
  Availability is probed at run start and the chosen path recorded in the manifest.
- **Numerics (verified present):** NumPy 2.5.2, FAISS 1.15.0, torch 2.13.0+cpu, sentence-transformers
  5.7.0, transformers 5.15.0. CPU-only; no CUDA assumptions.
- **Seed (locked):** `42` everywhere (`random`, `numpy`, split builder, bootstrap).

---

## 3. Hard constraints encoded by this design

1. **Never** modify/rebuild/overwrite/delete/re-embed `backend/data/faiss_index/` (production) or its
   `eval_split/` subdir. Enforced by: (a) all writes target `backend/data/indexes/bioasq_retrieval_eval/`
   and `backend/results/bioasq_eval_20261018/` only; (b) pre-run and post-run SHA-256 integrity checks
   against `integrity_baseline.json` (§10). Any mismatch is fatal and marks the run INVALID.
2. **Never** modify `backend/data/datasets/bioasq_eval_split/` membership or labels; it is opened
   read-only.
3. The retrieval-eval corpus is built from gold snippet text that **never feeds the generation
   pipeline**, so using it for *retrieval* metrics does not create leakage. This isolation is stated
   explicitly in the manifest and report.
4. Separate partitions for retrieval eval, calibration fitting, and final threshold validation; a
   threshold is never tuned on the examples used to report its final performance (§8).
5. No fabrication. Where data is insufficient (and §8 shows it is, for calibration), the report states
   "insufficient n" / "no positive class" / "provisional" and reports what *is* measurable rather than
   inventing numbers.
6. No bulk corpus download. Local gold snippets are the primary source; optional PubMed fetch of
   *missing* gold PMIDs only, rate-limited, provenance-recorded (§6.3, default OFF).
7. Checkpoint/resume, atomic incremental writes, config-hash manifest, single model load per process
   (§9).
8. Reproducibility: fixed seeds, recorded configs, dataset+index versions, config hashes, timestamps,
   completed-item counts (§9, §10).

---

## 4. Reconciling the "789" / "2,394" / "250" gold-PMID figures (finding 6)

**Verified counts (recomputed from disk during this revision):**

| Population | Source artifact | Count |
|---|---|---|
| Eval questions | `bioasq_eval_split/eval_questions.json` (`keys: body, documents, exact_answer, id, ideal_answer, snippets, type`) | **250** questions |
| Type distribution | same | factoid 72, yesno 62, list 59, summary 57 |
| Distinct gold PMIDs over all 250 | normalize `documents[]` with `re.search(r"(\d+)$", url)` | **2,394** |
| Phase5 benchmark universe | `confidence_abstention.jsonl` / `predictions.jsonl` (`experiment_id: phase5_40x8`) | **40** questions × 8 configs |
| 40-question subset distinct gold PMIDs | union of `gold_relevant_pmids` in `retrieval_results.jsonl` (also == Σ `n_gold`) | **172** |

**Predeclared candidate sources checked for the "789" figure, and the verdict:**
1. All-250 distinct gold PMIDs → 2,394 (not 789).
2. 40-question phase5 subset distinct gold PMIDs → 172 (not 789).
3. Σ `n_gold` over the 40 phase5 queries → 172 (not 789).
4. 40-subset BioASQ-only (13 questions) distinct gold PMIDs → 145 (not 789).

**Resolution (verifiable, not a shrug):** none of the concrete artifacts on disk reproduces 789. The
design therefore **does not treat 789 as a target to reproduce**. The diagnostic (§5) recomputes and
reports the authoritative figures (**2,394 distinct gold PMIDs over 250 questions**; **172 over the
40-question phase5 subset**) and records, by name, the four candidate populations checked above and the
count each yields. The report states plainly: "the originally quoted 789 figure could not be reproduced
from any artifact present in the repository; it most plausibly originated from a run configuration or
question subset not retained on disk. The canonical measured figures are 2,394/250 (full) and 172/40
(phase5 subset)." This makes "could not reproduce 789" a **verifiable, evidenced result** with the exact
artifacts and counts enumerated, satisfying the Phase-A success criterion.

---

## 5. Phase A — Diagnostic (runs WITHOUT the generation LLM)

Script: `backend/evaluation/bioasq_eval/diagnose.py`. Pure CPU; the embedding model is loaded **only**
if a retrieval round-trip check is requested (`--roundtrip`), otherwise no model is loaded.

Inputs (read-only): `eval_questions.json`; `faiss_index/eval_split/chunks.pkl` (via
`FAISSVectorStore.load`, needs backend on `sys.path`); optionally `faiss_index/chunks.pkl` (production)
for a corpus-coverage cross-check; the prior `retrieval_results.jsonl` for symptom confirmation.

**eval_split PMID field is verified, not assumed (finding 10):** this revision loaded
`eval_split/chunks.pkl` (58,173 `MedicalChunk` objects) and confirmed each chunk carries
`metadata` with keys `{pmid, pmid_url, begin_section, end_section, offset_begin, offset_end,
chunk_index, total_chunks, question_id, source_file}` and `source` = full PubMed URL (sample:
`metadata['pmid'] == '29040384'`, `source == 'http://www.ncbi.nlm.nih.gov/pubmed/29040384'`). Canonical
PMID = `chunk.metadata["pmid"]`; the `re.search(r"(\d+)$", chunk.source)` parse is the documented
fallback. The diagnostic still re-asserts field presence on first load and counts any chunk missing
`metadata['pmid']` (expected 0) so the assumption is validated at run time, not trusted.

Steps and outputs (written to `diagnostics/diagnostic_report.json` + `.md`):
- `n_queries` (expect 250), `n_queries_with_gold` (expect 250).
- `n_gold_pmids_raw`, `n_gold_pmids_unique_normalized` (expect 2,394). Normalization =
  `re.search(r"(\d+)$", str(url)).group(1)` on each entry of `question["documents"]`; non-matching
  entries counted and sampled under `unparsable_pmids`.
- `eval_split_chunks` (expect 58,173), `eval_split_distinct_pmids` (from `metadata["pmid"]`).
- `gold_pmids_matched_in_eval_split` (**expect 0**) + percentage → confirms the deliberate exclusion.
  The report cites the prior `retrieval_results.jsonl` `metric_reason` as the matching symptom.
- Optional production cross-check: `gold_pmids_present_in_production_index` + percentage (coverage
  context only; the production index is never used for retrieval scoring here).
- Per-query availability **against the NEW retrieval-eval corpus** (§6; computed after the corpus plan
  is materialized): `n_queries_with_at_least_one_available_gold`, `n_queries_with_all_gold_available`.
- `missing_pmid_examples`: up to 25 gold PMIDs absent from the retrieval-eval corpus, each tagged with a
  likely cause (`not_in_local_snippets`, `normalization_variant`, `fetch_skipped`).
- Explicit distinction section: **missing gold document** (corpus does not contain the PMID) vs
  **retrieval failure** (PMID is in corpus but not in top-k).
- **PMID→chunk traceability check:** for 20 sampled gold PMIDs present in the new corpus, assert
  `chunk.metadata["pmid"]` round-trips to the same bare PMID parsed from `chunk.source`, and that
  `document_id` groups chunks of one article; record pass/fail counts.
- The §4 figure reconciliation (789/2,394/172/250) is computed and written here.

---

## 6. Phase B — Build the isolated retrieval-eval index

### 6.1 Corpus decision (resolved)

**Decision: build the retrieval-eval corpus from the `eval_questions.json` snippets only (local), not
from `training13b.json`.** Each eval question's own `snippets[]` carry the gold-document text (`text`,
`document` URL, `beginSection`/`endSection`, `offsetInBeginSection`/`offsetInEndSection`) — the exact
gold passages the benchmark marks relevant for those 250 questions (verified: `eval_questions.json`
entries all carry `snippets`).

Rationale:
- Snippets give gold-passage coverage for the questions we actually score, which is what
  Precision/Recall/MRR/nDCG need. Ingesting all of `training13b.json` (5,389 questions) would add
  unrelated distractor articles that neither help nor are required and would greatly inflate build time
  with no methodological benefit to *conditional retrieval effectiveness*.
- **Coverage implication (reported honestly):** a snippet-only corpus means a gold PMID is present only
  if some eval question produced a snippet for it. Some gold `documents[]` entries have no accompanying
  snippet, so corpus coverage will be **< 100%** of the 2,394 gold PMIDs. The diagnostic quantifies
  exactly how many gold PMIDs are covered; this is the "corpus coverage" view in §7 and is reported, not
  hidden.

**Optional distractor expansion (default OFF, flag `--add-training-distractors N`):** add snippets from
the first N `training13b.json` questions whose PMIDs do **not** overlap any eval gold PMID, to create a
harder negative pool. Default OFF for a reproducible, fast first run; if enabled, N and the source file
SHA-256 are recorded in the manifest.

**Optional PubMed fetch (default OFF, flag `--fetch-missing`):** fetch abstracts for gold PMIDs that have
no local snippet, via NCBI E-utilities, ≤3 requests/sec, cached under
`bioasq_retrieval_eval/pubmed_cache/`, with full provenance (PMID, fetch timestamp, source URL) per
record. Default OFF to respect limits and keep the run self-contained; when OFF, uncovered gold PMIDs are
reported as corpus-coverage gaps. Fetching is a *coverage* improvement only and never changes how
retrieval effectiveness is computed on covered queries.

### 6.2 Chunking / provenance (resolved)

Reuse `MedicalChunk` with the **verified** schema so retrieved chunks trace back to PubMed articles:
- `id = f"bioasq_{qid}_s{snippet_idx}_chunk_{j}"`
- `document_id = f"bioasq_{qid}_s{snippet_idx}"`
- `content` = snippet `text`, **section-prefixed** to match the existing eval_split/phase5 convention
  (**resolved to ON, not optional**): prepend `"[{section}] "` where `section` is the snippet's
  **`beginSection`** (e.g. `[abstract]`, `[sections.0]`). The phase5 citation titles
  ("PubMed 24600005 [abstract]", "PubMed 22146585 [sections.0]") confirm this convention is live in the
  comparable indexes, so matching it keeps embeddings comparable. **Deliberate choice (finding 6):** a
  snippet carries both `beginSection` and `endSection`, which the schema permits to differ (observed
  samples have them equal, e.g. `abstract`/`abstract`); the prefix uses **`beginSection` only and ignores
  `endSection`**, matching the single-section `[abstract]`/`[sections.0]` form seen in the phase5
  citations. The exact prefix rule — "`[{beginSection}] ` prepended, `endSection` ignored" — is recorded
  **verbatim** in the index manifest so the choice is auditable and reproducible.
- `source` = full gold URL `http://www.ncbi.nlm.nih.gov/pubmed/<PMID>`.
- `metadata = {"pmid": "<bare PMID>", "pmid_url": "<full URL>", "begin_section", "end_section",
  "offset_begin", "offset_end", "question_id"}` (same key names as eval_split, verified).
- **Canonical PMID = `chunk.metadata["pmid"]`**, fallback `re.search(r"(\d+)$", chunk.source)`.

Chunking reuses the existing BioASQ-tuned settings (`retrieval_chunk_min_chars=50` so short snippets
survive; snippet text is typically short enough to be a single chunk). Any snippet longer than the
chunker's size is split with the same chunker used elsewhere; multiple chunks of one article share
`document_id` and `pmid`.

**Deduplication:** dedupe raw snippets by `(pmid, normalized_text)` before embedding to avoid identical
snippets inflating the index. Article-level dedup *for metrics* happens at scoring time (§7.1), not here
— the index legitimately holds multiple chunks per article.

### 6.3 Index build & verification

Script: `backend/evaluation/bioasq_eval/build_eval_index.py`. **One** `BGEEmbedding()` load; batch-embed
snippet contents at **`batch_size=16`** (finding 9 — pinned explicitly in the manifest; the embedder's
default is 8, we override to 16 for throughput on CPU and record it); `FAISSVectorStore(dimension=1024)`;
`store.add(embeddings, chunks)`; `store.save(backend/data/indexes/bioasq_retrieval_eval/v1/)`.

Layout:
```
backend/data/indexes/bioasq_retrieval_eval/
  v1/
    index.faiss
    chunks.pkl
    manifest.json   # embedding model id (BAAI/bge-m3), normalize, max_length, batch_size=16,
                    # section-prefix rule, seed=42, source file SHA-256(s), n_chunks,
                    # n_distinct_pmids, build timestamp (UTC), forced-bge-m3 note, config_hash
```
Verification (fatal on failure, written to `build_report.json`):
- `index.ntotal == len(chunks)` (store asserts this on load; re-assert post-build).
- `index.d == 1024`.
- distinct PMID count and chunk count recorded.
- round-trip: embed 10 gold snippet texts, retrieve top-5, assert each query's own gold PMID appears
  among retrieved (sanity that provenance + similarity work end to end).
- sample 10 chunks: assert `metadata["pmid"]` matches `source`-parsed PMID.

Idempotence: if `v1/manifest.json` exists with a matching `config_hash`, skip the rebuild (resume).
`--force-rebuild` creates `v2/` rather than overwriting (never destructive).

### 6.4 Why dense-only (resolved)

The comparison baselines in `retrieval_results.jsonl` were produced by the same dense FAISS path, and the
retrieval-metric question is about first-stage dense retrieval against a corpus that contains the gold.
Hybrid fusion and the cross-encoder reranker add moving parts (BM25 index, reranker model) that are not
needed to answer "can the retriever rank available gold in the top-5" and would make the result
incomparable to the prior dense run. Dense-only is chosen; a `--reranker` flag is left as a documented,
default-OFF extension for a later sensitivity pass, recorded in the manifest if used.

---

## 7. Phase C — Retrieval metrics (REUSE `evaluation/metrics.py`)

Script: `backend/evaluation/bioasq_eval/run_retrieval_eval.py`. **One** `BGEEmbedding()` load; load the
`v1` retrieval-eval index via `DenseRetriever.load(index_dir, embedder=BGEEmbedding())` (pinned
construction); for each of the 250 eval questions, `retriever.retrieve(question["body"], top_k=5)`.

**Id spaces (resolved — finding 7).** The retrieval phase reads `eval_questions.json`, whose records use
the **bare** `id` field (e.g. `604900391cb411341a000161`), **not** the namespaced `question_id` the
phase5 artifacts use (`bioasq::<id>`). The retrieval phase's resume key is therefore the eval **`id`**; it
is written into each `per_query.jsonl` record under the key name `question_id` **for output uniformity**
(so downstream tables have one id column), but it carries the bare eval id, and resume/idempotence compare
against that bare id. The **calibration** phase (§8) keys on the namespaced phase5 `question_id` and strips
the `<source>::` prefix only when joining to `eval_questions.json`. The two id spaces are never mixed in a
single key comparison.

### 7.1 Article-level normalization (the real work)
For each query:
1. Map each retrieved `MedicalChunk` → `metadata["pmid"]` (fallback `source` parse), then **dedup
   preserving first (highest-ranked) occurrence**. Example: ranks `[A_chunk1, A_chunk2, B_chunk1]` →
   article list `[A, B]`. This guarantees multiple chunks of one article never count as multiple
   relevant hits and preserves order for MRR/nDCG.
2. `relevant = set(normalized gold PMIDs for the query)` (handles multiple relevant PMIDs).
3. Compute per-query `precision_at_k(retrieved_articles, list(relevant), 5)`, `recall_at_k(...)`,
   `ndcg_at_k(...)` (binary), and collect `(retrieved_articles, list(relevant))` for the batch `mrr(...)`.

### 7.2 Graded relevance (resolved)
BioASQ provides no graded relevance labels for `documents[]`, so **nDCG uses binary relevance** (gain 1
if the article is gold, else 0). `ndcg_at_k` is binary by construction (verified) and returns 0.0 when a
query has no relevant items. This choice is documented in the report as deliberate and
benchmark-appropriate.

### 7.3 Two reported views + end-to-end view (resolved — finding 4)

Define, once, two helper quantities per query `q`:
- `gold(q)` = the normalized gold PMID set from `q["documents"]` (non-empty for all 250 — gold always
  exists in the benchmark even when it is not in our corpus).
- `available(q)` = `gold(q) ∩ corpus_pmids` (the gold PMIDs actually present in the retrieval-eval
  corpus). `q` is **covered** iff `available(q)` is non-empty.

Every per-query metric is computed **one way only**: the chunk→article dedup of the actual top-5 (§7.1)
compared against the **full** `gold(q)` set via `precision_at_k`/`recall_at_k`/`ndcg_at_k` and, for MRR,
by passing `(retrieved_articles, list(gold(q)))` into the list-of-lists `mrr`. Nothing about the metric
computation changes between views — **only the set of queries in the denominator changes.** This removes
the Revision-2 ambiguity between "score 0" and "excluded": an uncovered-gold query is never magically
excluded from a per-query score; it simply **scores 0** because its gold is unretrievable, and whether
that 0 enters a given average is decided solely by the denominator rule below.

- **Corpus coverage view (one formula):** `coverage@≥1 = |{q : available(q) ≠ ∅}| / 250` and
  `coverage@all = |{q : available(q) == gold(q)}| / 250`. This exposes the snippet-only coverage limit and
  is the number that explains any 0s in the other views.
- **End-to-end view (one formula):** denominator = **all 250** queries; each query contributes its
  per-query metric computed against the full `gold(q)` as above (so an uncovered-gold query contributes
  exactly **0** to P@5/R@5/nDCG@5 and reciprocal rank 0 to MRR). Macro-average = mean of the 250 per-query
  scores (MRR via the list-of-lists function over all 250 pairs — its internal denominator is likewise
  250). This view **never hides missing documents**.
- **Retrieval effectiveness (conditional) view (one formula):** denominator = **only covered queries**
  `{q : available(q) ≠ ∅}`; same per-query computation against the full `gold(q)`. Documented as
  "retrieval quality conditional on corpus coverage." (Note R@5 here is still against the full `gold(q)`,
  not just `available(q)`, so a query whose gold is only *partly* covered is correctly penalized for the
  uncovered portion — the conditional view answers "given at least one gold is reachable, how well does the
  retriever rank it," not "pretend uncovered gold does not exist.")

**Mere presence of a PMID in the corpus is never counted as a successful retrieval — only top-5 article
rank membership counts.** The report prints, alongside the two averaged views, the **count of
uncovered-gold queries** (`|{q : available(q) == ∅}|`) and partially-covered queries, so every 0 in the
end-to-end view is attributable to coverage versus ranking. Outputs: `retrieval/retrieval_metrics.json`
(all three views, the uncovered/partial counts, and a per-query table recording `gold`, `available`,
`covered`, retrieved articles, and each metric) and `retrieval/per_query.jsonl` (atomic incremental
append; resume key per finding 7 below).

### 7.4 Unit tests (new, small & manually verifiable)
`backend/evaluation/bioasq_eval/tests/test_retrieval_metrics.py` covering: all-5-irrelevant
(P=R=MRR=nDCG=0); first-result-relevant; relevant at rank 5; multiple relevant PMIDs; **multiple chunks
mapping to one PMID collapse to one article** (the dedup invariant); gold absent from corpus (conditional
vs end-to-end divergence). Plus a test asserting the chunk→article dedup helper preserves rank order.

---

## 8. Phase D — Confidence calibration & threshold validation

> **This section is substantially rewritten from Revision 1.** During this revision the authoring agent
> read the actual calibration artifacts. The findings below are facts verified on disk, not assumptions,
> and they bound what can honestly be claimed. The headline reality: **the only available per-sample
> confidence+correctness data is a 40-question × 8-config benchmark (`experiment_id: phase5_40x8`), and in
> the `phase5-full-pipeline` config there are 13 BioASQ questions, 2 non-abstained, and 0 correct
> answers.** With no positive class, ECE/Brier are degenerate and a selective-accuracy threshold cannot
> be meaningfully optimized. The design makes "insufficient data / no positive class → provisional,
> clearly-labelled, non-fabricated reporting" the **expected primary path**, and gives a bounded,
> optional, user-gated path to produce more labels.

### 8.1 The score is a heuristic, not a probability
Agent 7 (`app/agents/confidence_calibration.py`) computes
`calibrated_confidence = 0.25*retrieval + 0.30*evidence_quality + 0.25*answer_consistency +
0.20*inter_source_agreement`, clamped `[0,1]`, with hard-coded `CONFIDENCE_THRESHOLD = 0.65` and bands
`high≥0.80`, `moderate≥0.65`. This raw composite is treated as **uncalibrated**. We fit a **separate**
calibration map and report ECE/Brier **before and after**. We never claim the raw heuristic is
calibrated, and we do **not** modify Agent 7's weights or threshold constant (the field is literally
named `confidence_note: "calibrated"` in the data; §14 notes we re-frame it as uncalibrated without
touching the agent).

### 8.2 Calibration data source (resolved — findings 1, 2, 8)

**Verified data inventory (this revision):**

| Artifact | Path | n | Fields relevant to calibration |
|---|---|---|---|
| Per-sample confidence/abstention | `RAGMedXMA_Chapter5_Validated_Results/confidence_abstention.jsonl` | 320 (40×8 configs) | `question_id` (namespaced), `source`, `configuration`, `calibrated_confidence`, `threshold`, `released`, `withheld`, `correctness_token_containment`, `raw_confidence_components` (**null**) |
| Per-sample predictions | `RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` | 320 (40×8) | `question_id`, `dataset`, `reference_answer`, `predicted_answer`, `confidence_score`, `abstained`, `correctness` (bool), `scoring_method` (`token_containment>=0.6`), `confidence_components` (**null**) |
| Prior retrieval eval | `RAGMedXMA_Chapter5_Validated_Results/retrieval_results.jsonl` | 40 | per-query gold/metric fields (used in §5) |
| Per-config per-question | `backend/results/phase5_benchmark/per_question/<config>.jsonl` | 40 each | mirror of the above per config |
| Run-log full-pipeline | `RAGMedXMA_Chapter5_Validated_Results/run_logs/phase5_full-pipeline_per_question.jsonl` | 40 | `gold_answer` (flattened), `confidence`, `abstain_flag`, `status`, `source`, `citations` |

**Primary source (resolved):** `REPO_ROOT/"RAGMedXMA_Chapter5_Validated_Results"/"predictions.jsonl"`
(and the equivalent `confidence_abstention.jsonl`), **filtered to
`configuration == "phase5-full-pipeline"`** (the full Agent-7 path). The Revision-1 path
`backend/results/phase5_20261017_131123/per_question/<config>.jsonl` **does not exist** and is removed
from this design. The per-config file `phase5-full-pipeline.jsonl` under
`BACKEND_DIR/"results"/"phase5_benchmark"/"per_question"/` is the in-`backend` **mirror** (single config);
**the two are NOT assumed interchangeable** — the loader proves equivalence or fails (see below). There is
exactly one phase5 universe (40 questions × 8 ablation configs over the *same* 40 questions); the design
does not promise additional per-config files beyond those.

**Resolution base (resolved — finding 1):** all three candidates are resolved against the cwd-independent
anchors from §2 (`REPO_ROOT` / `BACKEND_DIR`), never the process cwd. The absolute forms are listed in §2;
the loader logs each probed candidate's fully-resolved absolute path with an exists/not-found marker.

**Discovery is fatal on a missing source, never a silent LLM fall-through (finding 1):** the loader checks
the three candidates in the fixed precedence order
(`REPO_ROOT/RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` →
`REPO_ROOT/RAGMedXMA_Chapter5_Validated_Results/confidence_abstention.jsonl` →
`BACKEND_DIR/results/phase5_benchmark/per_question/phase5-full-pipeline.jsonl`). If **none** resolves, the
script exits non-zero with `CALIBRATION_SOURCE_MISSING` and does **not** launch a generation run. A new
generation run happens **only** behind an explicit `--generate-labels` flag (§8.7), never implicitly.

**Source-equivalence check — mirror is not trusted blindly (resolved — finding 2):** the loader selects
the **first** candidate that resolves (canonical `predictions.jsonl`/`confidence_abstention.jsonl` win over
the mirror by precedence). It then **cross-checks every *other* candidate that also resolves** against the
selected one, restricted to the `phase5-full-pipeline` config: it asserts the three carry the **same
`n_rows`, the same `set(question_id)`, and the same per-id `correctness` (and `abstained`) vectors**. The
realized result is recorded in the manifest as `source_consistency: "ok" | "mismatch" | "single_source"`.
A **mismatch is fatal** (`CALIBRATION_SOURCE_MISMATCH`, exit non-zero) — it means the mirror and the
canonical file disagree and the reported N/P and all downstream calibration numbers would depend on an
unverified choice. This makes source selection deterministic (precedence) *and* verified (equivalence),
closing the "silently select a non-equivalent mirror" gap. The chosen source's absolute path is recorded
in the manifest.

**The join and label (resolved — finding 2):**
- `predictions.jsonl` already carries `reference_answer`, `correctness` (bool), and `dataset`, so the
  **primary** correctness label is read directly from `correctness` (method `token_containment>=0.6`,
  recorded as a limitation). No join is needed for the primary label.
- Where graded, per-type scoring is wanted (secondary/robustness view), the join to `eval_questions.json`
  is: **strip the `<source>::` prefix** from `question_id` (`bioasq::<id>` → `<id>`), look up
  `exact_answer`/`ideal_answer`/`type` by bare `id` in `eval_questions.json`. Verified: phase5 ids are
  namespaced (`bioasq::53372ba3...`, `pubmedqa::15943725`); eval ids are bare (`604900391cb411341a000161`).
- **Non-BioASQ rows are excluded (resolved):** `source == "pubmedqa"` rows (27 of 40 in full-pipeline)
  have **no** entry in the BioASQ-only `eval_questions.json` and no BioASQ `type`. They are **excluded**
  from the BioASQ calibration set; the reduced N is documented. The design does **not** rely on the
  flattened single `gold_answer` string for per-type graded scoring.
- Abstentions (`abstained==true` / empty answer) are **excluded** from the calibration/correctness set
  (they are a separate selective-prediction axis handled by `abstention_precision/recall` and the
  risk-coverage curve). Stated in the report.

**Realized counts after filtering (verified, this is the crux):** for `phase5-full-pipeline`:
40 rows → 13 BioASQ (27 pubmedqa excluded) → **2 non-abstained** → **0 correct**
(`correctness == False` for all 40 rows; `token_containment>=0.6`). `raw_confidence_components` and
`confidence_components` are **null** throughout, so the only usable calibration feature is the scalar
`calibrated_confidence` / `confidence_score`.

### 8.3 Automated correctness evaluator and its acceptance gate (resolved — finding 7)

The primary label is the dataset's own `correctness` (token-containment ≥ 0.6). For the optional graded
secondary view, the per-type evaluator is:
- `yesno`: normalized yes/no polarity match of `predicted_answer` vs `exact_answer`.
- `factoid`/`list`: `exact_match` or containment of any gold `exact_answer` entry in the normalized
  `predicted_answer` (reuse `metrics.exact_match`); `list` counted correct if **≥50%** of gold items
  present (documented threshold).
- `summary`: `compute_rouge_l(predicted_answer, ideal_answer)` F1 ≥ **0.30** (documented threshold).

**Predeclared acceptance gate:** a stratified sample of up to **40 questions** (as many as exist per
type given the tiny universe, seed 42) is written to `calibration/manual_review_sample.jsonl` with the
automated label and the fields a human needs. The evaluator is **"accepted"** only if, when human labels
are supplied in-run (`--manual-labels <path>`), **automated-vs-manual agreement ≥ 0.85** on that sample.
If no human labels are supplied, or agreement < 0.85, **all** downstream calibration numbers are reported
as **provisional/unvalidated regardless of n**, and the `list`/`summary` thresholds (0.5 fraction / 0.30
ROUGE-L) are flagged as the **dominant limitation** in the report. No human-review result is ever
fabricated.

**Honesty note on the gate's realized effect (finding 5):** on the current artifacts the manual gate will
**not** be satisfied — an unattended run supplies no human labels, and the full-pipeline BioASQ set is
~13 questions (≈2 non-abstained), so the "up to 40" review sample collapses to a dozen-ish rows. The gate
therefore resolves to **"provisional by default"** on this data and cannot flip the verdict; the graded
secondary view is provisional regardless. The gate is not wasted effort — it exists for the
`--generate-labels` path (§8.7), where a larger labelled set plus supplied `--manual-labels` can justify
accepting the automated evaluator. The implementer should wire it but not expect it to change the current
verdict.

### 8.4 Degenerate-data policy (resolved — the expected primary outcome)

Let `N` = non-abstained, BioASQ-only, labelled questions in the chosen config, and let `P` = count of
correct (positive) labels among them.

- **If `P == 0` or `N - P == 0` (no positive or no negative class) — the verified current state for
  full-pipeline (`N≈2`, `P==0`):** calibration mapping and threshold *selection* are **not run** (a
  logistic/isotonic fit and a selective-accuracy objective are undefined without both classes). Instead
  the report computes and presents, clearly labelled **"DEGENERATE — single-class / insufficient data,
  provisional, not a calibration claim"**:
  - raw-heuristic ECE and Brier over whatever labelled rows exist (these are still well-defined),
  - the confidence distribution (histogram) and the count released/withheld at the incumbent 0.65,
  - the empirical selective accuracy at 0.65 on the available rows (which, with `P==0`, is 0/answered),
  - an explicit statement that the correctness labeller found **zero correct** answers in this config, so
    no threshold can be validated and the 0.65 proposal **cannot be confirmed or refuted** from this data.
  The report names the exact N, P, and source config.
- **If `0 < N < 15`:** skip the three-way split; fit the calibration map (if both classes present) on all
  labelled rows and report raw-vs-mapped ECE/Brier **flagged diagnostic/provisional**; no threshold
  selection claim; no bootstrap CIs.
- **If `15 ≤ N < 40`:** run the full split+select+test protocol (§8.5–8.6) but mark every number
  **"provisional — insufficient n"** and skip bootstrap CIs.
- **If `N ≥ 40`:** run the full protocol and report bootstrap CIs where the test partition has n ≥ 30.

The thresholds (15, 40, 30) are predeclared here so the branch taken is a deterministic function of the
data, not a judgment call at write-up time.

### 8.5 Partition strategy and the build_splits contract (resolved — findings 3, 4)

Only reached when §8.4 selects the split path (`N ≥ 15` and both classes present). Partitions:
- **fit (60%)** — fit the calibration map and select the calibration method.
- **select (20%)** — select the operating threshold.
- **test (20%)** — untouched until the single final evaluation.

**Exact mapping onto `build_splits` positional slots (finding 4):**
`build_splits(dataset, dev_n, cal_n, test_n, seed=42, stratify_by=...)` returns
`(dev_split, cal_split, test_split)`. We map **design-fit → `dev_n` (returned `dev_split`)**,
**design-select → `cal_n` (returned `cal_split`)**, **design-test → `test_n` (returned `test_split`)**,
with `dev_n = round(0.6N)`, `cal_n = round(0.2N)`, `test_n = N - dev_n - cal_n`. After the call the driver
**asserts** the three returned id-sets are pairwise disjoint and their union equals the labelled set, and
records the realized sizes in the manifest. The variable names in code are `fit_split, select_split,
test_split = build_splits(...)` to make the mapping unambiguous.

**Per-stratum sparsity guard (finding 3):** before any stratified call, the driver computes the per-type
allocation `max(1, round(frac*n))` for each of `dev_n/cal_n/test_n` and checks every `type` stratum has
`count ≥ d+c+t`. **If any stratum is too sparse** (the expected case given 4 types and small N), the
driver falls back to **`stratify_by=None`** (non-stratified) and records `"stratified": false,
"fallback_reason": "sparse stratum <type>=<count>"` in the manifest. As a belt-and-suspenders measure the
`build_splits` call is also wrapped so a `ValueError` is caught and converted into the §8.4
"insufficient n — provisional" path rather than crashing. Seed 42 throughout.

The retrieval-eval queries (§7) are a *different measurement* over the same questions and do **not**
consume these partitions; retrieval metrics are reported over all 250 regardless.

### 8.6 Calibration method selection + threshold protocol (resolved)

**Method selection:** fit **both** Platt (logistic on the single confidence scalar) and isotonic on the
**fit** partition only. Select the method with the lower **ECE on the fit partition** (tie-break: lower
Brier on fit). The rule is fixed in code and recorded. The selected map is applied to select/test. Report
ECE and Brier **before** (raw) and **after** (selected map) on each of fit/select/test, clearly labelled.

**Per-partition single-class degeneracy guard (resolved — finding 3).** The §8.4 branch gates on the
**aggregate** labelled set having both classes, but seeded (possibly stratified) splitting of a small N
can still yield a `select` or `test` partition that is internally single-class (zero positives or zero
negatives). "Selective accuracy on answered" is then `0/answered` (if `P==0` in that partition) or `1`
trivially (if all positive) for **every** threshold, so the objective ties across the whole grid and the
incumbent-0.65 verdict would be a sampling artifact, not a property of the threshold. Therefore, **before**
selecting on `select` or evaluating on `test`, the driver checks that partition's label balance:
- If the **`select`** partition has zero positives or zero negatives: do **not** select a data-driven
  threshold. Record `select_partition: "single-class — threshold not selectable"`, skip to evaluating only
  the incumbent 0.65 on `test` (subject to the same guard below), and mark the result provisional.
- If the **`test`** partition has zero positives or zero negatives: do **not** emit a threshold verdict for
  either the selected threshold or 0.65. Record `test_partition: "single-class — threshold not evaluable"`
  and fall back to the §8.4 provisional/degenerate labelling for the reported numbers (raw-vs-mapped
  ECE/Brier and the confidence distribution are still reported; a *validation* claim is not).
Both guard outcomes are written to `calibration/calibration_report.json` with the exact per-partition
(n, P) counts so the reason is auditable and never mistaken for a threshold property.

**Threshold validation (only when `select` and `test` each contain both classes):**
- Predeclared grid on **calibrated** confidence: `0.30, 0.35, …, 0.90` (step 0.05).
- Predeclared objective: **maximize coverage** (fraction answered = calibrated confidence ≥ threshold)
  **subject to** selective accuracy on answered ≥ **0.80**, evaluated on the **select** partition. If no
  threshold meets the floor, pick the threshold with the highest selective accuracy and report that the
  floor was unreachable.
- **Deterministic tie-break (finding 3):** when several thresholds are tied on the objective (equal
  coverage among those meeting the floor, or equal selective accuracy when the floor is unreachable),
  choose the **smallest** threshold among the tied set. This makes selection a deterministic function of
  the data for any input, including the degenerate near-ties that small N produces.
- **Select on `select`, evaluate once on `test`:** apply the chosen threshold exactly once to the
  untouched `test` partition; report test coverage and selective accuracy. Also report the incumbent
  **0.65** threshold's coverage/selective-accuracy on `test`, so the proposal is validated rather than
  assumed.
- Predeclared acceptance: **0.65 is "validated"** iff on `test` it yields selective accuracy ≥ 0.80 with
  coverage ≥ 0.50; otherwise it is reported as "not meeting criteria" alongside the selected threshold's
  numbers. No label/threshold is tuned on `test`.
- Bootstrap CIs: 1,000 resamples, seed 42, 95% percentile CIs on test selective accuracy, coverage, ECE,
  Brier — **only if** test n ≥ 30; otherwise "n insufficient for CI."
- Risk–coverage curve (`metrics.risk_coverage_curve`) computed on `test` and tabulated.

Outputs: `calibration/calibration_report.json` + `.md`, `calibration/partitions.json` (ids per split for
auditability), `calibration/manual_review_sample.jsonl`.

### 8.7 Optional label generation (user-gated, bounded)

Because the on-disk data cannot support a calibration claim (§8.4), the only route to a defensible
calibration result is **more labelled, non-abstained, correctly-answered BioASQ questions**. This is
expensive (LLM generation) and is therefore **off by default**, behind `--generate-labels
--max-questions M --question-ids-file <path>`. When enabled, the driver runs the **existing** generation
pipeline (single model load per process) over the specified eval questions, writes records with the same
schema to `calibration/generation/*.jsonl` (atomic, resumable by `question_id`), and then re-enters §8.2.
The design does **not** silently trigger this. Diagnostic (§5) and retrieval (§7) never require the LLM.
The report states clearly whether calibration numbers came from reused artifacts or a new generation run,
with counts and timestamps.

---

## 9. Error handling, checkpointing, reproducibility (per operation)

General pattern for every script: structured logging via `app.utils.logger.logger`; a `manifest.json`
per phase capturing seed=42, the config dict + its SHA-256 `config_hash`, input file paths + their
SHA-256, embedding model id, package versions, UTC start/end timestamps, and completed-item counts.
Writes are atomic: write `*.tmp` then `os.replace`; JSONL appends flush per record and are keyed by
`question_id` so a resumed run skips completed items.

| Operation | Failure condition | Recoverable? | Caller receives | Logged |
|---|---|---|---|---|
| Load `eval_questions.json` | missing/corrupt JSON | Fatal | `FileNotFoundError`/`JSONDecodeError`, exit 2 | ERROR |
| Load eval_split / prod chunks (read-only) | missing/inconsistent | Fatal | `FileNotFoundError`/`ValueError` | ERROR |
| eval_split chunk missing `metadata['pmid']` | key absent | Recoverable | fall back to `source` parse; count in `pmid_fallback_used` | WARN |
| Normalize PMID | regex no match on a URL | Recoverable | entry skipped, counted in `unparsable_pmids`, sampled | WARN |
| Embed batch | OOM/model load error | Fatal | `RuntimeError`, exit 3 | ERROR |
| Build index | `ntotal != len(chunks)` or d≠1024 | Fatal | `ValueError`, no save | ERROR |
| Save index | partial write | Recoverable | `*.tmp` discarded, retry; never touches existing `v1` | ERROR |
| PubMed fetch (if ON) | HTTP error/timeout/rate limit | Recoverable | PMID marked `fetch_failed`, counted as coverage gap, continue | WARN |
| Retrieve a query | empty/invalid query | Recoverable | query recorded with 0 retrieved, scores 0, flagged | WARN |
| Calibration source discovery | all configured sources missing (abs paths probed, §2) | Fatal | exit `CALIBRATION_SOURCE_MISSING`; **no** implicit LLM run | ERROR |
| Calibration source equivalence | ≥2 candidates resolve but disagree on n_rows/id-set/correctness for full-pipeline | Fatal | exit `CALIBRATION_SOURCE_MISMATCH`; manifest `source_consistency: mismatch` | ERROR |
| `select` partition single-class | 0 positives or 0 negatives after split | Recoverable | no data-driven threshold; evaluate only 0.65; provisional | WARN |
| `test` partition single-class | 0 positives or 0 negatives after split | Recoverable | no threshold verdict; §8.4 provisional path; ECE/Brier still reported | WARN |
| Correctness label (graded view) | missing `exact_answer`/`type` after join | Recoverable | row excluded from graded view, counted | WARN |
| Calibration data | single-class / `N<15` / `N<40` | Recoverable | §8.4 degenerate/provisional path; no crash | WARN |
| `build_splits` | sparse stratum `ValueError` | Recoverable | non-stratified fallback, else §8.4 provisional path | WARN |
| Calibration fit | sklearn missing | Recoverable | NumPy logistic fallback, record it | WARN |
| Integrity re-check (§10) | checksum mismatch vs baseline | Fatal | `RuntimeError`, exit 4, run marked INVALID | ERROR |

### Input validation rules (external inputs)
- Gold `documents[]`: optional per entry; coerced via `str()`; PMID required to match `\d+$` else
  skipped+counted.
- `top_k`: fixed 5 (int, >0); `DenseRetriever` already raises on ≤0 and on empty/non-str query.
- CLI flags: `--roundtrip` (bool), `--fetch-missing` (bool, default False),
  `--add-training-distractors` (int ≥0, default 0), `--force-rebuild` (bool), `--reranker` (bool,
  default False), `--generate-labels` (bool, default False), `--max-questions` (int ≥1),
  `--question-ids-file` (path, must exist if given), `--manual-labels` (path, must exist if given),
  `--config` (path, must exist). Invalid → argparse error, exit 2.
- phase5 records: `question_id` (str, required), `confidence_score`/`calibrated_confidence` (float in
  [0,1]; out-of-range clamped+warned), `abstained`/`abstain_flag` (bool), `correctness` (bool),
  `source`/`dataset` (str). Missing required field → record skipped+counted.

### Invariant ownership
- **"New artifacts never touch production/eval_split"** — owned by the write layer (all paths rooted at
  the two isolated dirs) and enforced by the §10 integrity check. One enforcement point prevents drift
  from any script.
- **"index.ntotal == len(chunks)"** — owned by `FAISSVectorStore` (asserted on load) and re-asserted
  post-build. ID↔metadata alignment is the root-cause class we are fixing; it must be guarded at the
  store boundary.
- **"article-level dedup before metrics"** — owned by the retrieval-eval scorer (§7.1), not the index.
  The index keeps per-chunk granularity; collapsing to articles is a scoring-time concern so metrics stay
  correct while retrieval stays realistic.
- **"calibration partitions disjoint; test untouched until final"** — owned by the calibration driver via
  `build_splits` + the explicit disjoint/union assertion (§8.5) + a "test opened once" guard flag.
  Prevents the core methodological error of tuning on the reporting set.

---

## 10. Integrity safeguards

Before any work: recompute SHA-256 of the four files in `integrity_baseline.json` (production
`index.faiss`/`chunks.pkl`, eval_split `index.faiss`/`chunks.pkl`); abort if any file is missing or
already differs from baseline (environment pre-tampered). After the full run: recompute and assert all
four still match baseline, and assert eval_split membership (distinct PMID set + chunk count == 58,173) is
unchanged. Results written to `integrity_check.json` with `pre_run_ok`, `post_run_ok`; the final report is
marked INVALID if either fails. Helper: `backend/evaluation/bioasq_eval/integrity.py`.

---

## 11. Deliverable layout

```
backend/results/bioasq_eval_20261018/
  design.md                       # this file
  env_smoke.json                  # (given)
  integrity_baseline.json         # (given)
  integrity_check.json            # pre/post checksums
  diagnostics/
    diagnostic_report.json|.md    # §5, incl. 789/2394/172/250 reconciliation
  retrieval/
    retrieval_metrics.json        # coverage + conditional + end-to-end views
    per_query.jsonl               # resumable per-query records
  calibration/
    partitions.json               # ids per fit/select/test split (seed 42), if split path taken
    manual_review_sample.jsonl    # evaluator-validation sample + acceptance-gate inputs
    calibration_report.json|.md   # ECE/Brier before/after OR degenerate-data report (§8.4); 0.65 verdict
    generation/                   # ONLY if --generate-labels used
  manifests/
    diagnose.manifest.json
    build_index.manifest.json
    retrieval.manifest.json
    calibration.manifest.json
  final_report.md                 # consolidated, honest summary of all phases + limitations

backend/data/indexes/bioasq_retrieval_eval/
  v1/{index.faiss, chunks.pkl, manifest.json}
  pubmed_cache/                   # only if --fetch-missing

backend/evaluation/bioasq_eval/
  __init__.py
  diagnose.py
  build_eval_index.py
  run_retrieval_eval.py
  calibrate_confidence.py
  integrity.py
  tests/test_retrieval_metrics.py
  tests/test_calibration_logic.py
```

The calibration data source is read (anchored per §2, never resolved against cwd) from
`REPO_ROOT/"RAGMedXMA_Chapter5_Validated_Results"/"predictions.jsonl"` /
`REPO_ROOT/"RAGMedXMA_Chapter5_Validated_Results"/"confidence_abstention.jsonl"` /
`BACKEND_DIR/"results"/"phase5_benchmark"/"per_question"/"phase5-full-pipeline.jsonl"` (precedence order,
§8.2; equivalence cross-checked when ≥2 resolve) — all **outside** the two write roots and opened
**read-only**. Each script runs standalone from `backend/` as
`python -m evaluation.bioasq_eval.<name>`; diagnostic, retrieval, and calibration-from-reused-artifacts
all run **without** the generation LLM.

---

## 12. Testability

- **Unit-testable (fast, no model):** article-level dedup + rank preservation; PMID normalization; the
  §7.4 metric scenarios; the phase5-id → eval-id join (`bioasq::<id>` → `<id>`) incl. pubmedqa exclusion;
  per-type correctness logic (table-driven); the §8.4 degenerate-data branch selection on synthetic
  (N,P) inputs; the `build_splits` slot-mapping + disjoint/union assertion + sparse-stratum fallback on
  synthetic typed datasets; calibration method-selection rule on synthetic score/label arrays with a
  known-better map; threshold-selection objective on a synthetic select set.
- **Integration-testable (model load, slower):** build a tiny 10-snippet index and assert round-trip
  retrieval; end-to-end retrieval-eval on a 5-question fixture producing all three views; integrity check
  against a scratch dummy baseline.
- Because scoring (dedup, metrics, join, branch selection, calibration, thresholding) is pure and
  separated from I/O and the model, nearly all methodological logic is unit-testable without GPU/LLM —
  deliberate, and why the scorer/selector live in standalone functions rather than inside the retrieval
  or agent classes.

---

## 13. Decision summary (all resolved — no open questions)

1. **Corpus:** `eval_questions.json` snippets only (local), section-prefixed content; distractors +
   PubMed fetch are opt-in flags, default OFF; coverage reported honestly; dense-only retrieval.
2. **789 figure:** not reproducible from any on-disk artifact; report the canonical 2,394/250 and the
   phase5-subset 172/40, naming the four candidate populations checked (§4).
3. **Calibration source:** resolved against `REPO_ROOT`/`BACKEND_DIR` anchors (never cwd) —
   `REPO_ROOT/RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` → `…/confidence_abstention.jsonl` →
   `BACKEND_DIR/results/phase5_benchmark/per_question/phase5-full-pipeline.jsonl` (precedence),
   **full-pipeline config**; missing-source fatal (`CALIBRATION_SOURCE_MISSING`), never a silent LLM run;
   when ≥2 candidates resolve, equivalence (n_rows/id-set/correctness) is asserted and a mismatch is fatal
   (`CALIBRATION_SOURCE_MISMATCH`), recorded as `source_consistency` in the manifest; primary label =
   dataset `correctness` (`token_containment>=0.6`); graded view joins by stripping `<source>::` and
   looking up `eval_questions`; pubmedqa excluded; abstentions excluded.
4. **Data reality:** full-pipeline yields N≈2 non-abstained BioASQ with **0 correct** → §8.4
   degenerate-data path is the expected outcome; report raw ECE/Brier + confidence distribution +
   0.65 behaviour, labelled provisional/degenerate; no fabricated calibration claim; more labels only via
   explicit `--generate-labels`.
5. **Partitions (if reached):** 60/20/20 → `build_splits` slots `dev_n/cal_n/test_n` =
   fit/select/test; disjoint+union asserted; per-stratum sparsity guard with non-stratified fallback;
   seed 42.
6. **Calibration method:** fit Platt + isotonic on fit partition; pick lower fit-ECE (tie-break Brier);
   report before/after on fit/select/test.
7. **Threshold:** grid 0.30–0.90 step 0.05 on calibrated confidence; objective max coverage s.t.
   selective accuracy ≥ 0.80 on select, **smallest-threshold tie-break**; per-partition single-class guard
   — if `select` or `test` has only one class, no data-driven/validation verdict is emitted and the §8.4
   provisional path is used (counts recorded); otherwise evaluate selected + incumbent 0.65 once on test;
   acceptance = sel.acc ≥ 0.80 & coverage ≥ 0.50; bootstrap CIs only if test n ≥ 30; n-thresholds
   (15/40/30) predeclared.
8. **Evaluator acceptance gate:** automated-vs-manual agreement ≥ 0.85 on the review sample or all
   calibration numbers are provisional; list/summary thresholds flagged dominant limitation.
9. **Layout + reproducibility:** directories above; atomic writes; per-phase config-hash manifests;
   resumable by question_id; diagnostic/retrieval/calibration-from-reuse run without the LLM.
10. **Integrity:** SHA-256 pre/post vs baseline + eval_split membership unchanged; mismatch marks run
    INVALID.
11. **Embedding model:** `BAAI/bge-m3` (1024-dim, `batch_size=16`, section-prefix ON), overriding the
    stale `bge-small` config string; deviation + batch size recorded in manifests.
12. **Retriever construction pinned:** `DenseRetriever.load(index_dir, embedder=BGEEmbedding())` (or
    `vector_store=` keyword); the `store=` re-export docstring form is explicitly rejected.

---

## 14. Responses to the design review (findings 1–10)

**1 [HIGH] Calibration source path does not exist — ADDRESSED.** The non-existent
`backend/results/phase5_20261017_131123/per_question/<config>.jsonl` is removed. §8.2 and §11 now use the
verified sources (`RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` and
`confidence_abstention.jsonl`, with `backend/results/phase5_benchmark/per_question/
phase5-full-pipeline.jsonl` as the in-backend mirror), in a fixed precedence order. Missing source is
**fatal** (`CALIBRATION_SOURCE_MISSING`); a new generation run occurs only behind the explicit
`--generate-labels` flag (§8.7).

**2 [HIGH] Evaluator keyed on fields the records lack — ADDRESSED.** Verified: phase5 records carry
`correctness` (bool), `reference_answer`, `dataset`, `source`, and namespaced `question_id`; they do
**not** carry `exact_answer`/`ideal_answer`/`type`. The **primary label** now reads the records' own
`correctness` field directly (no join). The **graded secondary** view's join is specified: strip
`<source>::`, look up `exact_answer`/`ideal_answer`/`type` by bare id in `eval_questions.json`.
`pubmedqa` rows (no BioASQ entry/type) are **excluded** with documented reduced N; the flattened
`gold_answer` is not used for graded scoring.

**3 [MEDIUM] Stratified build_splits can raise — ADDRESSED.** §8.5 adds a pre-call per-stratum count check
against the `max(1, round(frac*n))` allocation; too-sparse strata trigger a recorded non-stratified
fallback, and the call is additionally wrapped to convert any `ValueError` into the §8.4 "insufficient n —
provisional" path. The §8.4 n-check now predeclares 15/40 thresholds rather than only guarding aggregate N.

**4 [MEDIUM] cal/val/test ↔ dev/cal/test mapping — ADDRESSED.** §8.5 states the exact mapping: design-fit
(60%) → `dev_n`/returned `dev_split`; design-select (20%) → `cal_n`/`cal_split`; design-test (20%) →
`test_n`/`test_split`; code unpacks as `fit_split, select_split, test_split = build_splits(...)` and
**asserts** the three id-sets are disjoint with union == labelled set.

**5 [MEDIUM] DenseRetriever construction keyword — ADDRESSED.** §2 and §7 pin construction to
`DenseRetriever.load(index_dir, embedder=BGEEmbedding())` (or `vector_store=` keyword) and explicitly warn
that the `store=` form in the `app/retrieval/dense_retriever.py` re-export docstring raises `TypeError`.

**6 [MEDIUM] 789 reconciliation — ADDRESSED.** §4 predeclares four concrete candidate populations, reports
the measured count for each (2,394 / 172 / 172 / 145), and states that **none reproduces 789**; the report
presents the canonical 2,394/250 and phase5-subset 172/40 figures and names every artifact checked, making
"could not reproduce 789" a verifiable, evidenced result.

**7 [MEDIUM] Evaluator validation gate — ADDRESSED.** §8.3 predeclares an acceptance gate
(automated-vs-manual agreement ≥ 0.85 on the review sample); below that (or with no human labels supplied)
all calibration numbers are provisional regardless of n, and the `list` (≥50%) / `summary` (ROUGE-L ≥0.30)
thresholds are flagged as the dominant limitation.

**8 [NIT] "other configs available" — ADDRESSED.** Softened: there is a single phase5 universe of 40
questions × 8 ablation configs; §8.2 no longer implies additional per-config files beyond those.

**9 [NIT] batch_size + section-prefix unpinned — ADDRESSED.** §6.3 pins `batch_size=16` (recorded in the
manifest); §6.2 makes section-prefixing **ON** (matching the `[abstract]`/`[sections.0]` convention
visible in phase5 citations) with the exact rule recorded.

**10 [NIT] eval_split `metadata['pmid']` assumed — ADDRESSED.** This revision loaded
`eval_split/chunks.pkl` and verified each `MedicalChunk` carries `metadata['pmid']` (sample `29040384`)
and a full-URL `source`; §5 states this is verified and the run still re-asserts field presence on first
load (with a counted `source`-parse fallback).

**Note on scope beyond the review:** verifying the calibration artifacts revealed that the usable
full-pipeline BioASQ calibration set is N≈2 with **zero correct answers**. The design folds this in as the
expected primary outcome (§8.4) so the implementer produces honest, clearly-labelled provisional results
rather than discovering infeasibility at run time or being tempted to fabricate a calibration claim.

---

## 15. Responses to the current design review (findings 1–7)

**1 [MEDIUM] Calibration source paths repo-root-relative but scripts run from `backend/` — ADDRESSED.**
Verified on disk this revision: `RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` and
`confidence_abstention.jsonl` exist **only** at the repo root (`D:\Ragmed_2\`); the
`backend/RAGMedXMA_Chapter5_Validated_Results/` and `backend/backend/results/...` forms do **not** exist.
§2 now pins two cwd-independent anchors derived from the script's own path —
`BACKEND_DIR = Path(__file__).resolve().parents[2]` and `REPO_ROOT = Path(__file__).resolve().parents[3]` —
and lists the absolute form of each of the three candidates. No path is resolved against the process cwd,
and each loader logs the fully-resolved absolute path of every probed candidate (exists/not-found). §8.2
and §11 use the anchored forms. This removes both the spurious-`CALIBRATION_SOURCE_MISSING` risk and the
silent fall-through to the mirror.

**2 [MEDIUM] Primary vs. mirror source treated as interchangeable without an equivalence check —
ADDRESSED.** §8.2 no longer calls the mirror interchangeable. The loader selects the first candidate by
precedence, then **cross-checks every other resolving candidate** (restricted to `phase5-full-pipeline`)
for identical `n_rows`, `set(question_id)`, and per-id `correctness`/`abstained` vectors. The outcome is
recorded as `source_consistency: ok|mismatch|single_source` in the manifest, the chosen source's absolute
path is recorded, and a **mismatch is fatal** (`CALIBRATION_SOURCE_MISMATCH`). Source selection is now both
deterministic (precedence) and verified (equivalence).

**3 [MEDIUM] Threshold objective undefined when a split partition is single-class — ADDRESSED.** §8.6 adds
an explicit **per-partition degeneracy guard** run before selection/evaluation: if the `select` partition
is single-class, no data-driven threshold is selected (only 0.65 is considered, provisionally); if the
`test` partition is single-class, **no** threshold verdict (selected or 0.65) is emitted and the §8.4
provisional/degenerate path supplies the reported numbers. Both outcomes record the per-partition (n, P)
counts. A deterministic **smallest-threshold tie-break** is added so grid ties resolve unambiguously. This
prevents a sampling artifact from being reported as a threshold property.

**4 [MEDIUM] End-to-end denominator rule stated two inconsistent ways — ADDRESSED.** §7.3 is rewritten to
state **one formula per view** over explicitly-defined `gold(q)` and `available(q)`. Every per-query metric
is computed exactly one way (dedup'd top-5 vs the **full** `gold(q)`); only the denominator query-set
differs: end-to-end = all 250 (uncovered-gold queries contribute 0, MRR via the 250-pair list-of-lists);
conditional = covered queries only. The report additionally prints the counts of uncovered- and
partially-covered queries so every 0 is attributable to coverage versus ranking. The ambiguous "score 0 /
excluded" phrasing is gone.

**5 [NIT] Evaluator gate effectively always "provisional" on real data size — ADDRESSED.** §8.3 adds an
explicit honesty note: on the current artifacts (no human labels in an unattended run, ~13 full-pipeline
questions) the gate resolves to provisional by default and cannot flip the verdict; it exists for the
`--generate-labels` path where a larger labelled set plus `--manual-labels` can justify acceptance.

**6 [NIT] Section-prefix uses `beginSection` only — ADDRESSED.** §6.2 documents this as a deliberate
choice: the prefix uses `beginSection` and **ignores `endSection`** (which the schema permits to differ),
matching the single-section `[abstract]`/`[sections.0]` phase5 convention; the exact rule is recorded
verbatim in the index manifest.

**7 [NIT] Retrieval resume key mixes eval `id` and phase5 `question_id` — ADDRESSED.** §7 now states the
retrieval phase keys on the eval **bare `id`** (written into each `per_query.jsonl` record under
`question_id` for output uniformity but carrying the bare id), while the calibration phase keys on the
namespaced phase5 `question_id` and strips `<source>::` only for the join. The two id spaces are never
mixed in a key comparison.
