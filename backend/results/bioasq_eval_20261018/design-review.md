# Design Review — Fix BioASQ Retrieval Evaluation and Confidence Calibration (RAGMedX-MA)

Reviewer: independent design-review subagent (fresh context).
Design reviewed: `d:\Ragmed_2\backend\results\bioasq_eval_20261018\design.md` (REVISION 3).
Workflow: `wf_f9d894c9ec5422e5`.
Method: every load-bearing on-disk / signature claim in the design was re-verified against the actual
source and artifacts, not taken on the design's word.

## Verdict

**APPROVED.** 0 HIGH, 0 MEDIUM. Three NITs are recorded below; by the mechanical rule (verdict is driven
only by HIGH/MEDIUM count) they do not block, and none was downgraded to dodge the gate — each is genuinely
cosmetic/documentation-level and does not change which code path runs or what number is reported.

This is a mature revision. Revision 3 has already closed the HIGH/MEDIUM findings of two prior review
rounds, and — critically for an independent reviewer — its central factual claims hold up under direct
re-verification on disk. The design resolves every architectural ambiguity with a concrete, deterministic
rule, and its most consequential property (that the available calibration data cannot support a
calibration claim) is treated honestly rather than papered over.

---

## How each required focus area is resolved

### (1) Which corpus the isolated retrieval-eval index is built from, and the gold-coverage consequence — RESOLVED
§6.1 fixes the corpus to the **local `eval_questions.json` snippets only** (not `training13b.json`), with
distractor expansion and PubMed fetch as default-OFF opt-in flags. The gold-coverage consequence is stated
plainly: a gold PMID is present only if some eval question produced a snippet for it, so corpus coverage is
**< 100%** of the 2,394 gold PMIDs, and §5/§7.3 quantify and report it as an explicit "corpus coverage"
view rather than hiding it. Verified: all 250 questions carry ≥1 snippet with keys
`offsetInBeginSection, offsetInEndSection, text, beginSection, document, endSection`, so the ingestion
plan is feasible.

### (2) Answer-level correctness label definition and evaluator validation — RESOLVED
§8.2 reads the **primary** correctness label directly from the dataset's own `correctness` field
(method `token_containment>=0.6`, recorded as a limitation), so no fragile join is needed for the primary
label. The **graded secondary** evaluator (§8.3) is fully specified per type (yes/no polarity;
factoid/list exact-match/containment with a documented ≥50% list threshold; summary ROUGE-L F1 ≥ 0.30) and
is gated by a **predeclared acceptance rule**: automated-vs-manual agreement ≥ 0.85 on a seed-42 review
sample, else all calibration numbers are provisional. §8.3 also honestly notes the gate resolves to
"provisional by default" on the current ~13-question universe. Verified on disk: records carry
`correctness` (bool), `scoring_method: "token_containment>=0.6"`, namespaced `question_id`, `reference_answer`,
`dataset`, `source`; they do NOT carry `exact_answer`/`type`, exactly as the design states, and the graded
view's `<source>::`-strip join to `eval_questions.json` is the correct bridge.

### (3) Separation of retrieval-eval / calibration-fit / final-threshold-test partitions; seed 42 — RESOLVED
§8.5 maps design-fit→`dev_n`, design-select→`cal_n`, design-test→`test_n` onto the verified
`build_splits(dataset, dev_n, cal_n, test_n, seed=42, stratify_by)` contract, unpacks as
`fit_split, select_split, test_split`, and **asserts the three id-sets are pairwise disjoint and their
union equals the labelled set**. The test partition is opened exactly once (§8.6, "test opened once"
guard). Retrieval-eval queries are explicitly a different measurement over all 250 questions and do not
consume calibration partitions. No test labels/outcomes feed selection. Seed 42 is locked everywhere
(§2). Crucially, on the current data (N≈2, P=0) the split path is **not** reached at all (§8.4), so there
is no leakage surface in practice; the protocol is nonetheless fully specified for the `--generate-labels`
path.

### (4) Calibration method selection on calibration data only — RESOLVED
§8.6 fits both Platt and isotonic on the **fit** partition only, selects the lower fit-ECE (tie-break
lower fit-Brier), and applies the frozen map to select/test. Selection never sees select/test labels.

### (5) Threshold protocol — RESOLVED
§8.6 predeclares the grid (0.30–0.90 step 0.05 on calibrated confidence), the objective (maximize coverage
s.t. selective accuracy ≥ 0.80 on `select`), a deterministic smallest-threshold tie-break,
select-on-select / evaluate-once-on-untouched-test, explicit incumbent-0.65 validation acceptance
(sel.acc ≥ 0.80 & coverage ≥ 0.50 on test), bootstrap CIs (1,000 resamples, seed 42) only when test
n ≥ 30, and a per-partition single-class guard that refuses to emit a verdict when `select` or `test` is
single-class. Insufficiency is handled by the predeclared 15/40/30 n-thresholds in §8.4. This fully
satisfies the "predeclared range, documented objective, select-then-evaluate-once, CIs, explicit
insufficiency handling" requirement.

### (6) Nothing touches the protected production index or eval_split; artifacts isolated — RESOLVED
§3 and §10 forbid any write outside the two isolated roots and enforce SHA-256 pre/post integrity checks
plus an eval_split membership (distinct-PMID + chunk-count) check, marking the run INVALID on any
mismatch. Verified: `integrity_baseline.json` contains SHA-256 for exactly the four protected files
(production `index.faiss`/`chunks.pkl`, eval_split `index.faiss`/`chunks.pkl`), and both eval_split
artifacts exist on disk. All read-only inputs are opened read-only; all outputs are rooted under
`backend/data/indexes/bioasq_retrieval_eval/` and `backend/results/bioasq_eval_20261018/`.

### (7) Reproducibility + checkpoint/resume + atomic writes + diagnostic-only runs without the model — RESOLVED
§9 specifies per-phase config-hash manifests (seed 42, input SHA-256s, versions, timestamps,
completed-item counts), atomic `*.tmp` → `os.replace` writes, per-`question_id` resumable JSONL appends,
and a single model load per process. §5 (diagnostic), §7 (retrieval, model only for retrieval itself),
and calibration-from-reused-artifacts all run **without** the generation LLM; a new generation run occurs
only behind the explicit `--generate-labels` flag (§8.7). The id-space resume-key issue (§7 vs §8) is
disambiguated: retrieval keys on the bare eval `id`, calibration on the namespaced phase5 `question_id`.

### (8) Raw confidence treated as uncalibrated (ECE/Brier before AND after) — RESOLVED
§8.1 explicitly treats the Agent-7 composite as **uncalibrated** and never modifies the agent's weights or
threshold. §8.6 reports ECE and Brier **before (raw)** and **after (selected map)** on each partition. In
the degenerate current state (§8.4) raw-heuristic ECE/Brier are still reported (well-defined), labelled
"DEGENERATE — provisional, not a calibration claim." Verified: the agent's weights (0.25/0.30/0.25/0.20)
and `CONFIDENCE_THRESHOLD = 0.65` match §8.1 exactly.

### No fabrication; "PMID present" ≠ "retrieved" — RESOLVED (would have been HIGH if violated)
§7.3 states repeatedly that **mere presence of a PMID in the corpus is never counted as a successful
retrieval — only top-5 article-rank membership counts**, and prints uncovered/partial-coverage counts so
every 0 is attributable. §5/§8.4 never fabricate: the design makes the honest "insufficient data / no
positive class → provisional" path the expected primary outcome. No fabrication pattern is present
anywhere; nothing equates corpus presence with retrieval success.

---

## Findings

1. **[NIT] §14 finding-1 response contains a false "does not exist" claim.**
   §14 (and §8.2) state that the Revision-1 path
   `backend/results/phase5_20261017_131123/per_question/<config>.jsonl` "does not exist." It **does** exist
   on disk (`backend/results/phase5_20261017_131123/per_question/phase5-full-pipeline.jsonl` is present, as
   is one under `phase5_20261018_053140/`). This does not harm the plan — the design deliberately pins a
   fixed three-candidate precedence list that excludes these timestamped dirs — but the stated
   justification ("removed because it does not exist") is factually wrong; it was removed because it is a
   non-canonical timestamped run, not because it is absent.
   **Fix:** reword §14/§8.2 to "removed because it is a non-canonical, timestamped run superseded by the
   `phase5_benchmark` mirror," rather than claiming non-existence.

2. **[NIT] A fourth on-disk `predictions.jsonl` candidate is not acknowledged.**
   `backend/results/chapter5_package/predictions.jsonl` also exists and is a plausible calibration source,
   but the design's three-candidate precedence list does not mention it. The fixed precedence makes
   discovery deterministic regardless, so this is not a correctness defect, but an implementer scanning the
   repo may be surprised the design is silent on it.
   **Fix:** add one sentence to §8.2 noting `backend/results/chapter5_package/predictions.jsonl` exists but
   is intentionally excluded from the precedence list (and why), so the omission is clearly deliberate.

3. **[NIT] Per-stratum sparsity pre-check formula may diverge from the realized `build_splits` allocation.**
   §8.5's pre-call guard checks each stratum against `max(1, round(frac*d/c/t))`, but the verified
   `build_splits` runs an `_adjust` pass that can shift per-stratum allocations up/down to hit exact totals,
   so a stratum that passes the design's pre-check could still raise inside `build_splits`. The design's
   belt-and-suspenders `ValueError` wrapper (→ §8.4 provisional path) fully covers this, so it never
   crashes; the pre-check is simply not a perfect predictor of the internal raise.
   **Fix:** note in §8.5 that the pre-check is advisory and the authoritative guard is the `ValueError`
   wrapper, so no one mistakes the pre-check for a guarantee that the stratified call will succeed.

---

## Verified Assumptions

- **Metrics (`evaluation/metrics.py`)** — signatures and semantics match §2 exactly:
  `precision_at_k(retrieved_ids, relevant_ids, k)` denominator `k`;
  `recall_at_k(...)` denominator `|relevant|`;
  `mrr(list[list[str]], list[list[str]])` denominator = #queries;
  `ndcg_at_k(...)` binary relevance, returns 0.0 when no relevant;
  `ece(confidences, correct_flags, n_bins=10)` equal-width bins;
  `brier_score`, `risk_coverage_curve`, `exact_match`, `compute_rouge_l` all present.
- **`DenseRetriever`** — `__init__(embedder, vector_store)`; `retrieve(query, top_k)` raises `TypeError`
  on non-str, `ValueError` on empty/`top_k<=0`; classmethod `load(directory, embedder)`. The design's
  pinned `DenseRetriever.load(index_dir, embedder=BGEEmbedding())` and `vector_store=` keyword are
  correct, and its warning that the re-export docstring's `store=` form raises `TypeError` is accurate
  (the docstring at `app/retrieval/dense_retriever.py` does use `store=`).
- **`BGEEmbedding`** — `DEFAULT_MODEL = "BAAI/bge-m3"`, `normalize=True`, `max_length=512` hard-coded,
  mean pooling, L2-normalized, `embed(texts, batch_size=8)` accepts a `batch_size` override (design uses
  16), `BGEEmbedder = BGEEmbedding` alias present.
- **`FAISSVectorStore`** — `__init__(dimension=1024)` `IndexFlatIP`; `search` returns
  `list[(MedicalChunk, score)]` ordered high→low; `add(embeddings, chunks)`; `save(dir)`;
  `load(dir)` derives `d` from the index and asserts `ntotal == len(chunks)`; position in `chunks` ==
  FAISS id.
- **`app/config.py`** — `embedding_model_name = "BAAI/bge-small-en-v1.5"`, confirming the design's claim
  that config is stale for this purpose and the index must instantiate `BGEEmbedding()` (bge-m3) directly.
- **Confidence agent** — weights `0.25 / 0.30 / 0.25 / 0.20` and `CONFIDENCE_THRESHOLD = 0.65` match §8.1.
- **`build_splits`** — signature `(dataset, dev_n, cal_n, test_n, seed=42, stratify_by)` → `(dev, cal,
  test)`; raises `ValueError` on insufficient data, duplicate ids, and sparse strata; stratified path runs
  an `_adjust` pass. Matches §2/§8.5.
- **`eval_questions.json`** — 250 questions; keys `body, documents, ideal_answer, exact_answer, type, id,
  snippets`; bare `id`; `documents[]` are PubMed URLs with trailing PMID (regex `(\d+)$` applies); all 250
  carry ≥1 snippet; snippet keys `offsetInBeginSection, offsetInEndSection, text, beginSection, document,
  endSection`.
- **Calibration sources** — `RAGMedXMA_Chapter5_Validated_Results/predictions.jsonl` and
  `confidence_abstention.jsonl` exist at the **repo root** (`d:\Ragmed_2\`); the in-backend mirror
  `backend/results/phase5_benchmark/per_question/phase5-full-pipeline.jsonl` exists. Records carry the
  fields §8.2 lists, with `confidence_components`/`raw_confidence_components` = null and namespaced
  `question_id`.
- **The calibration crux (§8.2/§8.4)** — directly recomputed from disk: full-pipeline = 40 rows →
  **13 BioASQ** → **2 non-abstained** → **0 correct**; 0 correct across all 40. The degenerate single-class
  path is the correct, honest outcome.
- **`integrity_baseline.json`** — contains SHA-256 for exactly the four protected files; both production
  and eval_split artifacts exist on disk.

## Unverified / Wrong Assumptions

- **WRONG (minor, NIT 1):** the design's claim that
  `backend/results/phase5_20261017_131123/per_question/<config>.jsonl` "does not exist" is false — the file
  is present. The plan is unaffected because that path is excluded from the pinned precedence list anyway.
- **UNVERIFIED (not independently re-run; design claims these were verified this revision):**
  - eval_split `chunks.pkl` holds exactly **58,173** `MedicalChunk` objects, each carrying
    `metadata['pmid']` (sample `29040384`) and a full-URL `source`. Loading the pickle requires the
    backend on `sys.path` plus the chunk model import, which this review did not execute; the artifacts
    exist and the design re-asserts field presence at run time with a counted `source`-parse fallback, so
    the risk is low and self-correcting if the field is ever absent.
  - The §4 figure reconciliation (2,394 distinct gold PMIDs over 250 questions; 172 over the 40-question
    phase5 subset; 145 for the 13-question BioASQ subset) was not recomputed here. These are reported by
    the diagnostic at run time, and the design correctly declines to treat the unreproducible "789" as a
    target — a defensible, non-fabricating stance regardless of the exact recomputed values.
  - `env_smoke.json` claims (Python 3.14 at `.venv`, `cwd = backend`, `syspath_insert_backend` the only
    working import strategy, package versions) were not re-executed. They affect operational mechanics, not
    methodology, and the §2 anchor-based path rule (`parents[2]`/`parents[3]`) is robust to the cwd
    regardless.
