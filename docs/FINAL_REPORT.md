# RAGMedX-MA — Final Report

**Branch:** `yuvraj/llm`
**Date:** 2026-10-17
**Scope:** Phases 0–6 (read-only audit → Workstreams D/B/A/C → GPU experiment → documentation)
**Honesty policy:** Every result carries one evidence tier
(SOFTWARE TEST | SYNTHETIC FIXTURE | AUTOMATED MODEL JUDGE | BENCHMARK GROUND TRUTH |
HUMAN-VALIDATED) or is marked `NOT MEASURED` with a reason. Tiers are never combined.
No unsupported superiority or clinical-safety claim is made.

---

## A. Executive summary

RAGMedX-MA is a medical retrieval-augmented-generation (RAG) backend built on a
7-agent LangGraph pipeline over hybrid retrieval (BGE-M3 dense + BM25 sparse + RRF),
cross-encoder reranking, and a 4-bit-quantised Qwen3-8B generator. Phases 0–6
hardened the evaluation and verification machinery and ran a controlled multi-config
GPU experiment.

What is established with high confidence (SOFTWARE TEST):

- Checkpoint/resume reliability under fault injection (Phase 1, 45 tests).
- A validated metrics + statistics library — accuracy, ROUGE-L, retrieval metrics,
  calibration (ECE/Brier), risk-coverage, Wilson CI, McNemar, paired bootstrap,
  Holm correction (Phase 2, 130 tests).
- Evidence verification, claim verification, contradiction detection, sufficiency,
  and reranker structural fixes with a documented sildenafil/Prolia regression
  (Phase 3, 20 tests).
- Agent-contribution analysis, a standalone query cache (default OFF), and a fixed
  profiling script (Phase 4, 67 tests across 4a/4b).

What is only **directional** (small pilot): the Phase 5 GPU experiment ran a
**budget-bounded pilot of 16 test questions × 8 configs** — not the full 500×8 sweep
(which is ~40+ h and exceeds the GPU budget; it was deliberately not run and the
command is recorded). At n=16 no paired comparison is Holm-significant, so **no
config is claimed superior**.

Key honest limitations: MedRGB absent (NOT MEASURABLE), PubMedQA retrieval-quality
not measurable in the current eval index, human annotation pending, and token-
containment correctness penalises abstention (a measurement property, not a safety
judgment).

---

## B. System architecture

- **7-agent LangGraph pipeline** (`backend/app/agents/orchestrator.py`):
  Query Understanding → Iterative Retrieval / Multi-Gap Analysis → Evidence
  Reranking → Generator → Critic → Confidence Calibration → Abstention Decision.
- **Retrieval** (`backend/app/rag/`): BGE-M3 dense embeddings + BM25 sparse,
  combined with Reciprocal Rank Fusion; FAISS index. Evaluation runs exclusively
  against `backend/data/faiss_index/eval_split/` (guarded; production index is
  never touched).
- **Reranking:** cross-encoder (CPU) plus an LLM reranker with bounded retry and
  safe fallback to original order (Phase 3b).
- **Generation:** Qwen3-8B, 4-bit NF4 quantisation, `generator_temperature=0.0`
  (GPU, `.venv-cu128-pilot`, torch 2.11.0+cu128 / CUDA 12.8).
- **Provenance:** `EvidenceSnippet` carries `index_id` + `retrieval_iteration`;
  `AgentState` carries `index_id` + `pipeline_version` (Phase 3a).
- **New (default OFF, versioning-compliant):** `record_node_timing` (real per-stage
  `duration_ms`), `pipeline_version="v2"`, and `RAGPipeline.build(cache_enabled=…)`
  consulting `backend/caching/query_cache.py` (Phase 5 flags).

---

## C. Phase-by-phase status (0–6)

| Phase | Workstream | Status | Evidence |
|---|---|---|---|
| 0 | Read-only audit | IMPLEMENTED AND TESTED | `docs/audit_20261017_005421.md` |
| 1 | D — Checkpoint/resume + fault injection | IMPLEMENTED AND TESTED | 45 tests pass |
| 2 | B — Metrics + statistics + leakage/split | IMPLEMENTED AND TESTED | 130 pass, 1 skip |
| 3 | A — Verification/claim/contradiction/provenance | IMPLEMENTED AND TESTED | 20 tests pass + regression |
| 4 | C — Critic analysis, cache, profiling | IMPLEMENTED AND TESTED | 67 tests (25 + 42) + reports |
| 5 | 8-config GPU experiment | IMPLEMENTED AND TESTED (budget-bounded pilot n=16; full 500×8 NOT RUN) | `backend/results/phase5_20261017_131123/` |
| 6 | Docs, component table, re-checks, final report | IMPLEMENTED | this report + re-checks below |

Commits on `yuvraj/llm` (no push): Phase 4 `5f080c57`, Phase 5 `a3bc7457`,
Phase 5 pilot `694e7168`, Phase 6 (this report; see tail of PROGRESS.md).

---

## D. Metrics methodology, denominators, evidence tiers

- **Correctness (Phase 5):** token-containment vs gold answer, threshold 0.6 →
  BENCHMARK GROUND TRUTH. Denominator = `n_measured` (16 pilot questions per config).
  An abstention counts as incorrect under this metric.
- **Correctness (n30 audit):** ROUGE-L F1 (SOFTWARE TEST). ROUGE-L is surface
  similarity, not a correctness or safety metric.
- **Abstention rate:** abstentions / total questions (SOFTWARE TEST).
- **Citation support rate (n30):** supported citations / total citations as recorded
  in the n30 metrics block (SOFTWARE TEST); known denominator-labelling caveat
  carried from Phase 0/2.
- **Retrieval metrics (R@K / MRR / nDCG):** implemented and unit-tested in
  `backend/evaluation/metrics.py` (Phase 2). In Phase 5 they are **NOT MEASURED**:
  PubMedQA gold PMIDs are not in the BioASQ-built eval index, and the BioASQ subset
  at n=16 is too small to report. Reason recorded in the dataset manifest.
- **Calibration (ECE / Brier):** implemented + tested (Phase 2); NOT MEASURED at
  n=16 (insufficient samples). `calibrated_confidence` is recorded per answered
  item; when calibration is disabled, confidence is a synthetic 1.0 and is labelled.
- **Statistics:** Wilson 95% CI on every pilot accuracy; paired McNemar vs
  `phase5-full-pipeline`, Holm-corrected across the 7 comparisons.
- **Evidence tiers:** never combined in a cell. `NOT MEASURED` always carries a
  reason. No AUTOMATED MODEL JUDGE tier was fabricated where the underlying judge
  data (e.g. per-question critic verdicts) is absent.

---

## E. Phase 5 results (8-config comparison)

Source: `backend/results/phase5_20261017_131123/metric_summary.json`,
`latency_breakdown.json`, `environment.json`. **Budget-bounded pilot, n=16 test
questions × 8 configs** (12 PubMedQA + 4 BioASQ, seed 42, stratified). Wall-clock
3.691 h; run peak VRAM 10,157 MB; 0 failures; `llm_mode==LOCAL` hard-asserted;
leakage canary PASS (first retrieved PMID 17128671 not in the eval-question set).

### E.1 Correctness + abstention (BENCHMARK GROUND TRUTH; ST for abstention)

| Config | n | correct | accuracy | 95% Wilson CI | abstention |
|---|---|---|---|---|---|
| phase5-plain-llm-no-rag | 16 | 6 | 0.375 | [0.185, 0.614] | 0.0 |
| phase5-basic-rag | 16 | 1 | 0.0625 | [0.011, 0.283] | 0.75 |
| phase5-no-critic | 16 | 1 | 0.0625 | [0.011, 0.283] | 0.75 |
| phase5-full-pipeline | 16 | 0 | 0.0 | [0.0, 0.194] | 0.9375 |
| phase5-improved-v2 | 16 | 0 | 0.0 | [0.0, 0.194] | 0.9375 |
| phase5-improved-no-verification | 16 | 0 | 0.0 | [0.0, 0.194] | 0.9375 |
| phase5-no-iterative-retrieval | 16 | 0 | 0.0 | [0.0, 0.194] | 0.9375 |
| phase5-no-calibration | 16 | 0 | 0.0 | [0.0, 0.194] | 0.75 |

### E.2 Paired McNemar vs `phase5-full-pipeline` (Holm-corrected)

All comparisons: raw p ∈ {1.0, 0.03125}; **every Holm-adjusted p ≥ 0.219 → none
significant at 0.05**. plain-llm-no-rag raw p=0.03125 → Holm p=0.219 (NOT
significant). **No superiority claim.**

### E.3 Latency vs PRD (SOFTWARE TEST)

Cold RAG means ≈ 63–141 s/question (plain-llm ≈ 9.6 s) — far above the PRD
single-pass < 5 s target on this hardware/model (reported honestly, not a pass).
Warm (cache hit): improved configs 0.12 ms and 19.62 ms — the query cache delivers
the expected warm-path speedup. Per-stage timing is populated only for the two
improved (v2) configs (`record_node_timing` ON); baselines carry NOT MEASURED
per-stage (flag OFF, reproducibility preserved).

### E.4 Cold vs warm

Cold = first pass / cache miss; warm = cache hit on the improved pipeline. Only the
improved configs (cache ON) have warm measurements; historical baselines (cache OFF)
are NOT MEASURED for warm by design.

**Interpretation (honest):** at n=16 the RAG configs abstain heavily on this
BioASQ/PubMedQA mix, and token-containment penalises abstention, so the always-
answering plain-LLM shows the highest raw accuracy. This is a measurement property;
it is not evidence that abstention is wrong or that plain-LLM is clinically better.
The full 500×8 sweep is required before any correctness conclusion.

---

## F. Agent-contribution / critic analysis (Phase 4a)

Source: `backend/results/critic_analysis_20261017_063736.json` (n30 ablation).

- Per-config abstention (SOFTWARE TEST): full-pipeline 0.5667, no-critic 0.3333,
  no-calibration 0.3333, basic-rag 0.4667, no-iterative-retrieval 0.6000,
  plain-llm-no-rag 0.0.
- Cross-config deltas (SOFTWARE TEST, population-level, n=30): critic net abstention
  Δ = 0.2334 (full − no-critic); calibration net abstention Δ = 0.0; ROUGE-L Δ
  (full − no-critic) = −0.0443.
- **NOT MEASURED (reason):** per-question `critic_accept/reject/rewrite`,
  `draft_answer`, and ROUGE-L draft-vs-final — these were never saved by
  `IncrementalResultWriter`. No AUTOMATED MODEL JUDGE tier was produced because the
  per-question critic-verdict data needed for that tier is absent.
- **Caveat:** cross-config deltas are not controlled per-question comparisons; they
  do not isolate the critic from retrieval/generation variance; at n=30 they are not
  statistically reliable; a McNemar test cannot be applied without per-question
  critic decisions. Three response-type-change examples (sildenafil, vivotif,
  mitochondria nanotubes) are recorded with the explicit note that the cause cannot
  be determined from the saved files.

---

## G. Caching + profiling (Phase 4b / 4c)

- **Caching (4b):** `backend/caching/query_cache.py` is a standalone, file-backed
  SHA-256-keyed cache (key = query + index_identity + retrieval_config +
  model/embedder/prompt version + preprocessing). It is **default OFF**: the default
  retrieval/agent pipeline imports nothing from `caching` at module load (verified by
  a fresh-subprocess `sys.modules` check); the cache is imported lazily only inside
  the `cache_enabled=True` branch. 42 cache tests pass. In Phase 5 the cache is wired
  only into the improved (v2) configs, where it produced the warm-path speedup.
- **Profiling (4c):** fixed the double `profiler.enable()` and a dead
  `pstats.Stats(...)` bug in `backend/scripts/generate_profile.py`. New report
  `backend/results/profile_20261017_071311.md` (SOFTWARE TEST, mock/fixture, 5 Q,
  25.231 s). Dominant measured cost on the mock run is LangGraph/async overhead;
  real GPU cost is in the n30/Phase-5 latency data. Recommended optimisations are
  NOT MEASURED (proposed, not benchmarked).

---

## H. Component table

See `docs/component_results.md` for the full table with columns:
component | correctness | retrieval | claim support | hallucination | abstention |
latency | memory | uncertainty | recommendation. Summary:

| Component (config) | Correctness | Abstention | Latency (cold) | Memory | Recommendation |
|---|---|---|---|---|---|
| phase5-improved-v2 | 0.0 [0.0,0.194] (BGT) | 0.9375 | 137,773 ms (warm 19.62 ms) | 10,157 MB | Cache warm-path works; correctness non-informative at n=16 |
| phase5-plain-llm-no-rag | 0.375 [0.185,0.614] (BGT) | 0.0 | 9,625 ms | 10,157 MB | Highest raw accuracy, never abstains; not Holm-significant |
| phase5-full-pipeline | 0.0 [0.0,0.194] (BGT) | 0.9375 | 140,713 ms | 10,157 MB | Baseline; high abstention on this mix |

Retrieval / claim-support (per-claim) / hallucination / calibration cells are
NOT MEASURED in the pilot (reasons in the component table). The n30 audit sub-table
provides ROUGE-L, citation support rate, abstention, and latency for the 6 historical
configs (all SOFTWARE TEST).

---

## I. Honest limitations

- **Sample size:** Phase 5 is a 16-question pilot, not the 500×8 sweep. All pilot
  numbers are directional; CIs are wide; no paired test is Holm-significant.
- **MedRGB:** absent from the repo and from MIRAGE → MedRGB metrics NOT MEASURABLE.
- **Retrieval-quality measurability:** PubMedQA gold PMIDs are not in the BioASQ-built
  eval index → R@K/MRR/nDCG NOT MEASURED for PubMedQA; BioASQ subset too small at n=16.
- **Human annotation:** abstention precision/recall, false-abstention/acceptance, and
  citation faithfulness need human labels (schema exists, `annotation_status: pending`).
- **NLI model:** `cross-encoder/nli-*` not installed in the CPU venv; claim-verifier
  NLI tests use mocks; `enable_claim_verification` default OFF.
- **Metric property:** token-containment penalises abstention; read correctness with
  abstention rate, never alone.
- **Profiling:** mock/fixture only; recommendations not benchmarked.
- **Integration tests:** `test_local_llm.py`, `test_langgraph_real_seed_integration.py`,
  `test_native_retrieval_integration.py` require a real local LLM / seed FAISS corpus
  and fail in the CPU-only test environment. They are pre-existing (committed before
  this work, outside the Phase 1–4 scope) and are not a regression from Phases 0–6.

---

## J. Reproducibility

- **Versioning rule honoured:** all new behaviour is behind flags default OFF under a
  new pipeline version `v2` with new experiment IDs; the six historical configs stay
  byte-for-byte reproducible. Tests prove default-OFF identical behaviour
  (`test_phase5_flags_default_off.py`, `test_workstream_c_cache.py`).
- **Seeds:** split and pilot sampling use seed 42, stratified by source.
- **Protected assets never modified:** production FAISS index, `backend/prompts/v1/*`,
  the 6 n30 result JSONs, the checkpoint — all verified unchanged (Section below).
- **Interpreters:** CPU `.venv` (3.14.3) for tests/analysis; GPU `.venv-cu128-pilot`
  (3.11.0, torch 2.11.0+cu128 / CUDA 12.8) for the experiment.
- **Environment metadata:** `backend/results/phase5_20261017_131123/environment.json`
  (index hash/mtime, canary, LLM mode/quantisation/temperature, versions).
- **Full-sweep command (not run):**
  `d:\Ragmed_2\.venv-cu128-pilot\Scripts\python.exe backend\scripts\run_phase5_experiment.py --full --output-root D:\Ragmed_2\backend\results\phase5_20261017_131123`.

---

## K. Recommendations

Framed as engineering next steps, not clinical or superiority claims:

1. **Run the full 500×8 sweep** before any correctness/superiority conclusion — the
   pilot n is too small for inference.
2. **Enrich per-sample logging** (`draft_answer`, `critic_verdict`, per-agent
   `duration_ms`) so critic accept/reject/rewrite and draft-vs-final become
   measurable rather than NOT MEASURED.
3. **Add a retrieval-quality-measurable split** (gold PMIDs present in the eval index)
   to report R@K/MRR/nDCG honestly.
4. **Install the NLI model** and enable `enable_claim_verification` to move claim
   support from NOT MEASURED to AUTOMATED MODEL JUDGE.
5. **Complete human annotation** to unlock HUMAN-VALIDATED abstention/faithfulness
   metrics.
6. **Treat latency honestly:** cold RAG latency is far above the PRD target on this
   hardware; the query cache helps only the warm path. Investigate the gap-analysis
   stage, which dominates per-stage timing in the improved configs.

---

## Appendix — Protected-asset re-check (Phase 6, P6-3)

See `docs/progress/PROGRESS.md` Phase 6 for the recorded command outputs. Summary:
production index hash/mtime unchanged, v1 prompt hashes unchanged, the 6 n30 JSON
hashes unchanged, branch `yuvraj/llm`, torch 2.11.0+cu128 / CUDA 12.8 unchanged vs P5-1.
