# RAGMedX-MA — Component Results Table

**Last updated:** 2026-10-17 (Phase 6)
**Branch:** `yuvraj/llm`

## Purpose

This table summarises per-component (per-config) results for the RAGMedX-MA
multi-agent medical-RAG pipeline. Every numeric cell is tagged with an evidence
tier; `NOT MEASURED` is used wherever data is genuinely absent (with a reason).
No tiers are combined in a single cell, and no superiority or clinical-safety
claim is made.

## Evidence tiers (legend)

- **BGT** — BENCHMARK GROUND TRUTH (token-containment vs gold answer, threshold 0.6; Phase 5 pilot)
- **ST** — SOFTWARE TEST (direct file parsing / wall-clock measurement; no model judgment)
- **SI** — SOFTWARE INSPECTION (static source analysis)
- **NM** — NOT MEASURED (data genuinely absent; reason given)

## Data provenance

- **Phase 5 pilot** — `backend/results/phase5_20261017_131123/` (`metric_summary.json`,
  `latency_breakdown.json`, `environment.json`, `summary.json`). Budget-bounded
  pilot: **n=16 test questions × 8 configs** (not the full 500×8 sweep, which
  exceeds the GPU budget and was deliberately not run). Treat all pilot numbers
  as **directional only** — wide Wilson CIs, no Holm-significant superiority.
- **Phase 4 critic analysis** — `backend/results/critic_analysis_20261017_063736.json`
  (n30 ablation files; abstention counts + cross-config deltas).
- **Phase 4 profile** — `backend/results/profile_20261017_071311.md` (cProfile on a
  mock/fixture run; SOFTWARE TEST for profile data, NM for GPU latency).
- **n30 audit** — `backend/data/results/step4_*_n30_20261008_010119.json`
  (6 configs × 30 Q; `rouge_l_f1`, `abstention_rate`, `citation_support_rate`,
  `latency_stats.avg_ms`).

---

## Component table — Phase 5 pilot (n=16 test questions × 8 configs)

Columns: component | correctness | retrieval | claim support | hallucination | abstention | latency | memory | uncertainty | recommendation.

| Component (config) | Correctness | Retrieval (R@K / MRR / nDCG) | Claim support | Hallucination | Abstention | Latency (cold mean) | Memory (peak VRAM) | Uncertainty (calibration) | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| phase5-improved-v2 (all new flags ON, v2) | 0.0 [0.0, 0.194] (BGT) | NM — PubMedQA PMIDs not in eval index; BioASQ subset too small to report at n=16 | NM — not scored at per-claim level in pilot | NM — no human/NLI hallucination labels on pilot outputs | 0.9375 (ST) | 137,773 ms cold; warm 19.62 ms (ST) | 10,157 MB run peak (ST) | calibrated_confidence present (e.g. 0.8425 on one answered item); ECE NM at n=16 | Keep ON for cache/warm-path; correctness at n=16 is non-informative (high abstention). Needs full 500-sweep to judge. |
| phase5-improved-no-verification (improved minus claim verification) | 0.0 [0.0, 0.194] (BGT) | NM (same as above) | NM | NM | 0.9375 (ST) | 125,242 ms cold; warm 0.12 ms (ST) | 10,157 MB run peak (ST) | calibrated confidence present; ECE NM | Cache warm-path works (0.12 ms); no correctness signal at n=16. |
| phase5-full-pipeline (historical baseline) | 0.0 [0.0, 0.194] (BGT) | NM | NM | NM | 0.9375 (ST) | 140,713 ms cold; warm NM (cache OFF) (ST) | 10,157 MB run peak (ST) | NM — per-stage timing OFF on baselines | Baseline for paired tests; high abstention on this mix. |
| phase5-no-critic (historical baseline) | 0.0625 [0.011, 0.283] (BGT) | NM | NM | NM | 0.75 (ST) | 123,891 ms cold; warm NM (ST) | 10,157 MB run peak (ST) | NM | Lower abstention than full-pipeline; no significant correctness delta (Holm p=1.0). |
| phase5-no-calibration (historical baseline) | 0.0 [0.0, 0.194] (BGT) | NM | NM | NM | 0.75 (ST) | 140,147 ms cold; warm NM (ST) | 10,157 MB run peak (ST) | synthetic confidence 1.0 when calibration OFF (labelled) | No significant delta vs baseline (Holm p=1.0). |
| phase5-no-iterative-retrieval (historical baseline) | 0.0 [0.0, 0.194] (BGT) | NM | NM | NM | 0.9375 (ST) | 89,239 ms cold; warm NM (ST) | 10,157 MB run peak (ST) | NM | Fastest RAG cold path; no significant delta (Holm p=1.0). |
| phase5-basic-rag (historical baseline) | 0.0625 [0.011, 0.283] (BGT) | NM | NM | NM | 0.75 (ST) | 63,743 ms cold; warm NM (ST) | 10,157 MB run peak (ST) | NM | No significant delta (Holm p=1.0). |
| phase5-plain-llm-no-rag (historical baseline) | 0.375 [0.185, 0.614] (BGT) | N/A — no retrieval | 0.0 (ST, no citations) | NM | 0.0 (ST) | 9,625 ms cold; warm NM (ST) | 10,157 MB run peak (ST) | N/A — no calibration path | Highest token-containment accuracy but NEVER abstains; raw McNemar p=0.03125 → **Holm p=0.219, NOT significant**. See metric caveat below. |

**Metric caveat (critical, from `metric_summary.json`):** token-containment scores an
abstention as incorrect, so high-abstention RAG configs are penalised relative to
the always-answering plain-LLM. This is a measurement property, **not** evidence
that abstention is wrong — abstention can be the safe behaviour. Read correctness
alongside abstention rate; never read accuracy alone. At n=16 all paired McNemar
tests are **not** Holm-significant → **no superiority claim**.

---

## Component table — n30 ablation audit (6 configs × 30 Q, 2026-10-08 run)

This is the earlier n30 ablation (answer-quality + latency only). Retrieval-quality,
claim support (per-claim), hallucination, and calibration were not recorded per
question in these files.

| Component (config) | Correctness (ROUGE-L F1) | Retrieval | Claim support (citation support rate) | Hallucination | Abstention rate | Latency (avg ms/Q) | Memory | Uncertainty | Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| full-pipeline | 0.0888 (ST) | NM — not recorded in n30 files | 0.4333 (ST) | NM — no hallucination labels | 0.5667 (ST) | 140,687 (ST) | NM — VRAM not recorded in n30 | NM — per-Q confidence not recorded | Highest abstention; critic+calibration both ON. |
| no-critic | 0.1331 (ST) | NM | 0.6667 (ST) | NM | 0.3333 (ST) | 111,785 (ST) | NM | NM | Lower abstention vs full-pipeline (net Δ 0.2334); ROUGE-L higher but ROUGE-L ≠ correctness. |
| no-calibration | 0.1404 (ST) | NM | 0.6667 (ST) | NM | 0.3333 (ST) | 119,656 (ST) | NM | synthetic confidence when OFF | Calibration net abstention Δ vs full-pipeline = 0.0. |
| basic-rag | 0.1225 (ST) | NM | 0.5333 (ST) | NM | 0.4667 (ST) | 59,078 (ST) | NM | NM | Mid abstention; no critic/iterative/calibration. |
| no-iterative-retrieval | 0.0860 (ST) | NM | 0.4000 (ST) | NM | 0.6000 (ST) | 91,429 (ST) | NM | NM | Highest abstention of the n30 set. |
| plain-llm-no-rag | 0.1575 (ST) | N/A — no retrieval | 0.0000 (ST) | NM | 0.0000 (ST) | 8,907 (ST) | NM | N/A | Never abstains; no citations. |

**NOT MEASURED in n30 (reason):** per-question `critic_verdict`, `draft_answer`,
per-agent latency, retrieval R@K/MRR/nDCG, calibration ECE/Brier, and VRAM were
never saved by `IncrementalResultWriter` for this run. Per-question critic
accept/reject/rewrite and ROUGE-L draft-vs-final are therefore NOT MEASURED
(see `critic_analysis_20261017_063736.json`).

---

## Pipeline-level profiling (Phase 4c, SOFTWARE TEST on mock/fixture run)

- Profiled 5 mock questions, wall clock 25.231 s. On the mock run the dominant
  measured cost is async-loop / LangGraph node overhead (`ainvoke`, `model_dump`,
  `model_validate`), **not** LLM inference (mock LLM returns instantly).
- Real GPU latency is NOT in this profile; see n30 `latency_stats.avg_ms`
  (≈140,687 ms/Q for full-pipeline) and the Phase 5 `latency_breakdown.json`.
- Recommended optimisations (query cache, batch rerank, conditional gap-analysis,
  singleton graph, fewer serialisation round-trips) are **NOT MEASURED** —
  proposed, not benchmarked.

## Latency vs PRD targets (Phase 5 pilot, SOFTWARE TEST)

PRD targets: single-pass < 5 s; iterative ≤ 3 rounds < 15 s; retrieval < 500 ms;
rerank < 2 s. On this GPU, cold RAG means range ≈ 63–141 s/question — **far above**
the single-pass target (observed hardware/model cost, reported honestly, not a
pass). The query cache delivers the expected warm-path speedup: warm means for the
improved configs are 0.12 ms and 19.62 ms (cache hit). plain-llm cold ≈ 9.6 s.
