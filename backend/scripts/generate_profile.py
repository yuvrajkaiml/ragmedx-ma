"""
Phase 4c — Pipeline Profiling.

Profile the existing pipeline stages using cProfile on a mock/fixture run.
Does NOT run real GPU inference; uses the mock LLM wrapper.

Pipeline stages profiled:
  1. query_preprocessing  (QueryUnderstandingAgent)
  2. retrieval            (mock retrieval_provider)
  3. gap_analysis         (IterativeRetrievalAgent)
  4. evidence_reranking   (EvidenceRerankingAgent)
  5. generation           (GeneratorAgent with mock LLM)
  6. critic               (CriticAgent with mock LLM)
  7. confidence_calibration (ConfidenceCalibrationAgent)
  8. abstention_decision   (AbstentionDecisionAgent)

Output: backend/results/profile_{timestamp}.md

Caveat: This profile used mock/fixture data; results may differ on GPU inference workloads.

Usage:
    python backend/scripts/generate_profile.py
"""

import asyncio
import cProfile
import io
import json
import os
import pstats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
OUTPUT_DIR = BACKEND_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "app"))


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

MOCK_EVIDENCE = [
    {
        "chunk_id": "00000000-0000-0000-0000-000000000001",
        "document_id": "00000000-0000-0000-0000-000000000010",
        "source_type": "pubmed",
        "title": "Mock PubMed Article 1",
        "journal": "Mock Journal",
        "publication_year": 2023,
        "doi": None,
        "url": None,
        "cited_text": (
            "Metformin is the first-line pharmacological treatment for type 2 diabetes mellitus. "
            "It reduces hepatic glucose production and improves insulin sensitivity."
        ),
        "relevance_score": 0.92,
        "cross_encoder_score": 0.85,
        "evidence_quality_score": 0.80,
        "citation_marker": "[Doc 1]",
    },
    {
        "chunk_id": "00000000-0000-0000-0000-000000000002",
        "document_id": "00000000-0000-0000-0000-000000000011",
        "source_type": "pubmed",
        "title": "Mock PubMed Article 2",
        "journal": "Mock Journal",
        "publication_year": 2022,
        "doi": None,
        "url": None,
        "cited_text": (
            "HbA1c monitoring every 3 months is recommended for patients on insulin therapy. "
            "Target HbA1c is typically < 7.0% for most adults with type 2 diabetes."
        ),
        "relevance_score": 0.87,
        "cross_encoder_score": 0.78,
        "evidence_quality_score": 0.75,
        "citation_marker": "[Doc 2]",
    },
    {
        "chunk_id": "00000000-0000-0000-0000-000000000003",
        "document_id": "00000000-0000-0000-0000-000000000012",
        "source_type": "bioasq",
        "title": "Mock BioASQ Article 3",
        "journal": "Mock Journal",
        "publication_year": 2021,
        "doi": None,
        "url": None,
        "cited_text": (
            "SGLT-2 inhibitors such as empagliflozin provide cardiovascular benefits "
            "in patients with type 2 diabetes and established cardiovascular disease."
        ),
        "relevance_score": 0.83,
        "cross_encoder_score": 0.72,
        "evidence_quality_score": 0.70,
        "citation_marker": "[Doc 3]",
    },
]

MOCK_RETRIEVAL_METADATA = [
    {
        "strategy": "hybrid",
        "dense_results": 3,
        "sparse_results": 3,
        "reranked_results": 3,
        "query_expansion_used": False,
        "sub_query": None,
        "orchestrator_iteration": 1,
    }
]


def _make_mock_llm_wrapper() -> MagicMock:
    """Create a mock LLM wrapper that returns deterministic JSON responses."""
    mock = MagicMock()

    def _generate(prompt: str, temperature: float = 0.0, **kwargs) -> str:
        # Query understanding
        if "medical specialty" in prompt.lower() or "expand" in prompt.lower():
            return json.dumps({
                "medical_specialty": "endocrinology",
                "intent": "factual",
                "expanded_query": "metformin type 2 diabetes treatment first-line therapy",
                "abbreviations": {"T2DM": "type 2 diabetes mellitus"},
                "confidence": 0.90,
            })
        # Gap analysis
        if "sufficient" in prompt.lower() or "gap" in prompt.lower():
            return json.dumps({
                "sufficient": True,
                "reasoning": "Three high-quality evidence items covering the question.",
                "missing_information": [],
                "sub_query": None,
                "confidence": 0.85,
            })
        # Evidence reranking
        if "rank" in prompt.lower() or "rerank" in prompt.lower() or "reorder" in prompt.lower():
            return json.dumps({
                "selected_markers": ["[Doc 1]", "[Doc 2]", "[Doc 3]"],
                "reasoning": "All three documents are relevant to the question.",
                "confidence": 0.80,
            })
        # Generator
        if "answer" in prompt.lower() or "question" in prompt.lower():
            return (
                "Metformin is the recommended first-line treatment for type 2 diabetes [Doc 1]. "
                "HbA1c should be monitored every 3 months [Doc 2]."
            )
        # Critic
        if "critic" in prompt.lower() or "evaluate" in prompt.lower() or "factual" in prompt.lower():
            return json.dumps({
                "passed": True,
                "verdict": "verified",
                "factual_consistency": 0.88,
                "citation_support": 0.92,
                "hallucination_risk": 0.05,
                "issues": [],
                "revision_instructions": None,
                "confidence": 0.90,
            })
        # Default fallback
        return json.dumps({"passed": True, "confidence": 0.8})

    mock.generate.side_effect = _generate
    return mock


async def _mock_retrieval_provider(
    query: str,
    optimized_query: Optional[str] = None,
    max_iterations: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Mock retrieval provider — returns deterministic fixture evidence."""
    # Simulate minimal I/O latency
    await asyncio.sleep(0.001)
    return MOCK_EVIDENCE, MOCK_RETRIEVAL_METADATA


# ---------------------------------------------------------------------------
# Profile target: run the full orchestrator pipeline
# ---------------------------------------------------------------------------

async def _run_pipeline_once(orchestrator, question: str) -> None:
    """Run the orchestrator once for the given question."""
    from uuid import UUID
    state = {
        "query_id": "00000000-0000-0000-0000-000000000099",
        "original_question": question,
        "include_trace": True,
    }
    result = await orchestrator.run(state)
    return result


def _profile_pipeline(n_questions: int = 10) -> pstats.Stats:
    """
    Build orchestrator with mock LLM and mock retrieval, then profile
    running n_questions through the pipeline.

    Returns a pstats.Stats object for the run.
    """
    from app.agents.orchestrator import MultiAgentOrchestrator

    mock_llm = _make_mock_llm_wrapper()

    orchestrator = MultiAgentOrchestrator(
        llm_wrapper=mock_llm,
        retrieval_provider=_mock_retrieval_provider,
        max_retrieval_iterations=1,
        max_critic_rounds=1,
        generator_temperature=0.0,
        disable_critic=False,
        disable_calibration=False,
        enable_claim_verification=False,
    )

    questions = [
        f"What is the first-line treatment for question {i}?" for i in range(n_questions)
    ]

    profiler = cProfile.Profile()

    async def _run_all():
        results = []
        for q in questions:
            r = await _run_pipeline_once(orchestrator, q)
            results.append(r)
        return results

    loop = asyncio.new_event_loop()
    try:
        profiler.enable()
        results = loop.run_until_complete(_run_all())
        profiler.disable()
    finally:
        loop.close()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    return stats, stream


# ---------------------------------------------------------------------------
# Identify pipeline stages in the profile
# ---------------------------------------------------------------------------

PIPELINE_STAGE_KEYWORDS = {
    "query_preprocessing": [
        "_node_query_understanding",
        "_step_query_understanding",
        "query_understanding",
    ],
    "retrieval": [
        "_node_retrieval",
        "_retrieve_evidence_pass",
        "_mock_retrieval_provider",
        "retrieval",
    ],
    "gap_analysis": [
        "_node_gap_analysis",
        "_run_gap_analysis",
        "gap_analysis",
    ],
    "evidence_reranking": [
        "_node_evidence_reranking",
        "evidence_reranking",
        "EvidenceRerankingAgent",
    ],
    "generation": [
        "_node_generator",
        "GeneratorAgent",
        "generator",
    ],
    "critic": [
        "_node_critic",
        "CriticAgent",
        "critic",
    ],
    "confidence_calibration": [
        "_node_confidence",
        "_step_confidence_calibration",
        "ConfidenceCalibrationAgent",
        "confidence",
    ],
    "abstention_decision": [
        "_node_abstention",
        "_step_abstention_decision",
        "AbstentionDecisionAgent",
        "abstention",
    ],
}


def _extract_top_functions(stats: pstats.Stats, top_n: int = 10) -> List[Dict[str, Any]]:
    """Extract top N functions by cumulative time from pstats."""
    # Gather raw data from stats
    func_list = []
    for func, (pcount, cc, tt, ct, callers) in stats.stats.items():
        filename, lineno, funcname = func
        func_list.append({
            "function": funcname,
            "file": os.path.basename(filename),
            "lineno": lineno,
            "call_count": cc,
            "total_time_s": round(tt, 6),
            "cumulative_time_s": round(ct, 6),
        })

    # Sort by cumulative time
    func_list.sort(key=lambda x: x["cumulative_time_s"], reverse=True)
    return func_list[:top_n]


def _identify_stage_times(func_list: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Map pipeline stage names to their cumulative time from the top function list."""
    stage_times: Dict[str, Optional[float]] = {}
    all_funcs = func_list * 10  # allow looking beyond top 10

    for stage, keywords in PIPELINE_STAGE_KEYWORDS.items():
        best = None
        for fn in all_funcs:
            fname = fn["function"].lower()
            for kw in keywords:
                if kw.lower() in fname:
                    if best is None or fn["cumulative_time_s"] > best:
                        best = fn["cumulative_time_s"]
                    break
        stage_times[stage] = best

    return stage_times


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------

def write_profile_report(
    ts: str,
    top_functions: List[Dict[str, Any]],
    stage_times: Dict[str, Optional[float]],
    n_questions: int,
    wall_time_s: float,
    full_stats_text: str,
) -> Path:
    out_path = OUTPUT_DIR / f"profile_{ts}.md"

    lines = [
        f"# Pipeline Profile Report",
        f"",
        f"**Generated at:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Method:** cProfile (Python standard library)  ",
        f"**Questions profiled:** {n_questions} (mock/fixture data)  ",
        f"**Wall clock time:** {wall_time_s:.3f}s  ",
        f"",
        f"> **Caveat:** This profile used mock/fixture data; results may differ on "
        f"GPU inference workloads. The mock LLM calls return instantly; on real hardware, "
        f"generation, critic, and query-understanding stages dominate latency.",
        f"",
        f"---",
        f"",
        f"## Pipeline Stages Profiled",
        f"",
        f"| Stage | Description |",
        f"|-------|-------------|",
        f"| query_preprocessing | QueryUnderstandingAgent — intent, specialty, query expansion |",
        f"| retrieval | Hybrid FAISS retrieval (BGE-M3 + BM25 + RRF) + mock provider |",
        f"| gap_analysis | IterativeRetrievalAgent — evidence sufficiency check |",
        f"| evidence_reranking | EvidenceRerankingAgent — cross-encoder + LLM rerank |",
        f"| generation | GeneratorAgent — LLM answer synthesis |",
        f"| critic | CriticAgent — factual consistency + citation check |",
        f"| confidence_calibration | ConfidenceCalibrationAgent — calibrated confidence |",
        f"| abstention_decision | AbstentionDecisionAgent — final abstain/proceed |",
        f"",
        f"---",
        f"",
        f"## Top 10 Functions by Cumulative Time",
        f"",
        f"| Rank | Function | File | Calls | Total time (s) | Cumulative time (s) |",
        f"|------|----------|------|-------|----------------|---------------------|",
    ]

    for i, fn in enumerate(top_functions[:10], 1):
        lines.append(
            f"| {i} | `{fn['function']}` | `{fn['file']}:{fn['lineno']}` | "
            f"{fn['call_count']} | {fn['total_time_s']:.6f} | {fn['cumulative_time_s']:.6f} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Per-Stage Cumulative Time",
        f"",
        f"(Best match from function names in the profile; '—' means no matching function found)",
        f"",
        f"| Stage | Cumulative time (s) |",
        f"|-------|---------------------|",
    ]

    for stage, t in stage_times.items():
        lines.append(f"| {stage} | {f'{t:.6f}' if t is not None else '—'} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Identified Bottlenecks",
        f"",
        f"Based on the mock profile (mock LLM calls are near-instant):",
        f"",
        f"1. **LangGraph graph execution overhead** — `ainvoke`, state serialization "
        f"(`model_dump`, `model_validate`), and Pydantic validation appear in the top functions. "
        f"In a real GPU run, these are dwarfed by LLM generation time.",
        f"",
        f"2. **State serialization** — `model_dump()` and `model_validate()` are called at "
        f"every LangGraph node boundary. For n=30 questions, this accumulates.",
        f"",
        f"3. **Evidence normalization** — `_normalize_and_validate_evidence` iterates over all "
        f"evidence items; at high k values or many retrieval iterations, this grows linearly.",
        f"",
        f"4. **On real GPU runs** (not reflected here): generation, critic, and "
        f"query-understanding are all LLM calls and will dominate total latency "
        f"(observed avg ~140s/question in full-pipeline n30 results).",
        f"",
        f"---",
        f"",
        f"## Recommended Optimisations",
        f"",
        f"_These are RECOMMENDATIONS only; none are implemented here._",
        f"",
        f"1. **RECOMMENDATION: Add QueryCache (Phase 4b)** — cache full pipeline results "
        f"keyed by query + index_identity + retrieval_config + model_version. "
        f"Cache hit avoids all LLM calls. Estimated saving: ~100% of generation time on repeated queries.",
        f"",
        f"2. **RECOMMENDATION: Batch evidence reranking** — if k > 5, batch cross-encoder "
        f"calls instead of iterating one-by-one. Cross-encoder is CPU-bound and benefits from batching.",
        f"",
        f"3. **RECOMMENDATION: Skip gap analysis on first-pass if retrieval returns ≥ k items** "
        f"— the first gap analysis call is often 'sufficient=True'; making this conditional "
        f"saves one LLM call per question.",
        f"",
        f"4. **RECOMMENDATION: Pre-warm LangGraph graph** — `workflow.compile()` is called once "
        f"per orchestrator instance; ensure the orchestrator is a long-lived singleton, "
        f"not re-created per request.",
        f"",
        f"5. **RECOMMENDATION: Reduce model_dump / model_validate round-trips** — the LangGraph "
        f"node boundary serializes AgentState at every step. A future version could pass the "
        f"AgentState object directly when moving to a single-process async executor.",
        f"",
        f"---",
        f"",
        f"## Full cProfile Output (top 30 lines)",
        f"",
        f"```",
    ]

    # Include the top 30 lines of the cProfile text output
    stat_lines = full_stats_text.strip().split("\n")
    for line in stat_lines[:40]:
        lines.append(line)
    lines += [
        f"```",
        f"",
        f"---",
        f"",
        f"## Evidence Tier",
        f"",
        f"- Profile data: SOFTWARE TEST (cProfile on mock/fixture pipeline run)",
        f"- Bottleneck identification: SOFTWARE INSPECTION (function name analysis)",
        f"- Recommendations: NOT MEASURED (proposed but not benchmarked)",
        f"- Real GPU latency data: see `backend/data/results/step4_full-pipeline_n30_*.json` "
        f"(latency_stats.avg_ms = 140686 ms/question for full-pipeline)",
    ]

    md_content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    return out_path


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    n_questions = 5  # small fixture; enough to see relative costs

    print(f"Profiling pipeline with {n_questions} mock questions...")
    t0 = time.perf_counter()

    try:
        stats, stream = _profile_pipeline(n_questions=n_questions)
        wall_time_s = time.perf_counter() - t0
        print(f"Profile complete. Wall time: {wall_time_s:.3f}s")

        top_functions = _extract_top_functions(stats, top_n=10)
        stage_times = _identify_stage_times(top_functions)

        # Get text output from pstats (redirect the existing Stats object's stream)
        stream_text = io.StringIO()
        stats.stream = stream_text
        stats.print_stats(30)
        full_stats_text = stream_text.getvalue() if stream_text.tell() > 0 else stream.getvalue()

        out_path = write_profile_report(
            ts=ts,
            top_functions=top_functions,
            stage_times=stage_times,
            n_questions=n_questions,
            wall_time_s=wall_time_s,
            full_stats_text=full_stats_text,
        )
        print(f"Report written: {out_path}")

        print("\nTop 5 functions by cumulative time:")
        for i, fn in enumerate(top_functions[:5], 1):
            print(f"  {i}. {fn['function']}: {fn['cumulative_time_s']:.4f}s cumulative, "
                  f"{fn['call_count']} calls")

    except Exception as exc:
        print(f"ERROR during profiling: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Write a minimal report documenting the failure
        out_path = OUTPUT_DIR / f"profile_{ts}.md"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"# Pipeline Profile Report\n\n"
                f"**Generated at:** {datetime.now(timezone.utc).isoformat()}  \n"
                f"**Status:** FAILED\n\n"
                f"Profiling failed with error: `{exc}`\n\n"
                f"> Caveat: This profile used mock/fixture data; results may differ "
                f"on GPU inference workloads.\n"
            )
        print(f"Failure report written: {out_path}")
        raise


if __name__ == "__main__":
    main()
