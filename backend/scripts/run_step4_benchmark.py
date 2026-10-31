"""Step 4 BioASQ benchmark launcher.

Drives ``BenchmarkRunner.run_rag_evaluation()`` against the BioASQ eval split,
with an explicit eval-split index guard baked in at the top level.

Usage
-----
Normal run (full eval, all 250 questions)::

    cd backend
    $env:RAGMEDX_FAISS_INDEX_DIR = "data/faiss_index/eval_split"
    ..\\.venv\\Scripts\\python.exe -m scripts.run_step4_benchmark

Smoke test (1 question, prints resolved index path + first retrieved PMID)::

    $env:RAGMEDX_FAISS_INDEX_DIR = "data/faiss_index/eval_split"
    ..\\.venv\\Scripts\\python.exe -m scripts.run_step4_benchmark --smoke-test

Explicit path (alternative to env var)::

    ..\\.venv\\Scripts\\python.exe -m scripts.run_step4_benchmark \\
        --index-dir data/faiss_index/eval_split

Guard behaviour
---------------
The script resolves ``settings.faiss_index_dir`` (or ``--index-dir``) and
immediately raises if the path points to the production index
(``data/faiss_index/``) rather than the eval_split subdirectory.  This mirrors
the fail-loud pattern in ``RetrievalService.build()`` and prevents silently
producing contaminated accuracy numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure backend/ is on sys.path so `app.*` and `evaluation.*` imports work ──
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import BASE_DIR, get_settings  # noqa: E402 (after path fix)
from app.utils.logger import logger             # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CANONICAL_EVAL_SPLIT_DIR = (BASE_DIR / "data" / "faiss_index" / "eval_split").resolve()
_PRODUCTION_INDEX_DIR     = (BASE_DIR / "data" / "faiss_index").resolve()
_DEFAULT_DATASET_PATH     = BASE_DIR / "data" / "datasets" / "bioasq_eval_split" / "eval_questions.json"
_DEFAULT_OUTPUT_DIR       = BASE_DIR / "data" / "results"


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def _resolve_and_guard_index_dir(raw: str | Path | None) -> Path:
    """Resolve the index directory and raise loudly if it's the production path.

    Resolution order (highest priority first):
      1. --index-dir CLI argument (``raw`` parameter).
      2. RAGMEDX_FAISS_INDEX_DIR env var → settings.faiss_index_dir.
      3. Hard default: _CANONICAL_EVAL_SPLIT_DIR.

    Raises ``SystemExit`` (not RuntimeError) here at the launcher level so the
    error message is printed cleanly without a Python traceback.  The equivalent
    guard inside ``BenchmarkRunner.run_rag_evaluation()`` raises ``RuntimeError``
    as a second line of defence.
    """
    if raw is not None:
        resolved = Path(raw)
        if not resolved.is_absolute():
            resolved = BASE_DIR / resolved
        resolved = resolved.resolve()
        source = "--index-dir argument"
    else:
        settings = get_settings()
        candidate = settings.faiss_index_dir
        if not candidate.is_absolute():
            candidate = BASE_DIR / candidate
        candidate = candidate.resolve()

        if candidate == _PRODUCTION_INDEX_DIR:
            logger.warning(
                f"RAGMEDX_FAISS_INDEX_DIR resolves to the production index "
                f"({candidate}).  Defaulting to the eval_split index "
                f"({_CANONICAL_EVAL_SPLIT_DIR}) to prevent leakage contamination."
            )
            resolved = _CANONICAL_EVAL_SPLIT_DIR
            source = "hardcoded eval_split default (production path was set)"
        else:
            resolved = candidate
            source = "RAGMEDX_FAISS_INDEX_DIR / settings.faiss_index_dir"

    logger.info(
        f"[Step 4 guard] Resolved FAISS index dir : {resolved}\n"
        f"               Source                   : {source}\n"
        f"               Canonical eval_split dir  : {_CANONICAL_EVAL_SPLIT_DIR}\n"
        f"               Paths match              : {resolved == _CANONICAL_EVAL_SPLIT_DIR}"
    )

    # Hard fail if the resolved path is the production index.
    if resolved == _PRODUCTION_INDEX_DIR:
        print(
            "\n"
            "ERROR: Step 4 benchmark index_dir resolves to the PRODUCTION FAISS index:\n"
            f"  {resolved}\n\n"
            "The production index contains 1,916 of the 2,394 eval-question PMIDs.\n"
            "Running the benchmark against it would contaminate every accuracy metric.\n\n"
            "Fix: set the eval_split index explicitly before running:\n"
            "  $env:RAGMEDX_FAISS_INDEX_DIR = 'data/faiss_index/eval_split'\n"
            "  or pass: --index-dir data/faiss_index/eval_split\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify the index files actually exist at the resolved path.
    index_faiss = resolved / "index.faiss"
    chunks_pkl  = resolved / "chunks.pkl"
    if not index_faiss.exists() or not chunks_pkl.exists():
        missing = index_faiss if not index_faiss.exists() else chunks_pkl
        print(
            "\n"
            f"ERROR: FAISS index file not found: {missing}\n"
            f"       index_dir = {resolved}\n\n"
            "Build the eval-split index first:\n"
            "  python -m scripts.build_eval_split\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return resolved


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

async def _smoke_test(index_dir: Path, dataset_path: Path) -> None:
    """Run one question through the full pipeline and print the first chunk PMID.

    This is the canary check: confirms that the RetrievalService at runtime is
    actually loading from ``index_dir``, and that the first retrieved chunk's
    PMID is NOT in the 2,394 eval-question PMIDs (leakage = immediate abort).
    """
    import json as _json
    import pickle

    from app.agents import run_agent_pipeline
    from app.llm.local_llm import LocalLLMWrapper
    from app.rag.pipeline import RAGPipeline

    logger.info("[Smoke test] Loading one question from eval dataset …")

    # Load the first question from the eval set.
    with open(dataset_path, encoding="utf-8") as f:
        eval_qs = _json.load(f)
    if isinstance(eval_qs, dict):
        eval_qs = eval_qs.get("questions", list(eval_qs.values())[0])
    question_obj = eval_qs[0]
    question_text = question_obj.get("body", question_obj.get("question", ""))
    question_pmids = {
        _extract_pmid(d) for d in question_obj.get("documents", [])
    }

    logger.info(f"[Smoke test] Question : {question_text[:120]}")
    logger.info(f"[Smoke test] Eval-question PMIDs ({len(question_pmids)}): "
                f"{sorted(question_pmids)[:5]} …")

    # Build the pipeline against the guarded index_dir.
    logger.info(f"[Smoke test] Building RAGPipeline from: {index_dir}")
    pipeline = RAGPipeline.build(
        index_dir=index_dir,
        top_k_retrieval=10,
        top_k_final=3,
    )

    # Retrieve evidence for the question.
    logger.info("[Smoke test] Retrieving evidence …")
    raw_evidence, _ = await pipeline.retrieve_evidence(question_text)

    if not raw_evidence:
        print("\nSMOKE TEST FAILED: no evidence chunks returned.", file=sys.stderr)
        sys.exit(1)

    first_chunk = raw_evidence[0]
    # Evidence chunks carry metadata; extract PMID from whatever field is present.
    first_pmid = (
        first_chunk.get("pmid")
        or _extract_pmid(first_chunk.get("source", ""))
        or _extract_pmid(first_chunk.get("url", ""))
        or "UNKNOWN"
    )

    print("\n" + "=" * 60)
    print("SMOKE TEST — index path and canary PMID check")
    print("=" * 60)
    print(f"  Resolved index_dir : {index_dir}")
    print(f"  Canonical eval_split dir: {_CANONICAL_EVAL_SPLIT_DIR}")
    print(f"  Paths match        : {index_dir == _CANONICAL_EVAL_SPLIT_DIR}")
    print()
    print(f"  Question           : {question_text[:100]}")
    print(f"  Evidence chunks returned: {len(raw_evidence)}")
    print(f"  First chunk PMID   : {first_pmid}")
    print(f"  First chunk title  : {first_chunk.get('title', '(no title)')[:80]}")
    print(f"  First chunk source : {first_chunk.get('source', '(no source)')}")
    print()

    # Leakage canary: the first retrieved PMID must NOT be one of the eval-question PMIDs.
    if first_pmid in question_pmids:
        print(
            f"  CANARY RESULT : FAIL — PMID {first_pmid} IS in the eval-question PMID set!\n"
            "  This indicates index contamination.  STOP — do not proceed with the full benchmark.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(
            f"  CANARY RESULT : PASS — PMID {first_pmid} is NOT in the {len(question_pmids)}-PMID "
            "eval-question set for this question."
        )
        print("  Safe to proceed with the full benchmark run.")
    print("=" * 60 + "\n")


def _extract_pmid(url_or_str: str) -> str:
    """Return the bare PMID integer string from a PubMed URL or bare string."""
    import re
    s = str(url_or_str)
    m = re.search(r"(\d+)$", s)
    return m.group(1) if m else s


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

async def _full_run(
    index_dir: Path,
    dataset_path: Path,
    output_path: Path,
    max_samples: int | None,
) -> None:
    from app.llm.local_llm import LocalLLMWrapper, LLMRuntimeMode, ModelLoadError
    from evaluation.mirage_runner import BenchmarkRunner

    logger.info(f"[Step 4] Starting full BioASQ benchmark run")
    logger.info(f"  index_dir    : {index_dir}")
    logger.info(f"  dataset_path : {dataset_path}")
    logger.info(f"  max_samples  : {max_samples or 'all'}")
    logger.info(f"  output_path  : {output_path}")

    # ── BioASQ format adapter ─────────────────────────────────────────────
    class BioASQBenchmarkRunner(BenchmarkRunner):
        """BenchmarkRunner subclass that loads BioASQ JSON format.

        BioASQ eval_questions.json is a JSON array (not JSONL) with BioASQ
        field names: 'body' (question text), 'ideal_answer' / 'exact_answer'
        (reference answer).  This subclass normalises those to the 'question'
        and 'answer' keys that run_rag_evaluation's evaluation loop expects,
        without modifying BenchmarkRunner or any other caller.
        """

        def load_dataset(self):
            if not self.dataset_path or not self.dataset_path.exists():
                return []
            with open(self.dataset_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Unwrap dict wrapper if present (e.g. {"questions": [...]})
            if isinstance(raw, dict):
                raw = raw.get("questions", list(raw.values())[0])
            normalised = []
            for item in raw:
                # 'body' → 'question'; prefer ideal_answer, fall back to exact_answer
                answer = item.get("ideal_answer") or item.get("exact_answer") or ""
                # ideal_answer is sometimes a list of strings; join if so
                if isinstance(answer, list):
                    answer = " ".join(str(a) for a in answer)
                normalised.append({
                    "question": item.get("body", item.get("question", "")),
                    "answer":   answer,
                    "_id":      item.get("id", ""),
                    "_type":    item.get("type", ""),
                })
            logger.info(
                f"[BioASQBenchmarkRunner] Loaded {len(normalised)} questions "
                f"from {self.dataset_path.name}"
            )
            return normalised

    # ── Eager LLM load — fail loud, not silent ────────────────────────────
    # Build the wrapper in LOCAL mode (no mock fallback) and call load_model()
    # explicitly here, before any question loop starts.  If Qwen fails to load
    # for any reason (CUDA missing, VRAM, bitsandbytes error, etc.) this raises
    # ModelLoadError immediately and the run never starts — we never silently
    # produce mock answers labelled as a real benchmark run.
    settings = get_settings()
    logger.info(
        f"[Step 4] Eager-loading LLM '{settings.llm_model_name}' "
        f"(llm_mode={settings.llm_mode}, allow_fallback=False) …"
    )
    llm_wrapper = LocalLLMWrapper(
        model_name_or_path=settings.llm_model_name,
        mode=LLMRuntimeMode.LOCAL,   # strict LOCAL — raises on failure, no mock fallback
        allow_fallback=False,
    )
    _t_llm_start = time.perf_counter()
    llm_wrapper.load_model()   # raises ModelLoadError on any failure — intentional
    _model_load_ms = int((time.perf_counter() - _t_llm_start) * 1000)

    # Hard assertion: confirm LOCAL mode is active after load.
    # With allow_fallback=False and mode=LOCAL the silent-mock path is closed,
    # but we assert explicitly so a future change can't reopen it quietly.
    if llm_wrapper.active_runtime_mode != LLMRuntimeMode.LOCAL:
        raise RuntimeError(
            f"LocalLLMWrapper.load_model() succeeded but active_runtime_mode is "
            f"'{llm_wrapper.active_runtime_mode.value}', not LOCAL.  "
            "This indicates an unexpected fallback.  Aborting rather than running "
            "the benchmark in the wrong mode."
        )
    logger.info(
        f"[Step 4] LLM loaded — active_runtime_mode={llm_wrapper.active_runtime_mode.value}  "
        f"is_loaded={llm_wrapper.is_loaded}  model={settings.llm_model_name}"
    )
    # Benchmark always uses temperature=0.0 for the generator so that
    # identical questions produce identical answers across runs.
    # The GeneratorAgent default (0.1) is intentionally preserved for the
    # live API — this override is benchmark-path-only.
    _BENCHMARK_GENERATOR_TEMPERATURE = 0.0
    logger.info(
        f"[Step 4] generator_temperature={_BENCHMARK_GENERATOR_TEMPERATURE} "
        "(deterministic mode — benchmark override of GeneratorAgent default 0.1)"
    )

    runner = BioASQBenchmarkRunner(
        dataset_path=dataset_path,
        llm_wrapper=llm_wrapper,
    )

    result = await runner.run_rag_evaluation(
        max_samples=max_samples,
        index_dir=index_dir,
        generator_temperature=_BENCHMARK_GENERATOR_TEMPERATURE,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_metadata": {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "index_dir": str(index_dir),
            "canonical_eval_split_dir": str(_CANONICAL_EVAL_SPLIT_DIR),
            "index_dir_is_eval_split": index_dir == _CANONICAL_EVAL_SPLIT_DIR,
            "dataset_path": str(dataset_path),
            "max_samples": max_samples,
            "llm_model": settings.llm_model_name,
            "llm_mode": llm_wrapper.active_runtime_mode.value,
            "generator_temperature": _BENCHMARK_GENERATOR_TEMPERATURE,
            "model_load_ms": _model_load_ms,
        },
        **result,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nStep 4 benchmark complete.")
    print(f"  Status           : {result.get('status')}")
    print(f"  LLM mode         : {llm_wrapper.active_runtime_mode.value}")
    print(f"  Samples evaluated: {result.get('samples_evaluated')}")
    print(f"  Metrics          : {json.dumps(result.get('metrics', {}), indent=4)}")
    print(f"  Output written to: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4 BioASQ benchmark launcher with eval-split index guard."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help=(
            "FAISS index directory to use. "
            "Defaults to RAGMEDX_FAISS_INDEX_DIR env var, "
            "or data/faiss_index/eval_split if that is unset or points at production."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET_PATH,
        help="Path to eval_questions.json (default: data/datasets/bioasq_eval_split/eval_questions.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR / f"step4_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON path for benchmark results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N questions (omit for all 250).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one question through the full pipeline, print the resolved index path "
            "and the first retrieved chunk PMID, and confirm it is not an eval-question PMID. "
            "Does not write a results file."
        ),
    )
    args = parser.parse_args()

    # ── Step 1: resolve and guard the index path ──────────────────────────
    index_dir = _resolve_and_guard_index_dir(args.index_dir)

    if args.smoke_test:
        asyncio.run(_smoke_test(index_dir=index_dir, dataset_path=args.dataset))
    else:
        asyncio.run(_full_run(
            index_dir=index_dir,
            dataset_path=args.dataset,
            output_path=args.output,
            max_samples=args.max_samples,
        ))


if __name__ == "__main__":
    main()
