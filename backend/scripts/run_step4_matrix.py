"""Step 4 config-matrix runner.

Loads Qwen3-8B ONCE, builds the RAGPipeline ONCE from the eval-split index,
then runs all 6 ablation configs sequentially, writing one output JSON per config.

Usage
-----
    cd d:\\Ragmed_2\\backend
    $env:RAGMEDX_FAISS_INDEX_DIR = "data/faiss_index/eval_split"
    $env:RAGMEDX_LLM_MODE = "real"
    d:\\Ragmed_2\\.venv-cu128-pilot\\Scripts\\python.exe -m scripts.run_step4_matrix

Runtime estimate (printed at startup)
--------------------------------------
  full-pipeline            : 30 × 120s = 3600s  (~60 min)
  no-critic                : 30 ×  70s = 2100s  (~35 min)
  no-iterative-retrieval   : 30 ×  70s = 2100s  (~35 min)
  no-calibration           : 30 × 120s = 3600s  (~60 min)
  basic-rag                : 30 ×  50s = 1500s  (~25 min)
  plain-llm-no-rag         : 30 ×  15s =  450s  ( ~8 min)
  Total                    :                     (~223 min / ~3.7 h)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure backend/ is on sys.path so `app.*` and `evaluation.*` imports work ──
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import BASE_DIR, get_settings  # noqa: E402
from app.utils.logger import logger            # noqa: E402

# Re-use the constants and guard helper from the single-run benchmark script.
from scripts.run_step4_benchmark import (  # noqa: E402
    _CANONICAL_EVAL_SPLIT_DIR,
    _DEFAULT_DATASET_PATH,
    _resolve_and_guard_index_dir,
)

_DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "results"


# ---------------------------------------------------------------------------
# BioASQ-aware BenchmarkRunner (same adapter as in run_step4_benchmark.py)
# ---------------------------------------------------------------------------

def _make_bioasq_runner(dataset_path: Path, llm_wrapper):
    """Return a BioASQBenchmarkRunner with BioASQ JSON normalisation."""
    from evaluation.mirage_runner import BenchmarkRunner

    class BioASQBenchmarkRunner(BenchmarkRunner):
        """BenchmarkRunner subclass that loads BioASQ JSON format.

        Normalises BioASQ field names ('body', 'ideal_answer') to the
        'question' / 'answer' keys that run_rag_evaluation expects.
        """

        def load_dataset(self):
            if not self.dataset_path or not self.dataset_path.exists():
                return []
            with open(self.dataset_path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("questions", list(raw.values())[0])
            normalised = []
            for item in raw:
                answer = item.get("ideal_answer") or item.get("exact_answer") or ""
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

    return BioASQBenchmarkRunner(dataset_path=dataset_path, llm_wrapper=llm_wrapper)


# ---------------------------------------------------------------------------
# Main async entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    settings = get_settings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Print runtime estimate ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Step 4 config-matrix runner — runtime estimate")
    print("=" * 65)
    print("  full-pipeline          : 30 × 120s = 3600s  (~60 min)")
    print("  no-critic              : 30 ×  70s = 2100s  (~35 min)")
    print("  no-iterative-retrieval : 30 ×  70s = 2100s  (~35 min)")
    print("  no-calibration         : 30 × 120s = 3600s  (~60 min)")
    print("  basic-rag              : 30 ×  50s = 1500s  (~25 min)")
    print("  plain-llm-no-rag       : 30 ×  15s =  450s  ( ~8 min)")
    print("  ─────────────────────────────────────────────────────")
    print("  Total                  :              ~223 min (~3.7 h)")
    print("=" * 65 + "\n")

    # ── 1. Resolve and guard index dir ────────────────────────────────────
    eval_split_dir = _resolve_and_guard_index_dir(None)

    # ── 2. Load Qwen3-8B ONCE ─────────────────────────────────────────────
    from app.llm.local_llm import LLMRuntimeMode, LocalLLMWrapper, ModelLoadError

    logger.info(
        f"[Matrix] Eager-loading LLM '{settings.llm_model_name}' "
        f"(mode=LOCAL, allow_fallback=False) …"
    )
    llm_wrapper = LocalLLMWrapper(
        model_name_or_path=settings.llm_model_name,
        mode=LLMRuntimeMode.LOCAL,
        allow_fallback=False,
    )
    t_llm_start = time.perf_counter()
    llm_wrapper.load_model()  # raises ModelLoadError on failure — intentional
    model_load_ms = int((time.perf_counter() - t_llm_start) * 1000)

    # Hard assertion: must be LOCAL after load.
    assert llm_wrapper.active_runtime_mode == LLMRuntimeMode.LOCAL, (
        f"Expected LOCAL runtime mode after load_model(), got "
        f"'{llm_wrapper.active_runtime_mode.value}'.  Aborting."
    )
    logger.info(
        f"[Matrix] LLM loaded in {model_load_ms} ms — "
        f"active_runtime_mode={llm_wrapper.active_runtime_mode.value}"
    )

    # ── 3. Build RAGPipeline ONCE ─────────────────────────────────────────
    from app.rag.pipeline import RAGPipeline

    logger.info(f"[Matrix] Building RAGPipeline from: {eval_split_dir}")
    pipeline = RAGPipeline.build(
        index_dir=eval_split_dir,
        top_k_retrieval=10,
        top_k_final=3,
    )
    logger.info("[Matrix] RAGPipeline ready.")

    # ── 4. Load dataset and extract first 30 questions ────────────────────
    dataset_path = _DEFAULT_DATASET_PATH
    runner_for_load = _make_bioasq_runner(dataset_path, llm_wrapper)
    all_questions = runner_for_load.load_dataset()
    first_30_questions = all_questions[:30]
    logger.info(f"[Matrix] Dataset loaded — using first {len(first_30_questions)} questions.")

    # ── 5. Config definitions ─────────────────────────────────────────────
    # Each config is a tuple: (name, coroutine_factory)
    # The coroutine_factory is called inside the loop so we get a fresh coroutine each time.

    async def _run_rag(config_name: str, **kwargs):
        """Helper that creates a fresh BioASQBenchmarkRunner and runs run_rag_evaluation."""
        runner = _make_bioasq_runner(dataset_path, llm_wrapper)
        return await runner.run_rag_evaluation(
            max_samples=30,
            index_dir=eval_split_dir,
            generator_temperature=0.0,
            retrieval_provider=pipeline.retrieve_evidence,
            **kwargs,
        )

    from evaluation.plain_llm_runner import run_plain_llm_batch

    configs = [
        # name, coroutine (created lazily below)
        ("full-pipeline",            lambda: _run_rag("full-pipeline")),
        ("no-critic",                lambda: _run_rag("no-critic",              disable_critic=True)),
        ("no-iterative-retrieval",   lambda: _run_rag("no-iterative-retrieval", max_retrieval_iterations=1)),
        ("no-calibration",           lambda: _run_rag("no-calibration",         disable_calibration=True)),
        ("basic-rag",                lambda: _run_rag("basic-rag",              max_retrieval_iterations=1,
                                                                                disable_critic=True,
                                                                                disable_calibration=True)),
        ("plain-llm-no-rag",         None),  # handled separately — no retrieval
    ]

    _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 6. Run all configs sequentially ───────────────────────────────────
    for config_name, coro_factory in configs:
        print(f"\n{'─' * 60}")
        print(f"[Matrix] Starting config: {config_name}")
        t_cfg_start = time.perf_counter()

        try:
            if config_name == "plain-llm-no-rag":
                # ── Index guard not applicable for plain-LLM (no retrieval) ──
                print("[Matrix] plain-llm-no-rag: no index guard (no retrieval)")
                result = run_plain_llm_batch(
                    questions=first_30_questions,
                    llm_wrapper=llm_wrapper,
                    temperature=0.0,
                )
                index_dir_meta = "N/A"
                generator_temp_meta: float | str = "N/A"
            else:
                # ── Safety guard: verify we are using the canonical eval-split ──
                if eval_split_dir != _CANONICAL_EVAL_SPLIT_DIR:
                    raise RuntimeError(
                        f"[Matrix] SAFETY GUARD FAILED for config '{config_name}': "
                        f"index_dir ({eval_split_dir}) != _CANONICAL_EVAL_SPLIT_DIR "
                        f"({_CANONICAL_EVAL_SPLIT_DIR}).  Aborting to prevent "
                        "index contamination."
                    )
                result = await coro_factory()
                index_dir_meta = str(eval_split_dir)
                generator_temp_meta = 0.0

            elapsed = int(time.perf_counter() - t_cfg_start)

            # ── Extract rouge_l_f1 for progress line ──────────────────────
            metrics = result.get("metrics", {})
            rouge_l = metrics.get("rouge_l_f1", metrics.get("rouge_l", "N/A"))
            n_q = result.get("samples_evaluated", 0)
            print(
                f"CONFIG {config_name} DONE: {n_q} questions, "
                f"{elapsed}s, rouge_l_f1={rouge_l}"
            )

            # ── Write output JSON ─────────────────────────────────────────
            output_path = _DEFAULT_OUTPUT_DIR / f"step4_{config_name}_n30_{timestamp}.json"
            payload = {
                "run_metadata": {
                    "config_name":           config_name,
                    "model_load_ms":         model_load_ms,
                    "generator_temperature": generator_temp_meta,
                    "index_dir":             index_dir_meta,
                    "n_questions":           30,
                    "completed_at":          datetime.now(timezone.utc).isoformat(),
                    "llm_model":             settings.llm_model_name,
                    "llm_mode":              llm_wrapper.active_runtime_mode.value,
                },
                **result,
            }
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info(f"[Matrix] Output written to: {output_path}")

        except Exception:
            print(f"\n[Matrix] EXCEPTION in config '{config_name}':", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 65)
    print("Step 4 config-matrix run COMPLETE.")
    print(f"  Outputs written to: {_DEFAULT_OUTPUT_DIR}")
    print("=" * 65 + "\n")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
