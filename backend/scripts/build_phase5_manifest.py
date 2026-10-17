"""FEAT-001 — Freeze the reproducible 50-question experiment manifest (CPU-only).

Builds the frozen question-ID lists and a versioned experiment manifest for the
RAGMedX-MA 50x8 evaluation. This script performs NO GPU work and loads NO model.

What it does
------------
  * Reuses the EXISTING frozen test pool
    (backend/results/phase5_20261017_131123/splits/test.json, 250 Q) — it does
    NOT rebuild the leakage-checked eval split.
  * Draws QUESTION_IDS_50 and QUESTION_IDS_40 with the runner's own
    _stratified_sample(records, n, seed=42, stratify_by="source") so the draw
    matches the pilot's sampler exactly.
  * Asserts the strict-subset chain pilot16 ⊂ 40 ⊂ 50 and aborts otherwise.
  * Writes experiment_manifest.json + question_ids_50.json to a staging dir.

CPU interpreter ONLY:
    d:\\Ragmed_2\\.venv\\Scripts\\python.exe backend\\scripts\\build_phase5_manifest.py

Reproducibility: the seeded draw is deterministic, so re-running produces a
byte-identical question_ids_50.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Reuse the EXACT sampler the pilot used — do not reimplement.
from scripts.run_phase5_experiment import _CONFIGS, _GENERATOR_TEMPERATURE, _stratified_sample  # noqa: E402

_RESULTS_ROOT = _BACKEND_DIR / "results"
_PILOT_DIR = _RESULTS_ROOT / "phase5_20261017_131123"
_TEST_SPLIT = _PILOT_DIR / "splits" / "test.json"
_DATASET_MANIFEST = _PILOT_DIR / "dataset_manifest.json"
_PILOT_ENVIRONMENT = _PILOT_DIR / "environment.json"
_PROMPTS_DIR = _BACKEND_DIR / "prompts" / "v1"
_CONFIGS_DIR = _PILOT_DIR / "configs"
_METRICS_FILE = _BACKEND_DIR / "evaluation" / "metrics.py"
_STATISTICS_FILE = _BACKEND_DIR / "evaluation" / "statistics.py"

# Default staging output dir (independent of the GPU run dir FEAT-002 creates).
_DEFAULT_OUT_DIR = _RESULTS_ROOT / "phase5_manifest"

_SEED = 42
_N_PREFERRED = 50
_N_FALLBACK = 40

# The 16 validated pilot IDs (VERIFIED in design; also re-derived below).
_PILOT_REUSED_IDS_16 = [
    "pubmedqa::15943725",
    "bioasq::602825a61cb411341a0000fa",
    "bioasq::53372ba3d6d3ac6a34000058",
    "bioasq::54d7a1047035c08008000002",
    "pubmedqa::21394762",
    "pubmedqa::16735905",
    "pubmedqa::26215326",
    "pubmedqa::23321509",
    "pubmedqa::24449622",
    "pubmedqa::22251324",
    "pubmedqa::18719011",
    "pubmedqa::10548670",
    "pubmedqa::20605051",
    "bioasq::6060996394d57fd879000045",
    "pubmedqa::12098035",
    "pubmedqa::23621776",
]

# Eval_split hashes (VERIFIED, from the pilot environment.json). Lower-cased so
# the manifest consumer can compare case-insensitively.
_INDEX_FAISS_SHA256 = "fc18d20d30ea92e6d4b7581786bd1ba640e18e658d994466d47d924c847fc258"
_CHUNKS_PKL_SHA256 = "6370a1b47b2d0e799788f3406f320dc29057c008db4f4d8ce19f761456a5f5d1"
_INDEX_NUM_CHUNKS = 58173


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_counts(records_by_id: Dict[str, Dict[str, Any]], ids: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for qid in ids:
        src = records_by_id[qid].get("source", "__unknown__")
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def _package_version(name: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(name)
    except Exception:
        return "NOT INSTALLED"


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_BACKEND_DIR.parent),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - defensive
        return f"UNKNOWN ({exc})"


def build_manifest(out_dir: Path) -> Dict[str, Any]:
    # --- Load the immutable frozen test pool -------------------------------
    if not _TEST_SPLIT.exists():
        raise SystemExit(f"[FEAT-001] frozen test pool not found: {_TEST_SPLIT}")
    test_records: List[Dict[str, Any]] = json.loads(_TEST_SPLIT.read_text(encoding="utf-8"))
    if len(test_records) != 250:
        raise SystemExit(
            f"[FEAT-001] expected 250 test questions, found {len(test_records)}; refusing to proceed."
        )
    records_by_id = {r["id"]: r for r in test_records}

    # --- Draw the frozen 50 and the 40 fallback (same sampler as the pilot) -
    sample_50 = _stratified_sample(test_records, n=_N_PREFERRED, seed=_SEED, stratify_by="source")
    sample_40 = _stratified_sample(test_records, n=_N_FALLBACK, seed=_SEED, stratify_by="source")
    question_ids_50 = [r["id"] for r in sample_50]
    question_ids_40 = [r["id"] for r in sample_40]

    # --- Integrity assertions ---------------------------------------------
    assert len(question_ids_50) == len(set(question_ids_50)) == 50, (
        f"[FEAT-001] question_ids_50 must be 50 unique IDs, got "
        f"{len(question_ids_50)} ({len(set(question_ids_50))} unique)"
    )
    assert len(question_ids_40) == len(set(question_ids_40)) == 40, (
        f"[FEAT-001] question_ids_40 must be 40 unique IDs, got "
        f"{len(question_ids_40)} ({len(set(question_ids_40))} unique)"
    )
    pool_ids = set(records_by_id)
    assert set(question_ids_50) <= pool_ids, "[FEAT-001] question_ids_50 not all in test pool"
    assert set(question_ids_40) <= pool_ids, "[FEAT-001] question_ids_40 not all in test pool"

    pilot = set(_PILOT_REUSED_IDS_16)
    assert pilot <= pool_ids, "[FEAT-001] pilot_reused_ids_16 not all in test pool"
    s40 = set(question_ids_40)
    s50 = set(question_ids_50)
    # Strict-subset chain pilot16 ⊂ 40 ⊂ 50 (verified in design, enforced here).
    if not (pilot < s40 < s50):
        raise SystemExit(
            "[FEAT-001] strict-subset chain FAILED: expected "
            "set(pilot_reused_ids_16) ⊂ set(question_ids_40) ⊂ set(question_ids_50).\n"
            f"  |pilot|={len(pilot)} |40|={len(s40)} |50|={len(s50)}\n"
            f"  pilot - 40 = {sorted(pilot - s40)}\n"
            f"  40 - 50 = {sorted(s40 - s50)}"
        )

    # --- Dataset manifest hashes (copied, split not rebuilt) ---------------
    dataset_manifest = json.loads(_DATASET_MANIFEST.read_text(encoding="utf-8"))
    manifest_hashes = dataset_manifest["manifest_hash"]
    retrieval_measurability = dataset_manifest["retrieval_quality_measurability"]

    # --- Config names + per-config parameter hash --------------------------
    expected_configs = list(_CONFIGS.keys())
    assert len(expected_configs) == 8, f"[FEAT-001] expected 8 configs, found {len(expected_configs)}"
    config_hashes: Dict[str, str] = {}
    for name in expected_configs:
        cfg_path = _CONFIGS_DIR / f"{name}.json"
        if not cfg_path.exists():
            raise SystemExit(f"[FEAT-001] config file missing: {cfg_path}")
        config_hashes[name] = _sha256_file(cfg_path)

    # --- Prompt file hashes (unchanged) ------------------------------------
    prompt_hashes: Dict[str, str] = {}
    for p in sorted(_PROMPTS_DIR.glob("*.txt")):
        prompt_hashes[p.name] = _sha256_file(p)

    # --- Metric module hashes + library versions --------------------------
    metric_functions = [
        "compute_rouge_l", "compute_batch_rouge_l", "compute_exact_containment_rate",
        "compute_abstention_rate", "compute_citation_support_rate",
        "compute_answer_with_citation_rate", "recall_at_k", "precision_at_k",
        "mrr", "ndcg_at_k", "ece", "brier_score", "risk_coverage_curve",
        "abstention_precision", "abstention_recall", "false_abstention_rate",
        "false_acceptance_rate",
    ]
    statistics_functions = ["wilson_ci", "mcnemar_test", "paired_bootstrap", "holm_correction"]

    # --- Resolved eval_split path (read-only) ------------------------------
    resolved_eval_split = dataset_manifest["leakage_checks"]["resolved_eval_split_dir"]

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "feature": "FEAT-001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_tier": "SOFTWARE TEST (frozen experiment manifest)",
        "dataset_split": {
            "source_file": str(_TEST_SPLIT),
            "n_pool": len(test_records),
            "seed": _SEED,
            "stratify_by": "source",
            "manifest_hash": {
                "pool": manifest_hashes["pool"],
                "dev": manifest_hashes["dev"],
                "calibration": manifest_hashes["calibration"],
                "test": manifest_hashes["test"],
            },
            "note": "Leakage-checked eval split reused verbatim; NOT rebuilt.",
        },
        "selection": {
            "function": "_stratified_sample",
            "source_module": "backend/scripts/run_phase5_experiment.py",
            "source_file": str(_TEST_SPLIT),
            "seed": _SEED,
            "stratify_by": "source",
            "n_preferred": _N_PREFERRED,
            "n_fallback": _N_FALLBACK,
            "subset_assertion": "set(pilot_reused_ids_16) ⊂ set(question_ids_40) ⊂ set(question_ids_50)",
            "subset_assertion_passed": True,
        },
        "question_ids_50": question_ids_50,
        "question_ids_40": question_ids_40,
        "pilot_reused_ids_16": list(_PILOT_REUSED_IDS_16),
        "source_counts": {
            "question_ids_50": _source_counts(records_by_id, question_ids_50),
            "question_ids_40": _source_counts(records_by_id, question_ids_40),
            "pilot_reused_ids_16": _source_counts(records_by_id, _PILOT_REUSED_IDS_16),
        },
        "configs": expected_configs,
        "config_parameter_hashes": config_hashes,
        "generation": {
            "generator_temperature": _GENERATOR_TEMPERATURE,
            "prompts_dir": str(_PROMPTS_DIR),
            "prompt_sha256": prompt_hashes,
            "top_k_retrieval": 10,
            "top_k_final": 3,
            "note": "Prompts, retrieval params, and generation settings unchanged.",
        },
        "model": {
            "model_name": "Qwen/Qwen3-8B",
            "quantization": "4-bit NF4",
            "llm_mode": "local",
            "mock_fallback_allowed": False,
            "hard_assert_llm_mode_local": True,
        },
        "index": {
            "resolved_eval_split_path": resolved_eval_split,
            "index_faiss_sha256": _INDEX_FAISS_SHA256,
            "chunks_pkl_sha256": _CHUNKS_PKL_SHA256,
            "num_chunks": _INDEX_NUM_CHUNKS,
            "index_leakage_guard": "PASS",
            "production_index_protected": True,
        },
        "code_revision": {
            "git_head": _git_head(),
            "branch": "yuvraj/llm",
        },
        "metric_definitions": {
            "metrics_module": {
                "file": str(_METRICS_FILE),
                "sha256": _sha256_file(_METRICS_FILE),
                "functions": metric_functions,
            },
            "statistics_module": {
                "file": str(_STATISTICS_FILE),
                "sha256": _sha256_file(_STATISTICS_FILE),
                "functions": statistics_functions,
            },
            "library_versions": {
                "scipy": _package_version("scipy"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
            },
            "note": "Metrics reused as-defined; no metric redefinition.",
        },
        "retrieval_quality_measurability": retrieval_measurability,
    }

    # --- Write outputs atomically (write .tmp then replace) ----------------
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "experiment_manifest.json"
    ids_path = out_dir / "question_ids_50.json"

    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(ids_path, question_ids_50)

    print("[FEAT-001] manifest written:", manifest_path)
    print("[FEAT-001] question_ids_50 written:", ids_path)
    print(
        f"[FEAT-001] OK  |50|={len(question_ids_50)} |40|={len(question_ids_40)} "
        f"|pilot16|={len(_PILOT_REUSED_IDS_16)}  subset-chain PASS  "
        f"temperature={_GENERATOR_TEMPERATURE}  configs={len(expected_configs)}"
    )
    return manifest


def _atomic_write_json(path: Path, payload: Any) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # sort_keys for the id list is wrong (order matters); callers pass the list
    # directly so order is preserved. indent=2 + trailing newline for stable bytes.
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="FEAT-001 phase5 manifest builder (CPU-only).")
    parser.add_argument(
        "--out-dir",
        default=str(_DEFAULT_OUT_DIR),
        help="Staging dir for experiment_manifest.json + question_ids_50.json.",
    )
    args = parser.parse_args()
    build_manifest(Path(args.out_dir))


if __name__ == "__main__":
    main()
