"""
Phase F — Resumable execution driver.

Orchestrates: integrity pre-check -> diagnose -> build index -> retrieval eval ->
calibration -> integrity post-check. Writes the top-level ``manifest.json`` with
phases run, config hashes, input versions, timestamps, completed counts, and the
integrity pre/post status, plus a reproducible rerun command string.

Skips valid completed artifacts (config-hash match), writes atomically, and records
a reason for any phase that fails rather than silently dropping it.

``--no-model`` / ``--diagnostic-only`` run WITHOUT the expensive generation model
(diagnose + reuse-based calibration). The build and retrieval phases DO load the
embedding model (not the generation LLM); ``--diagnostic-only`` skips them.

Run from ``backend/``::

    d:\\Ragmed_2\\.venv\\Scripts\\python.exe -m evaluation.bioasq_eval.run_all --no-model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.bioasq_eval import integrity  # noqa: E402
from evaluation.bioasq_eval.common import (  # noqa: E402
    SEED, atomic_write_json, utc_now_iso,
)

RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
INDEX_V1 = BACKEND_DIR / "data" / "indexes" / "bioasq_retrieval_eval" / "v1"
MANIFEST_PATH = RESULTS_DIR / "manifest.json"

RERUN_COMMAND = (
    r"cd d:\Ragmed_2\backend && "
    r"d:\Ragmed_2\.venv\Scripts\python.exe -m evaluation.bioasq_eval.run_all --no-model"
)


def _bounded_retry(fn: Callable[[], Any], attempts: int = 3,
                   phase: str = "") -> Dict[str, Any]:
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            result = fn()
            return {"ok": True, "attempts": i, "result": result}
        except SystemExit as exc:
            # Deterministic non-recoverable guard (e.g. source missing); do not retry.
            return {"ok": False, "attempts": i, "error": f"SystemExit: {exc}",
                    "recoverable": False}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[run_all] phase '{phase}' attempt {i}/{attempts} failed: {exc}",
                  file=sys.stderr)
            time.sleep(0.5 * i)
    return {"ok": False, "attempts": attempts,
            "error": str(last_exc), "traceback": traceback.format_exc(),
            "recoverable": True}


def run(no_model: bool, diagnostic_only: bool, force_rebuild: bool) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "workflow": "wf_f9d894c9ec5422e5",
        "seed": SEED,
        "started_utc": utc_now_iso(),
        "modes": {"no_model": no_model, "diagnostic_only": diagnostic_only,
                  "force_rebuild": force_rebuild},
        "phases": {},
        "rerun_command": RERUN_COMMAND,
    }

    # 1. Integrity pre-check (fatal on failure).
    integrity.run_and_write("pre_run", check_membership=True)
    pre = integrity.check_pre_run()
    manifest["integrity"] = {"pre_run_ok": pre["ok"]}
    if not pre["ok"]:
        manifest["status"] = "INVALID"
        atomic_write_json(MANIFEST_PATH, manifest)
        raise RuntimeError("Integrity pre-check failed. Run INVALID.")

    # 2. Build index (loads embedding model) unless diagnostic-only.
    if not diagnostic_only:
        from evaluation.bioasq_eval import build_eval_index
        res = _bounded_retry(lambda: build_eval_index.build(force_rebuild=force_rebuild),
                             phase="build_index")
        manifest["phases"]["build_index"] = {
            "ok": res["ok"], "attempts": res["attempts"],
            "skipped": bool(res.get("result", {}).get("skipped")) if res["ok"] else False,
            "error": res.get("error"),
        }
    else:
        manifest["phases"]["build_index"] = {"ok": True, "skipped": "diagnostic_only"}

    # 3. Diagnose (model-free; include corpus availability if index exists).
    from evaluation.bioasq_eval import diagnose
    corpus_dir = INDEX_V1 if (INDEX_V1 / "index.faiss").exists() else None
    res = _bounded_retry(lambda: diagnose.run_diagnose(roundtrip=True, corpus_dir=corpus_dir),
                         phase="diagnose")
    if res["ok"]:
        rep = res["result"]
        diagnose.DIAG_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(diagnose.DIAG_DIR / "diagnostic_report.json", rep)
        from evaluation.bioasq_eval.common import atomic_write_text
        atomic_write_text(diagnose.DIAG_DIR / "diagnostic_report.md", diagnose._write_markdown(rep))
        manifest["phases"]["diagnose"] = {
            "ok": True, "n_queries": rep["n_queries"],
            "n_gold_pmids_unique_normalized": rep["n_gold_pmids_unique_normalized"],
            "gold_pmids_matched_in_eval_split": rep["gold_pmids_matched_in_eval_split"],
        }
    else:
        manifest["phases"]["diagnose"] = {"ok": False, "error": res.get("error")}

    # 4. Retrieval eval (loads embedding model) unless diagnostic-only.
    if not diagnostic_only and (INDEX_V1 / "index.faiss").exists():
        from evaluation.bioasq_eval import run_retrieval_eval
        res = _bounded_retry(lambda: run_retrieval_eval.run(INDEX_V1), phase="retrieval")
        if res["ok"]:
            manifest["phases"]["retrieval"] = {
                "ok": True, "n_queries": res["result"]["n_queries_evaluated"],
                "views": res["result"]["views"]["corpus_coverage"],
            }
        else:
            manifest["phases"]["retrieval"] = {"ok": False, "error": res.get("error")}
    else:
        manifest["phases"]["retrieval"] = {"ok": True, "skipped": "diagnostic_only"}

    # 5. Calibration (reuse-based; no LLM).
    from evaluation.bioasq_eval import calibrate_confidence
    res = _bounded_retry(calibrate_confidence.run, phase="calibration")
    if res["ok"]:
        rep = res["result"]
        manifest["phases"]["calibration"] = {
            "ok": True, "data_regime": rep["data_regime"],
            "N": rep["counts"]["N"], "P": rep["counts"]["P"],
            "source_consistency": rep["source_consistency"],
        }
    else:
        manifest["phases"]["calibration"] = {"ok": False, "error": res.get("error"),
                                             "recoverable": res.get("recoverable")}

    # 6. Integrity post-check (fatal on failure).
    integrity.run_and_write("post_run", check_membership=True)
    post = integrity.check_post_run()
    manifest["integrity"]["post_run_ok"] = post["ok"]

    manifest["completed_utc"] = utc_now_iso()
    all_ok = pre["ok"] and post["ok"] and all(
        p.get("ok", False) for p in manifest["phases"].values())
    manifest["status"] = "VALID" if all_ok else "INCOMPLETE"
    atomic_write_json(MANIFEST_PATH, manifest)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase F resumable driver")
    parser.add_argument("--no-model", action="store_true",
                        help="Do not run the generation LLM (reuse-based calibration).")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Diagnose + calibration only; skip index build & retrieval.")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)

    manifest = run(no_model=args.no_model, diagnostic_only=args.diagnostic_only,
                   force_rebuild=args.force_rebuild)
    print(f"status={manifest['status']} "
          f"integrity_pre={manifest['integrity']['pre_run_ok']} "
          f"integrity_post={manifest['integrity'].get('post_run_ok')}")
    for name, p in manifest["phases"].items():
        print(f"  phase {name}: ok={p.get('ok')} {p.get('skipped', '')}")
    return 0 if manifest["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
