"""
Shared helpers for the BioASQ retrieval-eval + calibration pipeline.

- PMID normalization and canonical-PMID extraction from a MedicalChunk.
- Chunk -> article (PMID) rank-preserving dedup.
- Atomic JSON writes and a resumable JSONL checkpoint.
- Config hashing and per-phase manifest writing.

All paths are anchored on BACKEND_DIR / REPO_ROOT (never the process cwd).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BACKEND_DIR = Path(__file__).resolve().parents[2]   # d:\Ragmed_2\backend
REPO_ROOT = Path(__file__).resolve().parents[3]     # d:\Ragmed_2

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "results" / "bioasq_eval_20261018"
MANIFESTS_DIR = RESULTS_DIR / "manifests"

SEED = 42

_PMID_RE = re.compile(r"(\d+)$")


# ---------------------------------------------------------------------------
# PMID normalization
# ---------------------------------------------------------------------------

def normalize_pmid(value: Any) -> Optional[str]:
    """Return the trailing integer run of ``value`` as a string, else None.

    Handles full PubMed URLs (``http://.../pubmed/29040384`` -> ``"29040384"``),
    bare ints, and differently-formatted strings. Returns None for unparsable
    input (empty, None, or no trailing digits), which the caller counts.
    """
    if value is None:
        return None
    m = _PMID_RE.search(str(value).strip())
    return m.group(1) if m else None


def canonical_pmid(chunk: Any, fallback_counter: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Canonical PMID for a MedicalChunk: ``metadata['pmid']`` normalized.

    Falls back to parsing ``chunk.source`` when ``metadata['pmid']`` is absent;
    increments ``fallback_counter['pmid_fallback_used']`` when the fallback is used.
    """
    meta = getattr(chunk, "metadata", None) or {}
    raw = meta.get("pmid")
    if raw is not None:
        norm = normalize_pmid(raw)
        if norm is not None:
            return norm
    # Fallback: parse the source URL/string.
    source = getattr(chunk, "source", None)
    norm = normalize_pmid(source)
    if norm is not None and fallback_counter is not None:
        fallback_counter["pmid_fallback_used"] = fallback_counter.get("pmid_fallback_used", 0) + 1
    return norm


def dedup_articles(ranked_chunks: Iterable[Any],
                   fallback_counter: Optional[Dict[str, int]] = None) -> List[str]:
    """Collapse a rank-ordered chunk list to a rank-ordered unique PMID list.

    Preserves first (highest-ranked) occurrence. Chunks with no recoverable PMID
    are skipped. Multiple chunks of one article therefore collapse to a single
    article entry at the rank of their first appearance.
    """
    seen: set = set()
    articles: List[str] = []
    for chunk in ranked_chunks:
        pmid = canonical_pmid(chunk, fallback_counter)
        if pmid is None or pmid in seen:
            continue
        seen.add(pmid)
        articles.append(pmid)
    return articles


# ---------------------------------------------------------------------------
# Atomic IO + resumable checkpoint
# ---------------------------------------------------------------------------

def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Write JSON atomically: write ``*.tmp`` then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


class JsonlCheckpoint:
    """Append-with-flush JSONL writer that supports resume by a record key.

    On construction it loads the set of completed keys already present in the
    file (if any). ``is_done(key)`` tells the caller to skip; ``append(record)``
    writes one JSON line, flushes, and records the key. Duplicate keys are never
    written twice.
    """

    def __init__(self, path: str | Path, key_field: str = "question_id") -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Corrupt/truncated final line: ignore (will be re-appended).
                    continue
                key = rec.get(self.key_field)
                if key is not None:
                    self._done.add(key)

    @property
    def done_keys(self) -> set:
        return set(self._done)

    def is_done(self, key: Any) -> bool:
        return key in self._done

    def append(self, record: Dict[str, Any]) -> bool:
        """Append a record. Returns False (no write) if the key is already done."""
        key = record.get(self.key_field)
        if key is not None and key in self._done:
            return False
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if key is not None:
            self._done.add(key)
        return True

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every valid JSON record from the file."""
        out: List[Dict[str, Any]] = []
        if not self.path.exists():
            return out
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


# ---------------------------------------------------------------------------
# Config hashing + manifests
# ---------------------------------------------------------------------------

def config_hash(cfg: Dict[str, Any]) -> str:
    """SHA-256 of the canonical (sorted-key) JSON encoding of a config dict."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> Optional[str]:
    """SHA-256 of a file, or None if it does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("numpy", "faiss", "torch", "transformers", "sentence_transformers", "sklearn"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover - best effort
            versions[name] = "not-installed"
    return versions


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(phase: str, cfg: Dict[str, Any],
                   inputs: Dict[str, str | Path],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``manifests/<phase>.manifest.json`` capturing full reproducibility info."""
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "phase": phase,
        "seed": SEED,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "inputs": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in inputs.items()},
        "embedding_model": "BAAI/bge-m3",
        "package_versions": _package_versions(),
        "timestamp_utc": utc_now_iso(),
    }
    if extra:
        manifest.update(extra)
    atomic_write_json(MANIFESTS_DIR / f"{phase}.manifest.json", manifest)
    return manifest
