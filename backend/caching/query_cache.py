"""
Phase 4b — QueryCache: File-backed query result cache.

Cache key includes ALL of:
  - normalized query text
  - index_identity (hash or path of the index used)
  - retrieval_config (model, k, reranker settings as a stable dict)
  - model_version
  - embedder_version
  - prompt_version
  - preprocessing_config

Storage: one JSON file per cache entry under cache_dir.
File name is derived from the SHA-256 of the stable key string,
so it survives process restarts.

Rules enforced (see test_workstream_c_cache.py):
  1. Disabled cache: get() always returns None, put() is a no-op
  2. Different index_identity  → different cache entry (cross-config isolation)
  3. Different retrieval_config → different cache entry
  4. Different model_version   → different cache entry
  5. Cache never stores results from the component under test (caller responsibility)
  6. Invalidation clears only the targeted entries
  7. Cache misses are logged (not silently swallowed)
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Required key fields; extras are allowed and included in the hash.
_REQUIRED_KEY_FIELDS = frozenset(
    [
        "query",
        "index_identity",
        "retrieval_config",
        "model_version",
        "embedder_version",
        "prompt_version",
        "preprocessing_config",
    ]
)


def _normalize_query(text: str) -> str:
    """
    Normalize query text for stable cache key computation.
    Strips leading/trailing whitespace and collapses internal runs of whitespace.
    """
    return " ".join(text.split())


def _stable_key_string(key: Dict[str, Any]) -> str:
    """
    Produce a deterministic string representation of the cache key dict.

    All missing optional fields default to the empty string so that a key
    with an explicit empty field and a key with the field omitted are treated
    identically.  The query text is normalized before hashing.
    """
    normalized: Dict[str, Any] = {}
    for field in sorted(key.keys()):
        value = key[field]
        if field == "query":
            value = _normalize_query(str(value))
        normalized[field] = value
    return json.dumps(normalized, sort_keys=True, ensure_ascii=True)


def _key_hash(key: Dict[str, Any]) -> str:
    """Return the SHA-256 hex digest of the stable key string."""
    key_str = _stable_key_string(key)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def _validate_key(key: Dict[str, Any]) -> None:
    """Raise ValueError if any required field is missing from the key."""
    missing = _REQUIRED_KEY_FIELDS - set(key.keys())
    if missing:
        raise ValueError(
            f"Cache key is missing required fields: {sorted(missing)}. "
            f"Provided fields: {sorted(key.keys())}"
        )


class QueryCache:
    """
    File-backed query result cache.

    Each cache entry is stored as a JSON file:
        <cache_dir>/<sha256_of_key>.json

    The file contains:
        {
            "key": { ...original key dict... },
            "key_hash": "<sha256>",
            "value": { ...cached result... }
        }

    Args:
        cache_dir: Directory where cache files are stored.
                   Created if it does not exist.
        enabled:   If False, get() always returns None and put() is a no-op.
    """

    def __init__(self, cache_dir: str, enabled: bool = True) -> None:
        self._cache_dir = Path(cache_dir)
        self._enabled = enabled
        if self._enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True if caching is active."""
        return self._enabled

    def get(self, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Return the cached value for *key*, or None on a miss.

        Cache misses are logged at DEBUG level (rule 7: never silently swallowed).

        Args:
            key: Must contain all required fields (see module docstring).

        Returns:
            Cached result dict, or None on miss or when cache is disabled.

        Raises:
            ValueError: If a required key field is missing.
        """
        _validate_key(key)

        if not self._enabled:
            logger.debug("QueryCache.get: cache disabled — returning None (miss)")
            return None

        digest = _key_hash(key)
        entry_path = self._cache_dir / f"{digest}.json"

        if not entry_path.exists():
            logger.debug(
                "QueryCache.get: miss for key_hash=%s (file not found)", digest
            )
            return None

        try:
            with open(entry_path, "r", encoding="utf-8") as fh:
                entry = json.load(fh)
            logger.debug("QueryCache.get: hit for key_hash=%s", digest)
            return entry.get("value")
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            # Corrupted entry — treat as a miss and log
            logger.warning(
                "QueryCache.get: corrupted entry at %s (%s) — treating as miss",
                entry_path,
                exc,
            )
            return None

    def put(self, key: Dict[str, Any], value: Dict[str, Any]) -> None:
        """
        Store *value* under *key*.

        Uses an atomic write (same-directory temp file + os.replace) so that
        a process crash during the write never leaves a partial file visible
        to concurrent readers.

        Args:
            key:   Must contain all required fields.
            value: Result dict to cache.

        Raises:
            ValueError: If a required key field is missing.
        """
        _validate_key(key)

        if not self._enabled:
            logger.debug("QueryCache.put: cache disabled — no-op")
            return

        digest = _key_hash(key)
        entry_path = self._cache_dir / f"{digest}.json"
        entry = {
            "key": key,
            "key_hash": digest,
            "value": value,
        }
        entry_json = json.dumps(entry, indent=2, ensure_ascii=False)

        # Atomic write: write to a temp file in the same directory, then replace.
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", dir=self._cache_dir
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(entry_json)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, entry_path)
            logger.debug("QueryCache.put: stored key_hash=%s at %s", digest, entry_path)
        except Exception:
            # Clean up temp file if replace failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def invalidate(self, key: Dict[str, Any]) -> None:
        """
        Remove the cache entry for *key*.

        If the entry does not exist, this is a no-op.

        Args:
            key: Must contain all required fields.

        Raises:
            ValueError: If a required key field is missing.
        """
        _validate_key(key)

        if not self._enabled:
            return

        digest = _key_hash(key)
        entry_path = self._cache_dir / f"{digest}.json"

        if entry_path.exists():
            entry_path.unlink()
            logger.debug(
                "QueryCache.invalidate: removed key_hash=%s", digest
            )
        else:
            logger.debug(
                "QueryCache.invalidate: key_hash=%s not found (no-op)", digest
            )

    def invalidate_all(self) -> None:
        """
        Remove all cache entries in cache_dir.

        Only files matching the *.json pattern are removed (preserves any
        non-cache files that might have been placed in the directory).
        """
        if not self._enabled:
            return

        removed = 0
        for entry_path in self._cache_dir.glob("*.json"):
            try:
                entry_path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning(
                    "QueryCache.invalidate_all: could not remove %s: %s",
                    entry_path,
                    exc,
                )
        logger.debug("QueryCache.invalidate_all: removed %d entries", removed)
