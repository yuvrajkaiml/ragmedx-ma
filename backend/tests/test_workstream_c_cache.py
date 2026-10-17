"""
Phase 4b — QueryCache Tests.

Tests for backend/caching/query_cache.py.

All tests use temporary directories (no global state, no real FAISS index touched).
Evidence tier: SOFTWARE TEST.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add backend to path (pytest.ini sets pythonpath = app .)
# The caching package lives at backend/caching/ — import directly.
import sys

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from caching.query_cache import QueryCache, _key_hash, _normalize_query, _stable_key_string


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Return a fresh temp directory for each test."""
    d = tmp_path / "cache"
    d.mkdir()
    return d


def _make_key(
    query: str = "What is aspirin?",
    index_identity: str = "eval_split_v1",
    retrieval_config: dict = None,
    model_version: str = "Qwen3-8B",
    embedder_version: str = "bge-m3-v1",
    prompt_version: str = "v1",
    preprocessing_config: dict = None,
) -> Dict[str, Any]:
    """Helper: build a minimal valid cache key."""
    return {
        "query": query,
        "index_identity": index_identity,
        "retrieval_config": retrieval_config or {"model": "bge-m3", "k": 5, "reranker": "cross-encoder"},
        "model_version": model_version,
        "embedder_version": embedder_version,
        "prompt_version": prompt_version,
        "preprocessing_config": preprocessing_config or {"lowercase": False},
    }


def _make_value(answer: str = "Aspirin is an NSAID.") -> Dict[str, Any]:
    return {"answer": answer, "evidence_count": 3}


# ---------------------------------------------------------------------------
# 1. Cache hit / miss
# ---------------------------------------------------------------------------

class TestCacheHitMiss:
    def test_miss_on_empty_cache(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        result = cache.get(_make_key())
        assert result is None

    def test_hit_after_put(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        value = _make_value()
        cache.put(key, value)
        result = cache.get(key)
        assert result is not None
        assert result["answer"] == value["answer"]

    def test_put_then_get_roundtrip(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key(query="What are the symptoms of diabetes?")
        value = {"answer": "Polyuria, polydipsia, polyphagia.", "evidence_count": 5}
        cache.put(key, value)
        retrieved = cache.get(key)
        assert retrieved == value

    def test_get_returns_none_for_different_key(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        cache.put(_make_key(query="query A"), _make_value("Answer A"))
        result = cache.get(_make_key(query="query B"))
        assert result is None

    def test_overwrite_existing_entry(self, cache_dir: Path) -> None:
        """put() on existing key replaces the value."""
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value("First"))
        cache.put(key, _make_value("Second"))
        result = cache.get(key)
        assert result["answer"] == "Second"


# ---------------------------------------------------------------------------
# 2. Cache disabled → always miss
# ---------------------------------------------------------------------------

class TestCacheDisabled:
    def test_disabled_get_always_returns_none(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir), enabled=False)
        # Even after a (no-op) put, get() returns None
        key = _make_key()
        cache.put(key, _make_value())
        result = cache.get(key)
        assert result is None

    def test_disabled_put_is_noop(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir), enabled=False)
        key = _make_key()
        cache.put(key, _make_value())
        # No files should have been written
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 0

    def test_disabled_is_enabled_returns_false(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir), enabled=False)
        assert cache.is_enabled() is False

    def test_enabled_is_enabled_returns_true(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir), enabled=True)
        assert cache.is_enabled() is True

    def test_disabled_invalidate_is_noop(self, cache_dir: Path) -> None:
        """Calling invalidate on a disabled cache should not raise."""
        cache = QueryCache(str(cache_dir), enabled=False)
        cache.invalidate(_make_key())  # should not raise

    def test_disabled_invalidate_all_is_noop(self, cache_dir: Path) -> None:
        """Calling invalidate_all on a disabled cache should not raise."""
        cache = QueryCache(str(cache_dir), enabled=False)
        cache.invalidate_all()  # should not raise


# ---------------------------------------------------------------------------
# 3. Different index_identity → different cache entry (cross-config isolation)
# ---------------------------------------------------------------------------

class TestIndexIdentityIsolation:
    def test_different_index_identity_different_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_eval = _make_key(index_identity="eval_split_v1")
        key_prod = _make_key(index_identity="production_index")
        value_eval = _make_value("Eval answer")
        value_prod = _make_value("Prod answer")

        cache.put(key_eval, value_eval)
        cache.put(key_prod, value_prod)

        assert cache.get(key_eval)["answer"] == "Eval answer"
        assert cache.get(key_prod)["answer"] == "Prod answer"

    def test_eval_index_does_not_bleed_into_production(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_eval = _make_key(index_identity="eval_split_v1")
        cache.put(key_eval, _make_value("Eval answer"))

        # Same key but different index_identity → miss
        key_prod = _make_key(index_identity="production_index")
        assert cache.get(key_prod) is None

    def test_key_hash_differs_by_index_identity(self) -> None:
        key_a = _make_key(index_identity="eval_split_v1")
        key_b = _make_key(index_identity="production_index")
        assert _key_hash(key_a) != _key_hash(key_b)

    def test_same_query_different_index_two_files_created(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        cache.put(_make_key(index_identity="index_A"), _make_value())
        cache.put(_make_key(index_identity="index_B"), _make_value())
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 2


# ---------------------------------------------------------------------------
# 4. Different retrieval_config → different cache entry
# ---------------------------------------------------------------------------

class TestRetrievalConfigIsolation:
    def test_different_retrieval_config_different_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_k5 = _make_key(retrieval_config={"model": "bge-m3", "k": 5, "reranker": "cross-encoder"})
        key_k10 = _make_key(retrieval_config={"model": "bge-m3", "k": 10, "reranker": "cross-encoder"})
        cache.put(key_k5, _make_value("k=5 answer"))
        cache.put(key_k10, _make_value("k=10 answer"))

        assert cache.get(key_k5)["answer"] == "k=5 answer"
        assert cache.get(key_k10)["answer"] == "k=10 answer"

    def test_different_reranker_different_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_ce = _make_key(retrieval_config={"model": "bge-m3", "k": 5, "reranker": "cross-encoder"})
        key_no = _make_key(retrieval_config={"model": "bge-m3", "k": 5, "reranker": "none"})
        cache.put(key_ce, _make_value("cross-encoder answer"))
        result = cache.get(key_no)
        assert result is None

    def test_key_hash_differs_by_retrieval_config(self) -> None:
        key_a = _make_key(retrieval_config={"k": 5})
        key_b = _make_key(retrieval_config={"k": 10})
        assert _key_hash(key_a) != _key_hash(key_b)


# ---------------------------------------------------------------------------
# 5. Different model_version → different cache entry
# ---------------------------------------------------------------------------

class TestModelVersionIsolation:
    def test_different_model_version_different_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_v1 = _make_key(model_version="Qwen3-8B")
        key_v2 = _make_key(model_version="Qwen3-14B")
        cache.put(key_v1, _make_value("v1 answer"))

        assert cache.get(key_v2) is None

    def test_key_hash_differs_by_model_version(self) -> None:
        key_a = _make_key(model_version="Qwen3-8B")
        key_b = _make_key(model_version="Qwen3-14B")
        assert _key_hash(key_a) != _key_hash(key_b)


# ---------------------------------------------------------------------------
# 6. Invalidation removes correct entry
# ---------------------------------------------------------------------------

class TestInvalidation:
    def test_invalidate_removes_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value())
        cache.invalidate(key)
        assert cache.get(key) is None

    def test_invalidate_removes_only_targeted_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_a = _make_key(query="Query A")
        key_b = _make_key(query="Query B")
        cache.put(key_a, _make_value("Answer A"))
        cache.put(key_b, _make_value("Answer B"))

        cache.invalidate(key_a)

        assert cache.get(key_a) is None
        result_b = cache.get(key_b)
        assert result_b is not None
        assert result_b["answer"] == "Answer B"

    def test_invalidate_nonexistent_key_is_noop(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        cache.invalidate(_make_key())  # should not raise

    def test_invalidate_only_removes_one_file(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key_a = _make_key(query="Query A")
        key_b = _make_key(query="Query B")
        cache.put(key_a, _make_value())
        cache.put(key_b, _make_value())
        assert len(list(cache_dir.glob("*.json"))) == 2

        cache.invalidate(key_a)
        assert len(list(cache_dir.glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# 7. invalidate_all() clears cache
# ---------------------------------------------------------------------------

class TestInvalidateAll:
    def test_invalidate_all_clears_all_entries(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        for i in range(5):
            cache.put(_make_key(query=f"Query {i}"), _make_value())
        assert len(list(cache_dir.glob("*.json"))) == 5

        cache.invalidate_all()
        assert len(list(cache_dir.glob("*.json"))) == 0

    def test_invalidate_all_then_get_returns_none(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value())
        cache.invalidate_all()
        assert cache.get(key) is None

    def test_invalidate_all_on_empty_cache_is_noop(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        cache.invalidate_all()  # should not raise
        assert len(list(cache_dir.glob("*.json"))) == 0

    def test_cache_usable_after_invalidate_all(self, cache_dir: Path) -> None:
        """Cache should work normally after invalidate_all."""
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value("First"))
        cache.invalidate_all()
        cache.put(key, _make_value("Second"))
        assert cache.get(key)["answer"] == "Second"


# ---------------------------------------------------------------------------
# 8. Atomic write behaviour
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_cache_file_exists_after_put(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value())
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 1

    def test_no_tmp_files_left_after_put(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        cache.put(_make_key(), _make_value())
        tmp_files = list(cache_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_cache_file_is_valid_json(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value())
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0], "r", encoding="utf-8") as fh:
            entry = json.load(fh)
        assert "key" in entry
        assert "key_hash" in entry
        assert "value" in entry

    def test_cache_file_name_is_sha256_hex(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value())
        json_files = list(cache_dir.glob("*.json"))
        fname = json_files[0].stem
        # SHA-256 hex digest is 64 hex characters
        assert len(fname) == 64
        assert all(c in "0123456789abcdef" for c in fname)


# ---------------------------------------------------------------------------
# 9. Key validation
# ---------------------------------------------------------------------------

class TestKeyValidation:
    def test_missing_required_field_raises_value_error(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        incomplete_key = {"query": "test", "index_identity": "eval_v1"}
        with pytest.raises(ValueError, match="missing required fields"):
            cache.get(incomplete_key)

    def test_missing_required_field_on_put_raises(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        incomplete_key = {"query": "test"}
        with pytest.raises(ValueError, match="missing required fields"):
            cache.put(incomplete_key, _make_value())

    def test_missing_required_field_on_invalidate_raises(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        with pytest.raises(ValueError, match="missing required fields"):
            cache.invalidate({"query": "test"})


# ---------------------------------------------------------------------------
# 10. Query normalization
# ---------------------------------------------------------------------------

class TestQueryNormalization:
    def test_whitespace_normalized(self) -> None:
        assert _normalize_query("  What  is  aspirin? ") == "What is aspirin?"

    def test_normalized_query_hits_same_entry(self, cache_dir: Path) -> None:
        cache = QueryCache(str(cache_dir))
        key1 = _make_key(query="What is aspirin?")
        key2 = _make_key(query="  What  is  aspirin? ")
        cache.put(key1, _make_value("Aspirin is an NSAID."))
        result = cache.get(key2)
        assert result is not None
        assert result["answer"] == "Aspirin is an NSAID."

    def test_key_hash_same_for_whitespace_variants(self) -> None:
        key1 = _make_key(query="What is aspirin?")
        key2 = _make_key(query="  What  is  aspirin?  ")
        assert _key_hash(key1) == _key_hash(key2)


# ---------------------------------------------------------------------------
# 11. Corrupted entry treated as miss
# ---------------------------------------------------------------------------

class TestCorruptedEntry:
    def test_corrupted_json_treated_as_miss(self, cache_dir: Path, caplog) -> None:
        import logging
        cache = QueryCache(str(cache_dir))
        key = _make_key()
        cache.put(key, _make_value())

        # Corrupt the file
        from caching.query_cache import _key_hash
        digest = _key_hash(key)
        entry_path = cache_dir / f"{digest}.json"
        with open(entry_path, "w", encoding="utf-8") as fh:
            fh.write("{ invalid json }")

        with caplog.at_level(logging.WARNING, logger="caching.query_cache"):
            result = cache.get(key)
        assert result is None
        assert any("corrupted" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 12. Process restart survival (persistence across instances)
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_entry_survives_new_instance(self, cache_dir: Path) -> None:
        """
        Cache is file-based; a new QueryCache instance on the same directory
        should see entries written by the previous instance.
        """
        key = _make_key()
        value = _make_value("Persistent answer")

        # Instance 1: write
        cache1 = QueryCache(str(cache_dir))
        cache1.put(key, value)

        # Instance 2: read
        cache2 = QueryCache(str(cache_dir))
        result = cache2.get(key)
        assert result is not None
        assert result["answer"] == "Persistent answer"

    def test_in_memory_only_dict_would_fail_this_test(self, cache_dir: Path) -> None:
        """
        Demonstrate that an in-memory-only cache would fail the persistence test.
        (This test is documentation; it always passes because our implementation is file-based.)
        """
        key = _make_key(query="persistence test")
        value = _make_value("stored")

        cache = QueryCache(str(cache_dir))
        cache.put(key, value)

        # Verify there is actually a file on disk
        json_files = list(cache_dir.glob("*.json"))
        assert len(json_files) >= 1, (
            "QueryCache must use file-based storage (no in-memory-only dict). "
            "An in-memory cache would not write any files."
        )


# ---------------------------------------------------------------------------
# 13. Default-OFF: cache is a standalone module, NOT wired into the pipeline
# ---------------------------------------------------------------------------

class TestCacheDefaultOffWiring:
    """
    Versioning-rule compliance (Phase 4b / FEAT-002).

    backend/caching/query_cache.py is a standalone module. It must NOT be
    imported or constructed by the default retrieval/agent pipeline so that
    default behaviour is identical to before the cache was added — i.e. the
    new behaviour is default OFF. Evidence tier: SOFTWARE TEST.
    """

    def test_default_pipeline_does_not_import_or_construct_query_cache(self) -> None:
        import ast
        import subprocess

        # (1) Static source inspection.
        #
        # orchestrator.py must never reference the cache at all (the cache lives
        # entirely on the retrieval boundary).
        #
        # pipeline.py (Phase 5, P5-3) now wires an OPTIONAL QueryCache behind a
        # default-OFF `cache_enabled` flag. The versioning guarantee is NOT that
        # the string is absent, but that the caching package is imported LAZILY
        # (inside the cache_enabled branch only) so that the default path never
        # imports or constructs a cache. We therefore assert that any `caching`
        # import in pipeline.py is NOT at module top level (indentation > 0),
        # and that orchestrator.py has no cache reference whatsoever. The
        # authoritative default-OFF proof is the runtime subprocess check (2).
        pipeline_src = (_BACKEND_DIR / "app" / "rag" / "pipeline.py").read_text(encoding="utf-8")
        orchestrator_src = (_BACKEND_DIR / "app" / "agents" / "orchestrator.py").read_text(encoding="utf-8")

        assert "QueryCache" not in orchestrator_src, (
            "orchestrator.py must not reference QueryCache (cache lives on the retrieval boundary)"
        )
        assert "from caching" not in orchestrator_src and "import caching" not in orchestrator_src, (
            "orchestrator.py must not import the caching package"
        )

        # Any caching import in pipeline.py must be indented (lazy, inside a
        # method branch) — never a module-level import.
        for line in pipeline_src.splitlines():
            stripped = line.strip()
            if stripped.startswith("from caching") or stripped.startswith("import caching"):
                assert line != stripped, (
                    "pipeline.py must import the caching package lazily (indented, "
                    f"inside the cache_enabled branch), not at module level: {line!r}"
                )

        # (2) Runtime import-time check in a fresh subprocess: importing the
        # two default-pipeline modules must not pull `caching` (or
        # `caching.query_cache`) into sys.modules. A subprocess is required
        # because this test process already imported the cache package above.
        repo_root = _BACKEND_DIR.parent
        script = (
            "import sys; sys.path.insert(0, 'backend');"
            "import app.rag.pipeline, app.agents.orchestrator;"
            "print(repr(sorted(m for m in sys.modules"
            " if m == 'caching' or m == 'caching.query_cache'"
            " or m.startswith('caching.'))))"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Default-pipeline import failed in subprocess:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        pulled_lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("[")]
        assert pulled_lines, (
            f"No caching-modules result line in subprocess stdout: {result.stdout!r}"
        )
        pulled = ast.literal_eval(pulled_lines[-1])  # output is repr() of a sorted list
        assert pulled == [], (
            "Default pipeline must not import the caching package at module load, "
            f"but these caching modules were pulled in: {pulled}"
        )
