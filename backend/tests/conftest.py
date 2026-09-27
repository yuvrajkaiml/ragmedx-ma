"""
Shared pytest fixtures and configuration for the backend test suite.

asyncio_mode = auto is set in pytest.ini, so every ``async def test_*``
function is collected and run automatically — no @pytest.mark.asyncio needed.
"""

import json
import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Ensure tests always run in mock LLM mode regardless of shell environment.
#
# RAGMEDX_LLM_MODE=real in the developer's shell (set for running the server
# or build script) must not bleed into pytest — real mode makes
# RetrievalService.build() raise when the index path doesn't resolve from
# the test runner's cwd, and makes LocalLLMWrapper attempt GPU model loading.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def force_mock_llm_mode():
    """Remove RAGMEDX_LLM_MODE from the process environment for the test session
    and clear the lru_cache so every get_settings() call sees mock mode."""
    original = os.environ.pop("RAGMEDX_LLM_MODE", None)
    # Clear cached Settings singleton so the mode change takes effect.
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    # Restore after the session (safe for interactive re-runs in the same shell).
    if original is not None:
        os.environ["RAGMEDX_LLM_MODE"] = original
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR.parent / "data"
DATASETS_DIR = DATA_DIR / "datasets"


# ---------------------------------------------------------------------------
# MedQA fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def medqa_sample_path() -> Path:
    """Return the path to the bundled MedQA sample JSONL fixture."""
    path = DATASETS_DIR / "medqa" / "sample.jsonl"
    assert path.exists(), f"MedQA sample fixture not found: {path}"
    return path


@pytest.fixture(scope="session")
def medqa_sample_records(medqa_sample_path: Path) -> list[dict]:
    """Parse and return all records from the MedQA sample fixture."""
    records = []
    with open(medqa_sample_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert len(records) >= 1, "MedQA sample fixture must contain at least one record."
    return records
