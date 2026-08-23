"""
Shared pytest fixtures and configuration for the backend test suite.

asyncio_mode = auto is set in pytest.ini, so every ``async def test_*``
function is collected and run automatically — no @pytest.mark.asyncio needed.
"""

import json
from pathlib import Path

import pytest


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
