# Owner: Pranav Sukale
"""Evaluation dataset registry.

Defines metadata for every evaluation dataset the system supports.
The registry is consumed by the benchmark runner and experiment tracker
so dataset paths, splits, and task types are declared in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.config import DATASETS_DIR


@dataclass
class DatasetSpec:
    """Metadata descriptor for one evaluation dataset."""

    name: str
    task_type: str          # "bioasq" | "medqa" | "pubmedqa" | "mirage"
    split: str              # "train" | "dev" | "test" | "all"
    path: Path
    description: str
    answer_field: str = "answer"
    question_field: str = "question"
    answerable_field: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return self.path.exists()


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: Dict[str, DatasetSpec] = {
    "bioasq_train": DatasetSpec(
        name="BioASQ 13b Training",
        task_type="bioasq",
        split="train",
        path=DATASETS_DIR / "bioasq" / "training13b.json",
        description=(
            "BioASQ Task 13b training set — 5,389 biomedical questions with "
            "PubMed document references and ideal answers."
        ),
        answer_field="ideal_answer",
        question_field="body",
        tags=["biomedical", "open-domain", "pubmed"],
    ),
    "medqa_train": DatasetSpec(
        name="MedQA Training",
        task_type="medqa",
        split="train",
        path=DATASETS_DIR / "medqa" / "train.jsonl",
        description="MedQA USMLE-style multiple-choice training split.",
        answer_field="answer",
        question_field="question",
        tags=["usmle", "multiple-choice", "clinical"],
    ),
    "medqa_dev": DatasetSpec(
        name="MedQA Dev",
        task_type="medqa",
        split="dev",
        path=DATASETS_DIR / "medqa" / "dev.jsonl",
        description="MedQA USMLE-style multiple-choice development split.",
        answer_field="answer",
        question_field="question",
        tags=["usmle", "multiple-choice", "clinical"],
    ),
    "medqa_test": DatasetSpec(
        name="MedQA Test",
        task_type="medqa",
        split="test",
        path=DATASETS_DIR / "medqa" / "test.jsonl",
        description="MedQA USMLE-style multiple-choice test split.",
        answer_field="answer",
        question_field="question",
        tags=["usmle", "multiple-choice", "clinical"],
    ),
    "medqa_sample": DatasetSpec(
        name="MedQA Sample Fixture",
        task_type="medqa",
        split="dev",
        path=DATASETS_DIR / "medqa" / "sample.jsonl",
        description="4-record MedQA fixture for fast CI smoke tests.",
        answer_field="answer",
        question_field="question",
        tags=["fixture", "ci"],
    ),
}


def get_dataset(key: str) -> DatasetSpec:
    """Return a ``DatasetSpec`` by registry key.

    Raises
    ------
    KeyError
        If the key is not registered.
    FileNotFoundError
        If the dataset file does not exist on disk.
    """
    spec = DATASETS[key]
    if not spec.exists:
        raise FileNotFoundError(
            f"Dataset '{key}' not found at {spec.path}. "
            "Download or place the file before running evaluation."
        )
    return spec


def list_available() -> List[str]:
    """Return keys of datasets whose files are present on disk."""
    return [key for key, spec in DATASETS.items() if spec.exists]
