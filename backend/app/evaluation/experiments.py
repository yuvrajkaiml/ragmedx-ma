# Owner: Pranav Sukale
"""Experiment configuration and result tracking.

An ``ExperimentConfig`` describes one reproducible evaluation run:
which dataset, which pipeline configuration, and which metrics to report.
``ExperimentResult`` records the outcome and can be serialised to JSON
for later analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import EXPERIMENTS_DIR


@dataclass
class ExperimentConfig:
    """Describes one reproducible evaluation experiment."""

    name: str
    dataset_key: str                        # key in app.evaluation.datasets.DATASETS
    pipeline_config: str = "full"           # "full" | "no_critic" | "no_mga" | "baseline_rag"
    llm_mode: str = "mock"                  # "mock" | "real"
    max_samples: Optional[int] = None       # None = all samples
    top_k_retrieval: int = 10
    top_k_final: int = 5
    include_agent_trace: bool = False
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentResult:
    """Stores the outcome of one completed experiment run."""

    config: ExperimentConfig
    metrics: Dict[str, Any]
    samples_evaluated: int
    started_at: str
    completed_at: str
    status: str = "completed"       # "completed" | "failed" | "partial"
    error: Optional[str] = None
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Persist the result as a JSON file under ``EXPERIMENTS_DIR``."""
        directory = output_dir or EXPERIMENTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = self.config.name.replace(" ", "_").replace("/", "-")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = directory / f"{safe_name}_{ts}.json"
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load(cls, path: Path) -> "ExperimentResult":
        """Load a previously saved result from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        config = ExperimentConfig(**data.pop("config"))
        return cls(config=config, **data)


def list_experiments(directory: Optional[Path] = None) -> List[Path]:
    """Return paths of all saved experiment JSON files, newest first."""
    directory = directory or EXPERIMENTS_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
