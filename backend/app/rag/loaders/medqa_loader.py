import json

from pathlib import Path
from typing import List

from app.utils.logger import logger
from app.rag.loaders.models import MedicalDocument


class MedQALoader:
    """
    Loads and parses the MedQA dataset.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

        logger.info(f"MedQA Loader initialized: {self.dataset_path}")

    def _find_dataset_files(self) -> List[Path]:
        """
        Find all JSON and JSONL files in the dataset directory.
        """

        json_files = list(self.dataset_path.glob("*.json"))
        jsonl_files = list(self.dataset_path.glob("*.jsonl"))

        dataset_files = json_files + jsonl_files

        logger.info(f"Found {len(dataset_files)} dataset file(s).")

        return dataset_files

    def load(self) -> List[MedicalDocument]:
        """
        Load and parse all MedQA dataset files.
        """

        files = self._find_dataset_files()
        logger.info(f"Preparing to load {len(files)} file(s).")
        documents: List[MedicalDocument] = []
        for file in files:
            logger.info(f"Reading {file.name}")
            with open(file, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        document = MedicalDocument(
                        id=f"{file.stem}_{index}",
                        title=record.get("question", "")[:80],
                        content=record.get("question", ""),
                        source="MedQA",
                        metadata={
                            "answer": record.get("answer"),
                            "answer_idx": record.get("answer_idx"),
                            "options": record.get("options"),
                            "meta_info": record.get("meta_info")
                        }
                    )
                        documents.append(document)
                    except json.JSONDecodeError:
                        logger.warning(
                        f"Skipping invalid JSON in {file.name}"
                    )
        logger.success(
            f"Loaded {len(documents)} MedQA documents."
        )
        return documents