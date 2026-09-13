"""
BioASQ Dataset Loader

Loads and parses the BioASQ training dataset into the
standard MedicalDocument representation.

Author: Sana
"""

from pathlib import Path
from typing import List
import json

from app.utils.logger import logger
from app.rag.loaders.models import MedicalDocument


class BioASQLoader:
    """
    Loads and parses the BioASQ dataset.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

        logger.info(
            f"BioASQ Loader initialized: {self.dataset_path}"
        )

    def _find_dataset_files(self) -> List[Path]:
        """
        Find JSON files in the BioASQ dataset directory.
        """

        json_files = list(self.dataset_path.glob("*.json"))

        logger.info(
            f"Found {len(json_files)} BioASQ dataset file(s)."
        )

        return json_files

    def load(self) -> List[MedicalDocument]:
        """
        Load BioASQ questions and convert them into
        MedicalDocument objects.
        """

        files = self._find_dataset_files()

        documents: List[MedicalDocument] = []

        for file_path in files:

            logger.info(f"Reading {file_path.name}")

            try:
                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

            except (json.JSONDecodeError, OSError) as error:

                logger.error(
                    f"Failed to read {file_path.name}: {error}"
                )

                continue

            questions = data.get("questions", [])

            logger.info(
                f"Found {len(questions)} questions in {file_path.name}"
            )

            for question in questions:

                question_id = question.get("id", "")
                question_body = question.get("body", "")

                snippets = question.get("snippets", [])

                # Extract the actual evidence text from snippets
                snippet_texts = []

                for snippet in snippets:

                    text = snippet.get("text", "").strip()

                    if text:
                        snippet_texts.append(text)

                # Combine question and evidence into searchable content
                content_parts = [question_body]

                if snippet_texts:
                    content_parts.append(
                        "\n\n".join(snippet_texts)
                    )

                content = "\n\n".join(content_parts)

                metadata = {
                    "documents": question.get("documents", []),
                    "ideal_answer": question.get("ideal_answer", []),
                    "concepts": question.get("concepts", []),
                    "type": question.get("type"),
                    "snippets": snippets,
                }

                medical_document = MedicalDocument(
                    id=f"bioasq_{question_id}",
                    title=question_body[:200],
                    content=content,
                    source="BioASQ",
                    metadata=metadata,
                )

                documents.append(medical_document)

        logger.success(
            f"Loaded {len(documents)} BioASQ documents."
        )

        return documents