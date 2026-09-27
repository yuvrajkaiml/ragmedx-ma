"""
BioASQ Dataset Loader

Loads BioASQ training data into the standard MedicalDocument representation.

Schema expected (Task 13b and compatible formats):
  {
    "questions": [
      {
        "id": "...",
        "body": "...",           -- question text (used as context title only)
        "snippets": [
          {
            "text":                 "...",  -- corpus content; one doc per snippet
            "document":             "http://www.ncbi.nlm.nih.gov/pubmed/PMID",
            "beginSection":         "title" | "abstract",
            "endSection":           "title" | "abstract",
            "offsetInBeginSection": int,
            "offsetInEndSection":   int
          },
          ...
        ],
        "documents":    [...],  -- PubMed URLs for provenance only, NOT corpus
        "ideal_answer": [...],  -- gold answers, NOT corpus content
        ...
      }
    ]
  }

Each snippet's ``text`` becomes one MedicalDocument.  The snippet's
``document`` URL is stored as ``source`` and in ``metadata`` for citation.
The parent question's ``documents`` list and ``ideal_answer`` are deliberately
ignored to prevent evaluation-answer leakage into the retrieval corpus.

Deduplication (exact-text) is the caller's responsibility; the loader
does not deduplicate across files to keep it stateless and composable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.rag.loaders.models import MedicalDocument
from app.utils.logger import logger


def _pmid_from_url(url: str) -> str:
    """Extract a bare PMID string from a PubMed URL, or return the URL as-is."""
    return url.rstrip("/").rsplit("/", 1)[-1] if url else ""


class BioASQLoader:
    """
    Loads BioASQ training data, emitting one MedicalDocument per snippet.

    Each snippet's ``text`` field is the corpus passage.  The snippet's
    ``document`` URL (PMID URL) is stored as the document source so the
    retrieval pipeline can surface it as a citation.

    The question body is stored as ``metadata["question"]`` for traceability
    but is NOT mixed into the passage text — the snippets are already
    extracted evidence spans and should be embedded as-is.
    """

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = Path(dataset_path)
        logger.info(f"BioASQLoader initialised: {self.dataset_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_dataset_files(self) -> List[Path]:
        json_files = sorted(self.dataset_path.glob("*.json"))
        logger.info(f"BioASQLoader found {len(json_files)} JSON file(s).")
        return json_files

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> List[MedicalDocument]:
        """
        Parse all JSON files in ``dataset_path`` and return one
        MedicalDocument per non-empty snippet text.

        Returns
        -------
        list[MedicalDocument]
            Documents in file-then-question-then-snippet order.
            Exact-text deduplication is NOT performed here; the caller
            (build script) deduplicates before chunking.
        """
        files = self._find_dataset_files()
        documents: List[MedicalDocument] = []

        for file_path in files:
            logger.info(f"BioASQLoader reading {file_path.name}")
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as err:
                logger.error(f"BioASQLoader failed to read {file_path.name}: {err}")
                continue

            questions = data.get("questions", [])
            logger.info(
                f"  {file_path.name}: {len(questions)} questions"
            )

            file_docs = 0
            for question in questions:
                question_id = question.get("id", "")
                question_body = question.get("body", "")
                snippets = question.get("snippets", [])

                # Deliberately skip: question["documents"] (provenance URLs only)
                # Deliberately skip: question["ideal_answer"]  (eval-answer leakage)

                for snippet_idx, snippet in enumerate(snippets):
                    text = snippet.get("text", "").strip()
                    if not text:
                        continue

                    pmid_url = snippet.get("document", "")
                    pmid = _pmid_from_url(pmid_url)

                    # Stable, collision-resistant ID: file stem + question id + snippet index
                    doc_id = f"bioasq_{question_id}_s{snippet_idx}"

                    # Build a short, human-readable title from the source section
                    begin_section = snippet.get("beginSection", "abstract")
                    end_section = snippet.get("endSection", "abstract")
                    section_label = begin_section if begin_section == end_section else f"{begin_section}–{end_section}"
                    title = f"PubMed {pmid} [{section_label}]" if pmid else f"BioASQ snippet {doc_id}"

                    documents.append(
                        MedicalDocument(
                            id=doc_id,
                            title=title,
                            content=text,
                            source=pmid_url or "bioasq",
                            metadata={
                                # Citation / provenance
                                "pmid_url": pmid_url,
                                "pmid": pmid,
                                "begin_section": begin_section,
                                "end_section": end_section,
                                "offset_begin": snippet.get("offsetInBeginSection", 0),
                                "offset_end": snippet.get("offsetInEndSection", 0),
                                # Traceability back to the originating question
                                "question_id": question_id,
                                "question": question_body[:300],
                                "source_file": file_path.name,
                            },
                        )
                    )
                    file_docs += 1

            logger.info(f"  {file_path.name}: emitted {file_docs} snippet documents")

        logger.success(
            f"BioASQLoader complete — {len(documents)} snippet documents from {len(files)} file(s)."
        )
        return documents
