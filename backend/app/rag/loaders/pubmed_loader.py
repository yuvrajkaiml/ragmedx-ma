"""
PubMed Dataset Loader

Loads and parses PubMed baseline XML files and converts
articles into the standard MedicalDocument representation.

Author: Sana
"""

from pathlib import Path
from typing import List
import gzip
import xml.etree.ElementTree as ET

from app.utils.logger import logger
from app.rag.loaders.models import MedicalDocument


class PubMedLoader:
    """
    Loads and parses PubMed baseline XML files.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

        logger.info(
            f"PubMed Loader initialized: {self.dataset_path}"
        )

    def _find_dataset_files(self) -> List[Path]:
        """
        Find compressed PubMed XML files.
        """

        dataset_files = list(
            self.dataset_path.glob("*.xml.gz")
        )

        logger.info(
            f"Found {len(dataset_files)} PubMed dataset file(s)."
        )

        return dataset_files

    @staticmethod
    def _extract_text(element) -> str:
        """
        Extract all text contained inside an XML element.
        """

        if element is None:
            return ""

        return " ".join(
            text.strip()
            for text in element.itertext()
            if text.strip()
        )

    @staticmethod
    def _extract_year(article) -> int | None:
        """
        Extract publication year from a PubMed article.
        """

        year_element = article.find(
            ".//PubDate/Year"
        )

        if year_element is not None and year_element.text:
            try:
                return int(year_element.text)
            except ValueError:
                pass

        # Some PubMed records use MedlineDate instead of Year.
        medline_date = article.find(
            ".//PubDate/MedlineDate"
        )

        if medline_date is not None and medline_date.text:
            first_part = medline_date.text[:4]

            try:
                return int(first_part)
            except ValueError:
                pass

        return None

    def _parse_article(self, article) -> MedicalDocument | None:
        """
        Convert a single PubMedArticle XML element
        into a MedicalDocument.
        """

        pmid_element = article.find(".//PMID")

        if pmid_element is None or not pmid_element.text:
            return None

        pmid = pmid_element.text.strip()

        title_element = article.find(
            ".//ArticleTitle"
        )

        title = self._extract_text(title_element)

        if not title:
            title = "Untitled PubMed Article"

        abstract_elements = article.findall(
            ".//Abstract/AbstractText"
        )

        abstract_parts = []

        for abstract in abstract_elements:

            text = self._extract_text(abstract)

            if text:
                abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        if not abstract:
            abstract = "No abstract available."

        journal_element = article.find(
            ".//Journal/Title"
        )

        journal = self._extract_text(
            journal_element
        )

        publication_year = self._extract_year(article)

        content = f"{title}\n\n{abstract}"

        metadata = {
            "pmid": pmid,
            "article_id": pmid,
        }

        return MedicalDocument(
            id=f"pubmed_{pmid}",
            title=title,
            content=content,
            source="PubMed",
            journal=journal or None,
            publication_year=publication_year,
            metadata=metadata,
        )

    def load(self) -> List[MedicalDocument]:
        """
        Load all PubMed XML files in the dataset directory.
        """

        files = self._find_dataset_files()

        documents: List[MedicalDocument] = []

        for file_path in files:

            logger.info(
                f"Reading {file_path.name}"
            )

            try:

                with gzip.open(
                    file_path,
                    "rb"
                ) as compressed_file:

                    tree = ET.parse(compressed_file)

                root = tree.getroot()

            except (
                OSError,
                ET.ParseError
            ) as error:

                logger.error(
                    f"Failed to read {file_path.name}: {error}"
                )

                continue

            articles = root.findall(
                ".//PubmedArticle"
            )

            logger.info(
                f"Found {len(articles)} PubMed articles."
            )

            for article in articles:

                document = self._parse_article(
                    article
                )

                if document is not None:
                    documents.append(document)

        logger.success(
            f"Loaded {len(documents)} PubMed documents."
        )

        return documents