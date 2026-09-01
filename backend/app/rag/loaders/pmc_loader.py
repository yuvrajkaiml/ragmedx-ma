from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

from app.utils.logger import logger
from app.rag.loaders.models import MedicalDocument


class PMCLoader:
    """
    Loads and parses PMC Open Access articles
    provided in BioC XML format.

    Extracts:
    - PMC document ID
    - Article title
    - Journal
    - Publication year
    - Article content
    - Additional metadata
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)

        logger.info(
            f"PMC Loader initialized: {self.dataset_path}"
        )

    # ================================================================
    # DATASET DISCOVERY
    # ================================================================

    def _find_dataset_files(self) -> List[Path]:
        """
        Find all PMC XML files in the dataset directory.
        """

        if not self.dataset_path.exists():
            logger.error(
                f"PMC dataset directory does not exist: "
                f"{self.dataset_path}"
            )
            return []

        if not self.dataset_path.is_dir():
            logger.error(
                f"PMC dataset path is not a directory: "
                f"{self.dataset_path}"
            )
            return []

        xml_files = sorted(
            self.dataset_path.glob("*.xml")
        )

        logger.info(
            f"Found {len(xml_files)} PMC XML file(s)."
        )

        return xml_files

    # ================================================================
    # XML HELPERS
    # ================================================================

    @staticmethod
    def _clean_text(text: Optional[str]) -> str:
        """
        Normalize extracted text.
        """

        if not text:
            return ""

        return " ".join(
            text.split()
        ).strip()

    @staticmethod
    def _get_infon_value(
        element: ET.Element,
        key: str
    ) -> Optional[str]:
        """
        Extract a value from a BioC infon.

        Example:

            <infons>
                <infon key="journal">Nat Rev Cancer</infon>
            </infons>
        """

        for infon in element.findall(".//infon"):

            if infon.get("key") == key:

                value = PMCLoader._clean_text(
                    infon.text
                )

                if value:
                    return value

        return None

    @staticmethod
    def _get_first_infon_value(
        element: ET.Element,
        keys: List[str]
    ) -> Optional[str]:
        """
        Try multiple possible infon keys and return
        the first non-empty value.
        """

        for key in keys:

            value = PMCLoader._get_infon_value(
                element,
                key
            )

            if value:
                return value

        return None

    # ================================================================
    # METADATA EXTRACTION
    # ================================================================

    def _extract_metadata(
        self,
        document: ET.Element
    ):
        """
        Extract article-level metadata.

        Returns:
            title,
            journal,
            publication_year,
            metadata
        """

        title = ""
        journal = None
        publication_year = None
        metadata = {}

        # ------------------------------------------------------------
        # Extract document-level infons
        # ------------------------------------------------------------

        for infon in document.findall(
            "./infons/infon"
        ):

            key = infon.get("key")
            value = self._clean_text(
                infon.text
            )

            if not key or not value:
                continue

            normalized_key = key.lower().strip()

            # --------------------------------------------------------
            # TITLE
            # --------------------------------------------------------

            if normalized_key in {
                "title",
                "article-title",
                "document-title"
            }:

                title = value

            # --------------------------------------------------------
            # JOURNAL
            # --------------------------------------------------------

            elif normalized_key in {
                "journal",
                "journal-title",
                "source"
            }:

                journal = value

            # --------------------------------------------------------
            # PUBLICATION YEAR
            # --------------------------------------------------------

            elif normalized_key in {
                "year",
                "publication-year",
                "pub_year",
                "date"
            }:

                try:

                    year_value = value[:4]

                    if year_value.isdigit():
                        publication_year = int(
                            year_value
                        )

                except (ValueError, TypeError):
                    publication_year = None

            # --------------------------------------------------------
            # OTHER METADATA
            # --------------------------------------------------------

            else:

                metadata[key] = value

        # ------------------------------------------------------------
        # FALLBACK TITLE EXTRACTION
        # ------------------------------------------------------------

        if not title:

            passages = document.findall(
                ".//passage"
            )

            for passage in passages:

                passage_text = self._clean_text(
                    passage.findtext("text")
                )

                if not passage_text:
                    continue

                passage_type = (
                    self._get_infon_value(
                        passage,
                        "type"
                    )
                )

                if passage_type:

                    normalized_type = (
                        passage_type.lower().strip()
                    )

                    if normalized_type in {
                        "title",
                        "article-title",
                        "document-title"
                    }:

                        title = passage_text
                        break

                # ----------------------------------------------------
                # Fallback for your PMC BioC structure:
                #
                # The first meaningful passage is the article title.
                # ----------------------------------------------------

                title = passage_text
                break

        # ------------------------------------------------------------
        # FALLBACK JOURNAL
        # ------------------------------------------------------------

        if not journal:

            journal = (
                self._get_first_infon_value(
                    document,
                    [
                        "journal",
                        "journal-title",
                        "source"
                    ]
                )
            )

        # ------------------------------------------------------------
        # FALLBACK PUBLICATION YEAR
        # ------------------------------------------------------------

        if publication_year is None:

            year_value = (
                self._get_first_infon_value(
                    document,
                    [
                        "year",
                        "publication-year",
                        "pub_year",
                        "date"
                    ]
                )
            )

            if year_value:

                try:

                    year_value = year_value.strip()

                    if len(year_value) >= 4:
                        year_value = year_value[:4]

                    if year_value.isdigit():

                        publication_year = int(
                            year_value
                        )

                except (ValueError, TypeError):

                    publication_year = None

        return (
            title,
            journal,
            publication_year,
            metadata
        )

    # ================================================================
    # CONTENT EXTRACTION
    # ================================================================

    def _extract_content(
        self,
        document: ET.Element
    ) -> str:
        """
        Extract article text from all meaningful BioC passages.

        Reference/bibliography passages are excluded where the
        BioC metadata identifies them as references.
        """

        passages = []

        for passage in document.findall(
            ".//passage"
        ):

            text = self._clean_text(
                passage.findtext("text")
            )

            if not text:
                continue

            # --------------------------------------------------------
            # Check passage type
            # --------------------------------------------------------

            passage_type = (
                self._get_infon_value(
                    passage,
                    "type"
                )
            )

            if passage_type:

                normalized_type = (
                    passage_type.lower().strip()
                )

                if normalized_type in {
                    "ref",
                    "reference",
                    "references",
                    "bibliography"
                }:

                    continue

            # --------------------------------------------------------
            # Check section type
            # --------------------------------------------------------

            section_type = (
                self._get_infon_value(
                    passage,
                    "section_type"
                )
            )

            if section_type:

                normalized_section = (
                    section_type.lower().strip()
                )

                if normalized_section in {
                    "ref",
                    "reference",
                    "references",
                    "bibliography"
                }:

                    continue

            passages.append(text)

        return "\n\n".join(
            passages
        )

    # ================================================================
    # SINGLE DOCUMENT PARSING
    # ================================================================

    def _parse_document(
        self,
        document: ET.Element,
        file_path: Path
    ) -> Optional[MedicalDocument]:
        """
        Parse one BioC document into a MedicalDocument.
        """

        # ------------------------------------------------------------
        # DOCUMENT ID
        # ------------------------------------------------------------

        document_id = document.findtext("id")

        if document_id:
            document_id = document_id.strip()

        if not document_id:

            logger.warning(
                f"Skipping document without ID "
                f"in {file_path.name}"
            )

            return None

        # ------------------------------------------------------------
        # METADATA
        # ------------------------------------------------------------

        (
            title,
            journal,
            publication_year,
            metadata
        ) = self._extract_metadata(
            document
        )

        # ------------------------------------------------------------
        # CONTENT
        # ------------------------------------------------------------

        content = self._extract_content(
            document
        )

        if not content:

            logger.warning(
                f"Skipping document without content: "
                f"{document_id}"
            )

            return None

        # ------------------------------------------------------------
        # ADD PMC ID TO METADATA
        # ------------------------------------------------------------

        metadata["pmc_id"] = document_id

        # ------------------------------------------------------------
        # CREATE MEDICAL DOCUMENT
        # ------------------------------------------------------------

        medical_document = MedicalDocument(
            id=f"pmc_{document_id}",
            title=title,
            content=content,
            source="PMC",
            journal=journal,
            publication_year=publication_year,
            metadata=metadata
        )

        logger.info(
            f"Parsed PMC document: "
            f"{medical_document.id} | "
            f"Title: "
            f"{title[:80] if title else '[missing]'}"
        )

        return medical_document

    # ================================================================
    # PUBLIC LOAD METHOD
    # ================================================================

    def load(self) -> List[MedicalDocument]:
        """
        Load all PMC BioC XML files and convert them into
        MedicalDocument objects.
        """

        files = self._find_dataset_files()

        logger.info(
            f"Preparing to load "
            f"{len(files)} PMC file(s)."
        )

        documents: List[MedicalDocument] = []

        for file_path in files:

            logger.info(
                f"Reading {file_path.name}"
            )

            try:

                tree = ET.parse(
                    file_path
                )

                root = tree.getroot()

                # ----------------------------------------------------
                # BioC normally uses:
                #
                # <collection>
                #     <document>
                #         ...
                #     </document>
                # </collection>
                #
                # .//document makes this more tolerant.
                # ----------------------------------------------------

                xml_documents = root.findall(
                    ".//document"
                )

                if not xml_documents:

                    logger.warning(
                        f"No BioC documents found "
                        f"in {file_path.name}"
                    )

                    continue

                logger.info(
                    f"Found {len(xml_documents)} "
                    f"document(s) in {file_path.name}"
                )

                for xml_document in xml_documents:

                    try:

                        medical_document = (
                            self._parse_document(
                                xml_document,
                                file_path
                            )
                        )

                        if medical_document:

                            documents.append(
                                medical_document
                            )

                    except Exception as e:

                        logger.error(
                            f"Error parsing document "
                            f"in {file_path.name}: {e}"
                        )

            except ET.ParseError as e:

                logger.error(
                    f"Failed to parse "
                    f"{file_path.name}: {e}"
                )

            except Exception as e:

                logger.error(
                    f"Error loading "
                    f"{file_path.name}: {e}"
                )

        logger.success(
            f"Loaded {len(documents)} PMC documents."
        )

        return documents