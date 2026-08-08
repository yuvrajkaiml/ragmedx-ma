from dataclasses import dataclass
from typing import Optional


@dataclass
class MedicalDocument:
    """
    Standard internal representation of a medical document/question.
    Every loader (MedQA, BioASQ, PubMed, PMC) should return this format.
    """

    id: str
    title: str
    content: str

    source: str

    journal: Optional[str] = None
    publication_year: Optional[int] = None
    metadata: Optional[dict] = None