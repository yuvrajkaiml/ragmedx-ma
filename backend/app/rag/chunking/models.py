from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class MedicalChunk(BaseModel):
    """
    Represents a retrieval-ready chunk of a medical document.
    """

    id: str = Field(..., description="Unique chunk identifier")

    document_id: str = Field(
        ...,
        description="ID of the source medical document"
    )

    content: str = Field(
        ...,
        description="Text content of the chunk"
    )

    title: Optional[str] = None

    section: Optional[str] = None

    source: str = "PMC"

    journal: Optional[str] = None

    publication_year: Optional[int] = None

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )