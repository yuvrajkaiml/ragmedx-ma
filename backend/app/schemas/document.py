from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    source: str
    page: int


class Document(BaseModel):
    text: str
    metadata: DocumentMetadata