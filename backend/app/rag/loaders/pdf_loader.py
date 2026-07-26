from pathlib import Path

from pypdf import PdfReader

from app.schemas.document import Document, DocumentMetadata


class PDFLoader:
    """
    Loads PDF documents page by page.
    """

    def load(self, pdf_path: str | Path) -> list[Document]:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text:
                continue

            documents.append(
                Document(
                    text=text.strip(),
                    metadata=DocumentMetadata(
                        source=pdf_path.name,
                        page=page_number,
                    ),
                )
            )

        return documents