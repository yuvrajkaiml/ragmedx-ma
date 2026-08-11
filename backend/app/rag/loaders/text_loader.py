from pathlib import Path

from app.schemas.document import Document, DocumentMetadata


class TextLoader:

    def load(self, path: str | Path) -> list[Document]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        return [
            Document(
                text=path.read_text(encoding="utf-8"),
                metadata=DocumentMetadata(
                    source=path.name,
                    page=1,
                ),
            )
        ]