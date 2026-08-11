import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from app.config import get_settings
from app.schemas.document import (
    DocumentStatus,
    DocumentStatusResponse,
    DocumentUploadResponse,
)


class DocumentService:
    chunk_size = 1024 * 1024

    def __init__(self):
        settings = get_settings()
        self.upload_dir = settings.upload_data_dir
        self.max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_pdf(
        self,
        upload: UploadFile,
        user_id: UUID,
    ) -> DocumentUploadResponse:
        if upload.content_type not in {"application/pdf", "application/x-pdf"}:
            raise ValueError("Only PDF files are allowed")

        document_id = uuid4()
        filename = Path(upload.filename or "document.pdf").name
        file_path = self.upload_dir / f"{document_id}.pdf"
        uploaded_at = datetime.now(timezone.utc)
        total_size = 0

        try:
            with file_path.open("wb") as destination:
                while chunk := await upload.read(self.chunk_size):
                    total_size += len(chunk)

                    if total_size > self.max_upload_bytes:
                        raise ValueError(
                            f"PDF must be at most {get_settings().max_upload_size_mb} MB"
                        )

                    destination.write(chunk)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        metadata = {
            "document_id": str(document_id),
            "uploaded_by": str(user_id),
            "filename": filename,
            "status": DocumentStatus.UPLOADED.value,
            "uploaded_at": uploaded_at.isoformat(),
            "error_message": None,
            "chunk_count": None,
        }
        self._metadata_path(document_id).write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            status=DocumentStatus.UPLOADED,
            uploaded_at=uploaded_at,
            message="PDF uploaded successfully and is ready for retrieval processing.",
        )

    def get_status(
        self,
        document_id: UUID,
        user_id: UUID,
    ) -> DocumentStatusResponse | None:
        metadata_path = self._metadata_path(document_id)

        if not metadata_path.exists():
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if metadata.get("uploaded_by") != str(user_id):
            return None

        return DocumentStatusResponse(
            document_id=UUID(metadata["document_id"]),
            filename=metadata["filename"],
            status=DocumentStatus(metadata["status"]),
            uploaded_at=datetime.fromisoformat(metadata["uploaded_at"]),
            error_message=metadata["error_message"],
            chunk_count=metadata["chunk_count"],
        )

    def delete_document(
        self,
        document_id: UUID,
        user_id: UUID,
    ) -> bool:
        metadata_path = self._metadata_path(document_id)

        if not metadata_path.exists():
            return False

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if metadata.get("uploaded_by") != str(user_id):
            return False

        metadata_path.unlink(missing_ok=True)
        (self.upload_dir / f"{document_id}.pdf").unlink(missing_ok=True)
        return True

    def _metadata_path(self, document_id: UUID) -> Path:
        return self.upload_dir / f"{document_id}.json"