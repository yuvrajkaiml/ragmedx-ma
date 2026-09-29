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
from app.services.uploaded_document_index import UploadedDocumentIndexService
from app.utils.logger import logger


class DocumentService:
    chunk_size = 1024 * 1024

    def __init__(self):
        settings = get_settings()
        upload_dir = Path(settings.upload_data_dir)
        if not upload_dir.is_absolute():
            from app.config import BASE_DIR

            upload_dir = BASE_DIR / upload_dir
        self.upload_dir = upload_dir
        self.max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_service = UploadedDocumentIndexService(uploads_dir=self.upload_dir)

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
            "status": DocumentStatus.QUEUED.value,
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
            status=DocumentStatus.QUEUED,
            uploaded_at=uploaded_at,
            message=(
                "PDF uploaded successfully and queued for indexing. "
                "Poll GET /documents/{document_id}/status until it reports 'indexed'."
            ),
        )

    def process_document(self, document_id: UUID) -> None:
        """Background task: build the isolated FAISS index for one upload.

        Drives the status lifecycle QUEUED -> PROCESSING -> INDEXED | FAILED and
        records ``chunk_count`` / ``error_message`` in the metadata sidecar.
        Never raises — failures are captured in the document status so the
        background task cannot crash the worker.
        """
        metadata = self._read_metadata(document_id)
        if metadata is None:
            logger.warning(f"process_document: metadata missing for {document_id}")
            return

        filename = metadata.get("filename")
        self._update_metadata(document_id, status=DocumentStatus.PROCESSING.value)
        try:
            chunk_count = self.index_service.build_index(document_id, filename=filename)
            self._update_metadata(
                document_id,
                status=DocumentStatus.INDEXED.value,
                chunk_count=chunk_count,
                error_message=None,
            )
            logger.info(f"Document {document_id} indexed ({chunk_count} chunks).")
        except Exception as exc:  # noqa: BLE001 - status must capture all failures
            logger.exception(f"Failed to index document {document_id}: {exc}")
            self._update_metadata(
                document_id,
                status=DocumentStatus.FAILED.value,
                error_message=str(exc),
            )

    def _read_metadata(self, document_id: UUID) -> dict | None:
        metadata_path = self._metadata_path(document_id)
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _update_metadata(self, document_id: UUID, **changes) -> None:
        metadata = self._read_metadata(document_id)
        if metadata is None:
            return
        metadata.update(changes)
        self._metadata_path(document_id).write_text(
            json.dumps(metadata),
            encoding="utf-8",
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
        # Remove the isolated FAISS index directory, if one was built.
        self.index_service.delete_index(document_id)
        return True

    def _metadata_path(self, document_id: UUID) -> Path:
        return self.upload_dir / f"{document_id}.json"