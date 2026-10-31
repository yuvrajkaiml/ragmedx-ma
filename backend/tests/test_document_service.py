"""Lifecycle tests for :class:`DocumentService` (upload → status → delete).

The heavyweight FAISS build step is stubbed (``index_service.build_index``
monkeypatched) so these tests prove the status state machine and the metadata
sidecar behave correctly without loading an embedder. Everything is written
under a pytest ``tmp_path``; nothing touches the repo's real ``data/uploads``.

asyncio_mode = auto (pytest.ini) runs ``async def test_*`` without a decorator.
This file does not import ``app.main``, so no DEBUG env is required.
"""

from __future__ import annotations

import io
import json
from uuid import uuid4

import pytest
from starlette.datastructures import Headers, UploadFile

from app.schemas.document import DocumentStatus
from app.services.document_service import DocumentService

_PDF_BYTES = b"%PDF-1.4 minimal placeholder body for a fake upload\n%%EOF"


@pytest.fixture
def document_service(tmp_path, monkeypatch) -> DocumentService:
    """DocumentService pinned to a throwaway uploads dir."""
    from app.config import get_settings

    # Patch the Settings singleton so the service resolves uploads to tmp_path
    # (an absolute path, so DocumentService keeps it as-is) instead of the
    # repo's real data/uploads directory.
    monkeypatch.setattr(get_settings(), "upload_data_dir", tmp_path, raising=False)
    service = DocumentService()
    # Sanity: the service resolved to our tmp dir, not the repo data dir.
    assert service.upload_dir == tmp_path
    return service


def _upload(content_type: str = "application/pdf", filename: str = "report.pdf") -> UploadFile:
    headers = Headers({"content-type": content_type})
    return UploadFile(
        file=io.BytesIO(_PDF_BYTES),
        filename=filename,
        headers=headers,
    )


async def test_save_pdf_rejects_non_pdf(document_service):
    upload = _upload(content_type="text/plain", filename="notes.txt")

    with pytest.raises(ValueError):
        await document_service.save_pdf(upload, uuid4())


async def test_save_pdf_writes_queued_sidecar(document_service):
    user_id = uuid4()

    response = await document_service.save_pdf(_upload(), user_id)

    assert response.status == DocumentStatus.QUEUED
    sidecar = document_service._metadata_path(response.document_id)
    assert sidecar.exists()
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["status"] == DocumentStatus.QUEUED.value
    assert metadata["uploaded_by"] == str(user_id)
    assert metadata["filename"] == "report.pdf"
    assert metadata["chunk_count"] is None
    # The PDF bytes were persisted alongside the sidecar.
    assert document_service.index_service.pdf_path(response.document_id).exists()


async def test_process_document_drives_to_indexed(document_service, monkeypatch):
    user_id = uuid4()
    response = await document_service.save_pdf(_upload(), user_id)

    monkeypatch.setattr(
        document_service.index_service, "build_index", lambda *a, **k: 7
    )

    document_service.process_document(response.document_id)

    status = document_service.get_status(response.document_id, user_id)
    assert status.status == DocumentStatus.INDEXED
    assert status.chunk_count == 7
    assert status.error_message is None


async def test_process_document_records_failure_without_raising(document_service, monkeypatch):
    user_id = uuid4()
    response = await document_service.save_pdf(_upload(), user_id)

    def _boom(*a, **k):
        raise ValueError("The PDF contains no extractable text.")

    monkeypatch.setattr(document_service.index_service, "build_index", _boom)

    # Must not raise — failures are captured in the document status.
    document_service.process_document(response.document_id)

    status = document_service.get_status(response.document_id, user_id)
    assert status.status == DocumentStatus.FAILED
    assert "no extractable text" in status.error_message


async def test_get_status_scoped_to_owner(document_service):
    owner = uuid4()
    response = await document_service.save_pdf(_upload(), owner)

    assert document_service.get_status(response.document_id, owner) is not None
    assert document_service.get_status(response.document_id, uuid4()) is None


async def test_delete_document_removes_artifacts_and_calls_delete_index(document_service, monkeypatch):
    user_id = uuid4()
    response = await document_service.save_pdf(_upload(), user_id)
    document_id = response.document_id

    deleted = []
    monkeypatch.setattr(
        document_service.index_service,
        "delete_index",
        lambda doc_id: deleted.append(doc_id),
    )

    assert document_service.delete_document(document_id, user_id) is True

    assert not document_service._metadata_path(document_id).exists()
    assert not document_service.index_service.pdf_path(document_id).exists()
    assert deleted == [document_id]


async def test_delete_document_rejects_non_owner(document_service):
    response = await document_service.save_pdf(_upload(), uuid4())

    assert document_service.delete_document(response.document_id, uuid4()) is False
    # Sidecar is untouched for the real owner.
    assert document_service._metadata_path(response.document_id).exists()
