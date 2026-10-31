# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_current_user
from app.db.models import User
from app.schemas.document import DocumentStatusResponse, DocumentUploadResponse
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    service = DocumentService()
    try:
        response = await service.save_pdf(file, current_user.id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    # Parse -> chunk -> embed -> index happens off the request path so the
    # upload returns immediately; status moves QUEUED -> PROCESSING -> INDEXED.
    background_tasks.add_task(service.process_document, response.document_id)
    return response


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: UUID,
        current_user: User = Depends(get_current_user),
):
    document = DocumentService().get_status(document_id, current_user.id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
):
    deleted = DocumentService().delete_document(document_id, current_user.id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )    