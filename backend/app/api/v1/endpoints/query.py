# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.db.models import User
from app.schemas.query import MedicalQueryRequest
from app.schemas.response import MedicalResponse
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/ask", response_model=MedicalResponse)
async def ask_medical_question(
    request: MedicalQueryRequest,
    _: User = Depends(get_current_user),
):
    return await QueryService().ask(request)
