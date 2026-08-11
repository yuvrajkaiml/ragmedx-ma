# Owner: Pranav Sukale — not yet implemented, see Task_Assignment.md
from time import perf_counter

from app.schemas.query import MedicalQueryRequest
from app.schemas.response import AbstentionResponse, MedicalResponse


class QueryService:
    async def ask(self, request: MedicalQueryRequest) -> MedicalResponse:
        started_at = perf_counter()

        return AbstentionResponse(
            message="The medical-answering pipeline is not available yet.",
            abstention_reason=(
                "Evidence retrieval and peer-reviewed generation have not "
                "yet been integrated."
            ),
            suggested_followup=(
                "Please try again after the retrieval and agent modules are connected."
            ),
            processing_time_ms=int((perf_counter() - started_at) * 1000),
        )