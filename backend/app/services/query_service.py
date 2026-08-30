# Owner: Pranav Sukale
# Query service: API-facing service layer for the medical QA pipeline.

from time import perf_counter

from app.schemas.query import MedicalQueryRequest
from app.schemas.response import AbstentionResponse, MedicalResponse


class QueryService:
    """Service layer for processing medical queries."""

    async def ask(self, request: MedicalQueryRequest) -> MedicalResponse:
        started_at = perf_counter()

        # Temporary response until the agent orchestration pipeline
        # is integrated.
        return AbstentionResponse(
            message="The medical-answering pipeline is not available yet.",
            abstention_reason=(
                "Evidence retrieval and peer-reviewed generation have not "
                "yet been integrated."
            ),
            suggested_followup=(
                "Please try again after the retrieval and agent modules are connected."
            ),
            processing_time_ms=int(
                (perf_counter() - started_at) * 1000
            ),
        )