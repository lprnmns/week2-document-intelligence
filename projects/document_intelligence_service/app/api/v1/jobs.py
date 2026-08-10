"""Asynchronous ingestion job contract routes."""

from typing import Annotated

from fastapi import APIRouter, Path, Request

from ..errors import openapi_error_responses
from ...application.ingestion_service import IngestionService
from ...observability.request_id import get_request_id
from .contracts import JobResponse, StageEventResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={**openapi_error_responses()},
)
async def get_job(
    request: Request,
    job_id: Annotated[str, Path(min_length=1, max_length=128)],
) -> JobResponse:
    """Return asynchronous ingestion progress."""

    ingestion_service: IngestionService = request.app.state.ingestion_service
    snapshot = await ingestion_service.get_job(job_id)
    return JobResponse(
        job_id=snapshot.job_id,
        document_id=snapshot.document_id,
        status=snapshot.status,
        progress_percent=snapshot.progress_percent,
        error_code=snapshot.error_code,
        request_id=get_request_id(),
        current_stage=snapshot.current_stage,
        stages=[
            StageEventResponse(
                name=event.name,
                status=event.status,
                started_at=event.started_at,
                finished_at=event.finished_at,
                duration_ms=event.duration_ms,
                inputs=event.inputs or {},
                outputs=event.outputs or {},
                decision=event.decision,
                warnings=list(event.warnings),
                error_code=event.error_code,
                error_message=event.error_message,
            )
            for event in snapshot.stages
        ],
        page_count=snapshot.page_count,
        point_count=snapshot.point_count,
        error_message=snapshot.error_message,
        failed_stage=snapshot.failed_stage,
        attempt_count=snapshot.attempt_count,
        max_attempts=snapshot.max_attempts,
        next_attempt_at=snapshot.next_attempt_at,
        last_attempt_at=snapshot.last_attempt_at,
    )
