"""Evaluation run lifecycle routes."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query, Request, status

from ..errors import openapi_error_responses
from ...application.evaluation_service import EvaluationService, EvaluationSpec
from ...domain.evaluation import EvaluationRunSnapshot
from .contracts import (
    EvaluationRunListResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post(
    "/runs",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"model": EvaluationRunResponse},
        **openapi_error_responses(),
    },
)
async def create_evaluation_run(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: EvaluationRunRequest,
) -> EvaluationRunResponse:
    """Queue a golden-set evaluation and return its run ID."""

    service: EvaluationService = request.app.state.evaluation_service
    run = await service.create_run(
        EvaluationSpec(
            evaluation_type=payload.evaluation_type,
            dataset=payload.dataset,
            split=payload.split,
            mode=payload.mode,
            top_k=payload.top_k,
            reranker_enabled=payload.reranker_enabled,
        )
    )
    background_tasks.add_task(service.execute_run, run.run_id)
    return _response(run)


@router.get(
    "/config",
    responses={**openapi_error_responses()},
)
async def get_evaluation_configuration(request: Request) -> dict[str, object]:
    """Return the active corpus and reproducibility context for the UI."""

    service: EvaluationService = request.app.state.evaluation_service
    return service.default_configuration()


@router.get(
    "/runs",
    response_model=EvaluationRunListResponse,
    responses={**openapi_error_responses()},
)
async def list_evaluation_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> EvaluationRunListResponse:
    """Return the newest bounded evaluation run states."""

    service: EvaluationService = request.app.state.evaluation_service
    runs = await service.list_runs(limit)
    return EvaluationRunListResponse(items=[_response(run) for run in runs])


@router.get(
    "/runs/{run_id}",
    response_model=EvaluationRunResponse,
    responses={**openapi_error_responses()},
)
async def get_evaluation_run(request: Request, run_id: str) -> EvaluationRunResponse:
    """Return one evaluation state or a stable not-found response."""

    service: EvaluationService = request.app.state.evaluation_service
    return _response(await service.get_run(run_id))


def _response(run: EvaluationRunSnapshot) -> EvaluationRunResponse:
    """Map the domain snapshot to the versioned HTTP response."""

    return EvaluationRunResponse(
        run_id=run.run_id,
        status=run.status,
        evaluation_type=run.evaluation_type,
        dataset=run.dataset,
        split=run.split,
        mode=run.mode,
        top_k=run.top_k,
        reranker_enabled=run.reranker_enabled,
        requested_at=run.requested_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        case_count=run.case_count,
        metrics=run.metrics,
        artifact_path=run.artifact_path,
        git_sha=run.git_sha,
        error_code=run.error_code,
        error_message=run.error_message,
        configuration=run.configuration or {},
    )
