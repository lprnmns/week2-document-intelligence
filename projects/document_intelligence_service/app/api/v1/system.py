"""System profile and controlled local-model operations for the Demo UI."""

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

from ...application.model_service import ModelService
from ...domain.errors import ErrorCode, ServiceError
from ..errors import openapi_error_responses

router = APIRouter(prefix="/system", tags=["system"], include_in_schema=False)


class ModelPullRequest(BaseModel):
    """Only a model identifier is accepted; no command/shell field exists."""

    model_id: str = Field(min_length=1, max_length=128)


class ModelPullStartResponse(BaseModel):
    pull_id: str
    model_id: str
    status: str


@dataclass(slots=True)
class _PullState:
    pull_id: str
    model_id: str
    status: str = "queued"
    progress_percent: int | None = None
    runtime_status: str | None = None
    error: str | None = None
    details: dict[str, object] = field(default_factory=dict)


@router.get("/profile", responses={**openapi_error_responses()})
async def get_profile(request: Request) -> dict[str, object]:
    """Return sanitized hardware, runtime and model readiness information."""

    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="System profile is not wired",
        )
    settings = getattr(request.app.state, "settings", None)
    if settings is not None and not settings.system_profile_enabled:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="System profile is disabled",
        )
    snapshot = await service.snapshot()
    snapshot["model_management_enabled"] = bool(
        settings is not None and settings.local_model_management_enabled
    )
    return snapshot


@router.post(
    "/models/pulls",
    response_model=ModelPullStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"model": ModelPullStartResponse},
        **openapi_error_responses(),
    },
)
async def pull_model(
    request: Request,
    payload: ModelPullRequest,
) -> ModelPullStartResponse:
    """Start an allow-listed local runtime pull when explicitly enabled."""

    settings = getattr(request.app.state, "settings", None)
    if settings is None or not settings.local_model_management_enabled:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Local model management is disabled",
        )
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Model runtime is not wired",
        )
    validation = await service.validate_model_pull(payload.model_id)
    if validation != "ready":
        raise ServiceError(
            code=(
                ErrorCode.INVALID_REQUEST
                if validation == "not_allowlisted"
                else ErrorCode.DEPENDENCY_UNAVAILABLE
            ),
            message=(
                "Selected model is not in the configured local catalog"
                if validation == "not_allowlisted"
                else "Selected model runtime is unavailable"
            ),
            stage="model_pull",
            reason=validation,
        )
    pull_store: dict[str, _PullState] = request.app.state.model_pull_store
    pull_id = f"pull_{uuid4().hex}"
    state = _PullState(pull_id=pull_id, model_id=payload.model_id)
    pull_store[pull_id] = state
    asyncio.create_task(_run_pull(service, state))
    return ModelPullStartResponse(
        pull_id=pull_id,
        model_id=payload.model_id,
        status=state.status,
    )


@router.get("/models/pulls/{pull_id}", responses={**openapi_error_responses()})
async def get_pull(request: Request, pull_id: str) -> dict[str, object]:
    """Return actual runtime pull progress, never simulated progress."""

    state: _PullState | None = request.app.state.model_pull_store.get(pull_id)
    if state is None:
        raise ServiceError(
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message="Model pull was not found or expired",
        )
    return {
        "pull_id": state.pull_id,
        "model_id": state.model_id,
        "status": state.status,
        "progress_percent": state.progress_percent,
        "runtime_status": state.runtime_status,
        "error": state.error,
        "details": state.details,
    }


async def _run_pull(service: ModelService, state: _PullState) -> None:
    state.status = "pulling"

    async def on_progress(payload: dict[str, object]) -> None:
        state.runtime_status = str(payload.get("status")) if payload.get("status") else None
        total = payload.get("total")
        completed = payload.get("completed")
        if isinstance(total, (int, float)) and isinstance(completed, (int, float)) and total > 0:
            state.progress_percent = max(0, min(100, round(completed / total * 100)))
        state.details = {
            "status": state.runtime_status,
            "total": total if isinstance(total, (int, float)) else None,
            "completed": completed if isinstance(completed, (int, float)) else None,
        }

    try:
        await service.pull_model(state.model_id, on_progress=on_progress)
    except ValueError as error:
        state.status = "failed"
        state.error = str(error)
    except Exception:
        state.status = "failed"
        state.error = "Local model pull failed"
    else:
        state.status = "completed"
        state.progress_percent = 100
