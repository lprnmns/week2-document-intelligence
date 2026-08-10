"""Versioned health endpoints."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...application.health_service import HealthService

router = APIRouter(prefix="/health", tags=["health"])


class DependencyCheckResponse(BaseModel):
    """Public, non-sensitive dependency health information."""

    status: Literal["up", "down"]
    latency_ms: float = Field(ge=0)
    detail: str | None = None


class LiveHealthResponse(BaseModel):
    """Liveness response contract."""

    status: Literal["alive"]


class StartupHealthResponse(BaseModel):
    """Startup response contract."""

    status: Literal["started", "starting"]


class ReadyHealthResponse(BaseModel):
    """Readiness response contract."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, DependencyCheckResponse]


def get_health_service(request: Request) -> HealthService:
    """Resolve the application service from the composition root."""

    service: HealthService = request.app.state.health_service
    return service


HealthServiceDependency = Annotated[HealthService, Depends(get_health_service)]


@router.get("/live", response_model=LiveHealthResponse)
async def live() -> LiveHealthResponse:
    """Report process liveness without calling external dependencies."""

    return LiveHealthResponse(status="alive")


@router.get(
    "/startup",
    response_model=StartupHealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": StartupHealthResponse}},
)
async def startup(health_service: HealthServiceDependency) -> StartupHealthResponse | JSONResponse:
    """Report whether application startup wiring has completed."""

    payload = StartupHealthResponse(
        status="started" if health_service.startup_complete else "starting"
    )
    if health_service.startup_complete:
        return payload
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )


@router.get(
    "/ready",
    response_model=ReadyHealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadyHealthResponse}},
)
async def ready(health_service: HealthServiceDependency) -> ReadyHealthResponse | JSONResponse:
    """Report whether all required dependencies can serve traffic."""

    report = await health_service.readiness()
    payload = ReadyHealthResponse(
        status="ready" if report.ready else "not_ready",
        checks={
            check.name: DependencyCheckResponse(
                status=check.state.value,
                latency_ms=check.latency_ms,
                detail=check.detail,
            )
            for check in report.checks
        },
    )
    if report.ready:
        return payload
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )
