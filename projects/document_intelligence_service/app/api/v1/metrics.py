"""Privacy-safe operational metrics endpoint."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ...observability.metrics import MetricsRegistry

router = APIRouter(prefix="/metrics", tags=["observability"])


class MetricsResponse(BaseModel):
    """Counters and latency summaries; raw query text is never included."""

    metrics: dict[str, object]


@router.get("", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    """Return in-process metrics for local debugging and demo evidence."""

    registry: MetricsRegistry = request.app.state.metrics
    return MetricsResponse(metrics=registry.snapshot())
