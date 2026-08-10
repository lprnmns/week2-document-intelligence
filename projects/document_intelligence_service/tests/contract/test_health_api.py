"""Contract tests for versioned health endpoints."""

import asyncio

import httpx
from fastapi import FastAPI

from projects.document_intelligence_service.app.application.health_service import (
    HealthService,
)
from projects.document_intelligence_service.app.domain.errors import (
    ErrorCode,
    ServiceError,
)
from projects.document_intelligence_service.app.domain.health import (
    DependencyHealth,
    DependencyState,
)
from projects.document_intelligence_service.app.main import create_app


class FakeProbe:
    """Probe with an observable call count for contract tests."""

    def __init__(self, result: DependencyHealth) -> None:
        self.result = result
        self.call_count = 0

    async def check(self) -> DependencyHealth:
        self.call_count += 1
        return self.result


async def request(app: FastAPI, path: str) -> httpx.Response:
    """Call the ASGI app while running its real lifespan hooks."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)


def test_liveness_returns_200_without_calling_dependencies() -> None:
    probe = FakeProbe(DependencyHealth("qdrant", DependencyState.DOWN, 0.0))
    app = create_app(health_service=HealthService((probe,)))

    response = asyncio.run(request(app, "/v1/health/live"))

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert probe.call_count == 0


def test_request_id_is_preserved_and_returned_in_response_header() -> None:
    app = create_app(health_service=HealthService(()))

    response = asyncio.run(
        request_with_headers(app, "/v1/health/live", {"X-Request-ID": "mentor-42"})
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "mentor-42"


def test_invalid_request_id_is_replaced_with_generated_id() -> None:
    app = create_app(health_service=HealthService(()))

    response = asyncio.run(
        request_with_headers(app, "/v1/health/live", {"X-Request-ID": "bad id with spaces"})
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"].startswith("req_")


def test_service_error_returns_safe_error_envelope() -> None:
    app = create_app(health_service=HealthService(()))

    async def raise_service_error() -> None:
        raise ServiceError(
            code=ErrorCode.DEPENDENCY_UNAVAILABLE,
            message="Required service is temporarily unavailable",
        )

    app.add_api_route("/v1/test-error", raise_service_error)
    response = asyncio.run(
        request_with_headers(app, "/v1/test-error", {"X-Request-ID": "error-7"})
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "DEPENDENCY_UNAVAILABLE",
            "message": "Required service is temporarily unavailable",
            "request_id": "error-7",
        }
    }
    assert response.headers["X-Request-ID"] == "error-7"


def test_validation_error_returns_generic_safe_message() -> None:
    app = create_app(health_service=HealthService(()))

    async def requires_integer(limit: int) -> dict[str, int]:
        return {"limit": limit}

    app.add_api_route("/v1/test-validation", requires_integer)
    response = asyncio.run(
        request_with_headers(
            app,
            "/v1/test-validation?limit=not-an-integer",
            {"X-Request-ID": "validation-3"},
        )
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_REQUEST",
            "message": "Request validation failed",
            "request_id": "validation-3",
        }
    }


def test_readiness_returns_503_with_safe_dependency_details() -> None:
    qdrant = FakeProbe(
        DependencyHealth(
            "qdrant",
            DependencyState.DOWN,
            3.5,
            detail="dependency unavailable",
        )
    )
    ollama = FakeProbe(DependencyHealth("ollama", DependencyState.UP, 4.0))
    app = create_app(health_service=HealthService((qdrant, ollama)))

    response = asyncio.run(request(app, "/v1/health/ready"))

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "qdrant": {
                "status": "down",
                "latency_ms": 3.5,
                "detail": "dependency unavailable",
            },
            "ollama": {"status": "up", "latency_ms": 4.0, "detail": None},
        },
    }


async def request_with_headers(
    app: FastAPI,
    path: str,
    headers: dict[str, str],
) -> httpx.Response:
    """Call the ASGI app with explicit request headers."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=headers,
        ) as client:
            return await client.get(path)


def test_startup_is_ready_inside_application_lifespan() -> None:
    app = create_app(health_service=HealthService(()))

    response = asyncio.run(request(app, "/v1/health/startup"))

    assert response.status_code == 200
    assert response.json() == {"status": "started"}


def test_rest_contract_paths_are_published_in_openapi() -> None:
    app = create_app(health_service=HealthService(()))

    assert sorted(app.openapi()["paths"]) == [
            "/v1/documents",
            "/v1/documents/{document_id}",
            "/v1/evaluations/config",
            "/v1/evaluations/runs",
            "/v1/evaluations/runs/{run_id}",
            "/v1/health/live",
        "/v1/health/ready",
            "/v1/health/startup",
            "/v1/jobs/{job_id}",
            "/v1/metrics",
            "/v1/queries",
            "/v1/query",
            "/v1/search",
    ]
