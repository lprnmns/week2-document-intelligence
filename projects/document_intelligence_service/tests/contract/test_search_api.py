"""Contract tests for the evidence-only search endpoint."""

import asyncio
from collections.abc import Sequence

import httpx
from fastapi import FastAPI

from projects.document_intelligence_service.app.application.health_service import (
    HealthService,
)
from projects.document_intelligence_service.app.domain.entities import RetrievalMode
from projects.document_intelligence_service.app.domain.retrieval import RetrievalResult
from projects.document_intelligence_service.app.main import create_app
from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
    make_service,
)


class RecordingRetrievalService:
    """Capture the resolved request scope while returning fixture evidence."""

    def __init__(self) -> None:
        self.tenant_id: str | None = None
        self.acl_tags: tuple[str, ...] | None = None

    def search(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[str] = (),
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> RetrievalResult:
        from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
            make_service,
        )

        self.tenant_id = tenant_id
        self.acl_tags = tuple(acl_tags)
        return make_service().search(
            question=question,
            mode=mode,
            top_k=top_k,
            document_ids=document_ids,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
        )


async def post_search(app: FastAPI, payload: object) -> httpx.Response:
    """Post one search through the real request-id middleware and lifespan."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-Request-ID": "search-contract-1"},
        ) as client:
            return await client.post("/v1/search", json=payload)


def test_search_returns_evidence_without_llm_fields() -> None:
    app = create_app(
        health_service=HealthService(()),
        retrieval_service=make_service(),
    )

    response = asyncio.run(
        post_search(
            app,
            {
                "question": "Qdrant ne işe yarar?",
                "retrieval_mode": "hybrid",
                "top_k": 3,
            },
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "search-contract-1"
    assert body["retrieval"] == {
        "mode": "hybrid",
        "dense_candidates": 2,
        "sparse_candidates": 2,
        "rrf_candidates": 3,
        "reranked_candidates": 0,
    }
    assert body["sources"][0]["source_id"] == "shared"
    assert body["latency"]["llm_ms"] == 0


def test_search_accepts_acl_scope_without_disabling_filtering() -> None:
    app = create_app(
        health_service=HealthService(()),
        retrieval_service=make_service(),
    )

    response = asyncio.run(
        post_search(
            app,
            {
                "question": "Qdrant ne işe yarar?",
                "acl_tags": ["finance"],
            },
        )
    )

    assert response.status_code == 200
    # The fixture is public, so a caller with an additional finance scope may
    # still see it. The request must reach the retrieval service rather than
    # silently falling back to the old FEATURE_NOT_READY scaffold.
    assert response.json()["sources"][0]["source_id"] == "shared"


def test_search_resolves_tenant_and_acl_headers() -> None:
    retrieval = RecordingRetrievalService()
    app = create_app(
        health_service=HealthService(()),
        retrieval_service=retrieval,  # type: ignore[arg-type]
    )

    response = asyncio.run(
        post_search_with_headers(
            app,
            {"question": "Qdrant ne işe yarar?"},
            {"X-Tenant-ID": "tenant-a", "X-ACL-Tags": "finance,public"},
        )
    )

    assert response.status_code == 200
    assert retrieval.tenant_id == "tenant-a"
    assert retrieval.acl_tags == ("finance", "public")


def test_search_rejects_tenant_header_body_mismatch() -> None:
    app = create_app(
        health_service=HealthService(()),
        retrieval_service=make_service(),
    )

    response = asyncio.run(
        post_search_with_headers(
            app,
            {"question": "Qdrant ne işe yarar?", "tenant_id": "tenant-a"},
            {"X-Tenant-ID": "tenant-b"},
        )
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


async def post_search_with_headers(
    app: FastAPI,
    payload: object,
    headers: dict[str, str],
) -> httpx.Response:
    """Post search with explicit scope headers."""

    transport = httpx.ASGITransport(app=app)
    request_headers = {"X-Request-ID": "search-scope-contract-1", **headers}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=request_headers,
        ) as client:
            return await client.post("/v1/search", json=payload)
