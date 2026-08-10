"""Contract tests for resource, job and query routes."""

import asyncio
from io import BytesIO

import httpx
from fastapi import FastAPI
from pypdf import PdfWriter

from projects.document_intelligence_service.app.application.health_service import (
    HealthService,
)
from projects.document_intelligence_service.app.main import create_app


async def post_json(app: FastAPI, path: str, payload: object) -> httpx.Response:
    """POST JSON through the real ASGI lifespan and middleware."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-Request-ID": "contract-1"},
        ) as client:
            return await client.post(path, json=payload)


async def post_json_with_headers(
    app: FastAPI,
    path: str,
    payload: object,
    headers: dict[str, str],
) -> httpx.Response:
    """POST JSON with explicit transport scope headers."""

    transport = httpx.ASGITransport(app=app)
    request_headers = {"X-Request-ID": "contract-scope-1", **headers}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=request_headers,
        ) as client:
            return await client.post(path, json=payload)


def pdf_bytes(page_count: int = 1) -> bytes:
    """Create a small valid PDF upload fixture."""

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def post_multipart(
    app: FastAPI,
    path: str,
    content: bytes,
    idempotency_key: str | None = None,
) -> httpx.Response:
    """POST one PDF through the real upload adapter."""

    transport = httpx.ASGITransport(app=app)
    headers = {"X-Request-ID": "upload-contract-1"}
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=headers,
        ) as client:
            return await client.post(
                path,
                files={"file": ("guide.pdf", content, "application/pdf")},
            )


async def get(app: FastAPI, path: str) -> httpx.Response:
    """GET one resource through the real ASGI lifespan."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-Request-ID": "job-contract-1"},
        ) as client:
            return await client.get(path)


def test_valid_query_does_not_fabricate_an_answer_before_wiring() -> None:
    app = create_app(health_service=HealthService(()))

    response = asyncio.run(
        post_json(
            app,
            "/v1/query",
            {"question": "Qdrant ne işe yarar?"},
        )
    )

    assert response.status_code == 501
    assert response.json() == {
        "error": {
            "code": "FEATURE_NOT_READY",
            "message": "Query workflow is not wired yet",
            "request_id": "contract-1",
        }
    }


def test_wired_query_returns_structured_no_answer_and_skips_llm() -> None:
    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        make_service,
    )

    generator = FakeAnswerGenerator()
    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(),
            answerability=AnswerabilityPolicy(min_dense_score=0.99),
            answer_generator=generator,
        ),
    )

    response = asyncio.run(
        post_json(
            app,
            "/v1/query",
            {"question": "Stajyer maaşı ne kadar?"},
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "no_answer"
    assert body["no_answer_reason"] == "LOW_RELEVANCE"
    assert body["no_answer"]["reason_code"] == "LOW_RELEVANCE"
    assert "atlandı" in body["no_answer"]["message"]
    assert body["no_answer"]["searched_document_ids"] == []
    assert body["model"] == {"provider": None, "model": None}
    assert body["latency"]["llm_ms"] == 0
    assert generator.call_count == 0


def test_query_plural_contract_path_returns_same_no_answer_shape() -> None:
    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        make_service,
    )

    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(),
            answerability=AnswerabilityPolicy(min_dense_score=0.99),
            answer_generator=FakeAnswerGenerator(),
        ),
    )

    response = asyncio.run(
        post_json(
            app,
            "/v1/queries",
            {"question": "Stajyer maaşı ne kadar?"},
        )
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "no_answer"
    assert response.json()["no_answer"]["reason_code"] == "LOW_RELEVANCE"


def test_query_plural_contract_honors_reranker_on_and_off() -> None:
    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        FakeReranker,
        make_service,
    )

    reranker = FakeReranker()
    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(reranker),
            answerability=AnswerabilityPolicy(min_dense_score=0.45),
            answer_generator=FakeAnswerGenerator(),
        ),
    )

    off = asyncio.run(
        post_json(
            app,
            "/v1/queries",
            {
                "question": "Qdrant ne işe yarar?",
                "top_k": 2,
                "include_debug": True,
                "reranker_enabled": False,
            },
        )
    )
    assert off.status_code == 200
    assert off.json()["retrieval"]["reranker_enabled"] is False
    assert off.json()["retrieval"]["reranked_candidates"] == 0
    assert reranker.seen_count == 0

    on = asyncio.run(
        post_json(
            app,
            "/v1/queries",
            {
                "question": "Qdrant ne işe yarar?",
                "top_k": 2,
                "include_debug": True,
                "reranker_enabled": True,
            },
        )
    )
    assert on.status_code == 200
    assert on.json()["retrieval"]["reranker_enabled"] is True
    assert on.json()["retrieval"]["reranked_candidates"] == 2
    assert reranker.seen_count == 3


def test_query_rejects_tenant_header_body_mismatch() -> None:
    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        make_service,
    )

    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(),
            answerability=AnswerabilityPolicy(min_dense_score=0.99),
            answer_generator=FakeAnswerGenerator(),
        ),
    )

    response = asyncio.run(
        post_json_with_headers(
            app,
            "/v1/queries",
            {"question": "Stajyer maaşı ne kadar?", "tenant_id": "tenant-a"},
            {"X-Tenant-ID": "tenant-b"},
        )
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_wired_query_exposes_output_warning_and_canonical_sources() -> None:
    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        make_service,
    )

    generator = FakeAnswerGenerator(answer="Sistem 64 GB RAM kullanır.")
    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(),
            answerability=AnswerabilityPolicy(min_dense_score=0.45),
            answer_generator=generator,
        ),
    )

    response = asyncio.run(
        post_json(
            app,
            "/v1/query",
            {"question": "Qdrant ne işe yarar?", "top_k": 2},
        )
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "answered"
    assert body["warnings"] == [
        {
            "code": "UNSUPPORTED_NUMBER",
            "message": (
                "Cevapta geçen bazı sayılar getirilen kanıtta bulunamadı; "
                "cevap insan incelemesine gönderilmelidir."
            ),
            "values": ["64"],
        }
    ]
    assert [source["source_id"] for source in body["sources"]] == [
        "shared",
        "dense-top",
    ]


def test_query_debug_flag_exposes_ranks_without_raw_evidence() -> None:
    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        make_service,
    )

    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(),
            answerability=AnswerabilityPolicy(min_dense_score=0.45),
            answer_generator=FakeAnswerGenerator(answer="Kanıta dayalı cevap."),
        ),
    )

    response = asyncio.run(
        post_json(
            app,
            "/v1/query",
            {"question": "Qdrant ne işe yarar?", "include_debug": True},
        )
    )

    assert response.status_code == 200
    debug = response.json()["debug"]
    assert [item["source_id"] for item in debug["candidates"]] == [
        "shared",
        "dense-top",
        "sparse-only",
    ]
    assert "text" not in debug["candidates"][0]


def test_query_keeps_html_like_answer_in_json_transport() -> None:
    """Prove the API does not directly render model text as HTML."""

    from projects.document_intelligence_service.app.application.query_service import (
        QueryService,
    )
    from projects.document_intelligence_service.app.domain.answerability import (
        AnswerabilityPolicy,
    )
    from projects.document_intelligence_service.tests.unit.test_query_service import (
        FakeAnswerGenerator,
    )
    from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
        make_service,
    )

    app = create_app(
        health_service=HealthService(()),
        query_service=QueryService(
            retrieval_service=make_service(),
            answerability=AnswerabilityPolicy(min_dense_score=0.45),
            answer_generator=FakeAnswerGenerator(
                answer="<script>alert('exfiltrate')</script> **cevap**"
            ),
        ),
    )

    response = asyncio.run(
        post_json(
            app,
            "/v1/query",
            {"question": "Qdrant ne işe yarar?"},
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["answer"] == (
        "<script>alert('exfiltrate')</script> **cevap**"
    )


def test_invalid_query_uses_common_validation_envelope() -> None:
    app = create_app(health_service=HealthService(()))

    response = asyncio.run(post_json(app, "/v1/query", {"question": ""}))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert response.json()["error"]["request_id"] == "contract-1"


def test_upload_returns_202_and_job_can_be_read() -> None:
    app = create_app(health_service=HealthService(()))

    response = asyncio.run(post_multipart(app, "/v1/documents", pdf_bytes()))

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "indexing"
    assert payload["request_id"] == "upload-contract-1"

    job_response = asyncio.run(get(app, f"/v1/jobs/{payload['job_id']}"))
    assert job_response.status_code == 200
    assert job_response.json() == {
        "job_id": payload["job_id"],
        "document_id": payload["document_id"],
        "status": "queued",
        "progress_percent": 0,
        "error_code": None,
        "request_id": "job-contract-1",
        "current_stage": None,
        "stages": [],
        "page_count": 1,
            "point_count": None,
            "error_message": None,
            "failed_stage": None,
            "attempt_count": 0,
            "max_attempts": 3,
            "next_attempt_at": None,
            "last_attempt_at": None,
        }


def test_same_pdf_and_pipeline_do_not_create_duplicate_job() -> None:
    app = create_app(health_service=HealthService(()))
    content = pdf_bytes()

    first = asyncio.run(post_multipart(app, "/v1/documents", content))
    second = asyncio.run(post_multipart(app, "/v1/documents", content))

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["document_id"] == second.json()["document_id"]
    assert first.json()["version_id"] == second.json()["version_id"]
    assert first.json()["job_id"] == second.json()["job_id"]
    assert first.json()["idempotent_hit"] is False
    assert second.json()["idempotent_hit"] is True


def test_reusing_idempotency_key_for_different_content_returns_409() -> None:
    app = create_app(health_service=HealthService(()))

    first = asyncio.run(
        post_multipart(app, "/v1/documents", pdf_bytes(1), "same-key")
    )
    conflict = asyncio.run(
        post_multipart(app, "/v1/documents", pdf_bytes(2), "same-key")
    )

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json() == {
        "error": {
            "code": "INGESTION_CONFLICT",
            "message": "Idempotency-Key was already used for another upload",
            "request_id": "upload-contract-1",
        }
    }


def test_document_catalog_lists_details_and_rejects_busy_delete() -> None:
    app = create_app(health_service=HealthService(()))

    upload = asyncio.run(post_multipart(app, "/v1/documents", pdf_bytes()))
    payload = upload.json()

    listing = asyncio.run(get(app, "/v1/documents"))
    assert listing.status_code == 200
    assert listing.json()["next_cursor"] is None
    assert listing.json()["items"] == [
        {
            "document_id": payload["document_id"],
            "title": "guide.pdf",
            "content_hash": payload["document_id"][4:],
            "active_version_id": None,
                "status": "indexing",
                "created_at": listing.json()["items"][0]["created_at"],
                "tenant_id": "default",
            }
        ]

    detail = asyncio.run(get(app, f"/v1/documents/{payload['document_id']}"))
    assert detail.status_code == 200
    assert detail.json()["available_version_ids"] == [payload["version_id"]]

    deletion = asyncio.run(
        request_with_method(app, "DELETE", f"/v1/documents/{payload['document_id']}")
    )
    assert deletion.status_code == 409
    assert deletion.json()["error"]["code"] == "DOCUMENT_BUSY"


def test_document_catalog_returns_safe_not_found_and_cursor_errors() -> None:
    app = create_app(health_service=HealthService(()))

    missing = asyncio.run(get(app, "/v1/documents/doc_missing"))
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    invalid_cursor = asyncio.run(get(app, "/v1/documents?cursor=not-a-cursor"))
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.json()["error"]["code"] == "INVALID_REQUEST"


def test_evaluation_run_contract_returns_run_and_list_state() -> None:
    from projects.document_intelligence_service.app.application.evaluation_service import (
        EvaluationExecution,
        EvaluationService,
    )
    from projects.document_intelligence_service.app.domain.entities import RetrievalMode
    from projects.document_intelligence_service.app.infrastructure.storage.in_memory_evaluation_registry import (
        InMemoryEvaluationRegistry,
    )

    class FakeExecutor:
        def execute(self, spec: object) -> EvaluationExecution:
            del spec
            return EvaluationExecution(
                case_count=2,
                metrics={"recall_at_5": 0.5},
                raw={"observations": []},
            )

    service = EvaluationService(
        registry=InMemoryEvaluationRegistry(),
        executor=FakeExecutor(),
        artifact_dir="/tmp/document-intelligence-api-eval",
        repo_root="/tmp",
    )
    app = create_app(
        health_service=HealthService(()),
        evaluation_service=service,
    )

    response = asyncio.run(
        post_json(
            app,
            "/v1/evaluations/runs",
            {
                "evaluation_type": "retrieval",
                "dataset": "mentor_program_pdf_rag_golden_v1",
                "split": "test",
                "mode": RetrievalMode.HYBRID.value,
                "top_k": 5,
            },
        )
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"

    completed = asyncio.run(service.execute_run(payload["run_id"]))
    assert completed.status.value == "succeeded"
    assert completed.case_count == 2
    assert completed.metrics == {
        "recall_at_5": 0.5,
        "failure_count": 0,
        "failure_rate": 0.0,
    }

    detail = asyncio.run(get(app, f"/v1/evaluations/runs/{payload['run_id']}"))
    assert detail.status_code == 200
    assert detail.json()["artifact_path"]
    listing = asyncio.run(get(app, "/v1/evaluations/runs"))
    assert listing.status_code == 200
    assert listing.json()["items"][0]["run_id"] == payload["run_id"]


async def request_with_method(
    app: FastAPI,
    method: str,
    path: str,
) -> httpx.Response:
    """Call a non-GET resource route through the ASGI lifespan."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-Request-ID": "delete-contract-1"},
        ) as client:
            return await client.request(method, path)
