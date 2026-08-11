"""Development-only polling transport for the engineering inspection UI."""

import asyncio
from typing import cast
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Header, Request, status
from pydantic import BaseModel, Field

from ...application.gold_diagnostic import GoldDiagnosticService
from ...application.query_service import QueryExecutionResult
from ...domain.entities import RetrievalMode
from ...domain.errors import ErrorCode, ServiceError
from ...observability.request_id import new_request_id, request_id_context
from ...observability.query_trace import LiveQueryTraceStore
from ..errors import openapi_error_responses
from .scope import resolve_request_scope

router = APIRouter(prefix="/demo", tags=["demo"], include_in_schema=False)


class DemoQueryRunRequest(BaseModel):
    """Bounded query configuration for the local live-trace console."""

    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    reranker_enabled: bool | None = None
    generation_model: str | None = Field(default=None, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=128)
    acl_tags: list[str] = Field(default_factory=list, max_length=50)


class DemoQueryRunStartResponse(BaseModel):
    """Identifier returned before the asynchronous demo run completes."""

    run_id: str
    request_id: str
    status: str = "pending"


class GoldDiagnosticRunRequest(BaseModel):
    """Bounded request for one curated or advanced diagnostic run."""

    case_id: str | None = Field(default=None, max_length=128)
    question: str | None = Field(default=None, min_length=1, max_length=4000)
    expected_answer: str | None = Field(default=None, max_length=4000)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    reranker_enabled: bool = False
    generation_model: str | None = Field(default=None, max_length=128)
    retrieval_only: bool = False
    tenant_id: str | None = Field(default=None, max_length=128)
    acl_tags: list[str] = Field(default_factory=list, max_length=50)


@router.get("/gold/cases")
async def list_gold_cases(request: Request) -> dict[str, object]:
    """Return committed curated cases for the separate Demo Lab UI."""

    service = _gold_service(request)
    return {"cases": list(service.list_cases())}


@router.post("/gold/prepare")
async def prepare_gold_corpus(
    request: Request,
    background_tasks: BackgroundTasks,
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> dict[str, object]:
    """Prepare Atlas assets through the regular document ingestion path."""

    service = _gold_service(request)
    scope = resolve_request_scope(
        body_tenant_id=None,
        header_tenant_id=header_tenant_id,
        body_acl_tags=(),
        header_acl_tags=header_acl_tags,
    )
    receipts = await service.prepare_corpus(
        tenant_id=scope.tenant_id,
        acl_tags=scope.acl_tags,
    )
    worker = getattr(request.app.state, "ingestion_worker", None)
    if worker is not None:
        for receipt in receipts:
            job_id = receipt.get("job_id")
            if isinstance(job_id, str):
                background_tasks.add_task(worker.run_job, job_id)
    return {"status": "accepted", "receipts": list(receipts)}


@router.post("/gold/runs")
async def run_gold_diagnostic(
    request: Request,
    payload: GoldDiagnosticRunRequest,
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> dict[str, object]:
    """Run a deterministic gold-aware diagnostic without changing Query Trace."""

    service = _gold_service(request)
    scope = resolve_request_scope(
        body_tenant_id=payload.tenant_id,
        header_tenant_id=header_tenant_id,
        body_acl_tags=payload.acl_tags,
        header_acl_tags=header_acl_tags,
    )
    if payload.case_id is None and (payload.question is None or payload.expected_answer is None):
        raise ServiceError(
            code=ErrorCode.INVALID_REQUEST,
            message="case_id or question plus expected_answer is required",
        )
    if payload.generation_model is not None and not payload.retrieval_only:
        model_service = getattr(request.app.state, "model_service", None)
        if model_service is None:
            raise ServiceError(
                code=ErrorCode.FEATURE_NOT_READY,
                message="Model service is not wired",
            )
        model_status = await model_service.validate_generation_model(
            payload.generation_model
        )
        if model_status != "ready":
            raise ServiceError(
                code=(
                    ErrorCode.INVALID_REQUEST
                    if model_status == "not_allowlisted"
                    else ErrorCode.DEPENDENCY_UNAVAILABLE
                ),
                message="Selected diagnostic generation model is not ready",
                stage="model",
                reason=model_status,
            )
    if payload.case_id is not None:
        return await service.run_case(
            case_id=payload.case_id,
            mode=payload.retrieval_mode,
            top_k=payload.top_k,
            reranker_enabled=payload.reranker_enabled,
            generation_model=payload.generation_model,
            retrieval_only=payload.retrieval_only,
            tenant_id=scope.tenant_id,
            acl_tags=scope.acl_tags,
        )
    assert payload.question is not None
    assert payload.expected_answer is not None
    return await service.run_custom(
        question=payload.question,
        expected_answer=payload.expected_answer,
        mode=payload.retrieval_mode,
        top_k=payload.top_k,
        reranker_enabled=payload.reranker_enabled,
        generation_model=payload.generation_model,
        tenant_id=scope.tenant_id,
        acl_tags=scope.acl_tags,
    )


@router.get("/gold/recorded-reranker")
async def get_recorded_reranker_case(
    request: Request,
    case_id: str = "direct_08",
) -> dict[str, object]:
    """Expose a read-only historical reranker flip, never a live run."""

    return await _gold_service(request).recorded_reranker_case(case_id)


def _gold_service(request: Request) -> GoldDiagnosticService:
    service = getattr(request.app.state, "gold_diagnostic_service", None)
    if service is None:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Gold Diagnostic service is not wired",
        )
    return cast(GoldDiagnosticService, service)


@router.post(
    "/query-runs",
    response_model=DemoQueryRunStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"model": DemoQueryRunStartResponse},
        **openapi_error_responses(),
    },
)
async def start_query_run(
    request: Request,
    payload: DemoQueryRunRequest,
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> DemoQueryRunStartResponse:
    """Start the real QueryService and return a polling handle."""

    settings = getattr(request.app.state, "settings", None)
    if settings is not None and not settings.demo_trace_enabled:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Demo trace transport is disabled",
        )
    query_service = getattr(request.app.state, "query_service", None)
    if query_service is None:
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Query workflow is not wired yet",
        )
    scope = resolve_request_scope(
        body_tenant_id=payload.tenant_id,
        header_tenant_id=header_tenant_id,
        body_acl_tags=payload.acl_tags,
        header_acl_tags=header_acl_tags,
    )
    if payload.generation_model is not None:
        model_service = getattr(request.app.state, "model_service", None)
        if model_service is None:
            raise ServiceError(
                code=ErrorCode.FEATURE_NOT_READY,
                message="Model service is not wired",
            )
        model_status = await model_service.validate_generation_model(
            payload.generation_model
        )
        if model_status != "ready":
            raise ServiceError(
                code=(
                    ErrorCode.INVALID_REQUEST
                    if model_status == "not_allowlisted"
                    else ErrorCode.DEPENDENCY_UNAVAILABLE
                ),
                message=(
                    "Selected generation model is not in the configured catalog"
                    if model_status == "not_allowlisted"
                    else (
                        "Selected generation model is not installed"
                        if model_status == "model_missing"
                        else "Selected generation runtime is unavailable"
                    )
                ),
                stage="model",
                reason=model_status,
            )
    store: LiveQueryTraceStore = request.app.state.demo_trace_store
    request_id = new_request_id()
    run_id = store.create(request_id=request_id)
    asyncio.create_task(
        _run_query(
            query_service=query_service,
            model_service=getattr(request.app.state, "model_service", None),
            store=store,
            run_id=run_id,
            request_id=request_id,
            payload=payload,
            tenant_id=scope.tenant_id,
            acl_tags=scope.acl_tags,
        )
    )
    return DemoQueryRunStartResponse(run_id=run_id, request_id=request_id)


@router.get(
    "/query-runs/{run_id}",
    responses={**openapi_error_responses()},
)
async def get_query_run(request: Request, run_id: str) -> dict[str, object]:
    """Return the current bounded stage trace for the local UI."""

    try:
        return cast(
            dict[str, object],
            request.app.state.demo_trace_store.snapshot(run_id),
        )
    except KeyError as error:
        raise ServiceError(
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message="Demo query trace was not found or expired",
        ) from error


async def _run_query(
    *,
    query_service: object,
    model_service: object | None,
    store: LiveQueryTraceStore,
    run_id: str,
    request_id: str,
    payload: DemoQueryRunRequest,
    tenant_id: str,
    acl_tags: tuple[str, ...],
) -> None:
    """Run the application use-case in a background task with correlation."""

    with request_id_context(request_id):
        recorder = store.recorder(run_id)
        try:
            result = await query_service.execute(  # type: ignore[attr-defined]
                question=payload.question,
                mode=payload.retrieval_mode,
                top_k=payload.top_k,
                document_ids=payload.document_ids,
                tenant_id=tenant_id,
                acl_tags=acl_tags,
                reranker_enabled=payload.reranker_enabled,
                generation_model=payload.generation_model,
                trace=recorder.emit,
            )
        except ServiceError as error:
            if error.stage == "llm" and payload.generation_model:
                _record_generation_probe(
                    model_service,
                    payload.generation_model,
                    status="last_probe_failed",
                    reason=error.reason,
                )
            recorder.emit(
                "response",
                "failed",
                "Query execution failed",
                {
                    "code": error.code.value,
                    "stage": error.stage,
                    "reason": error.reason,
                },
                None,
            )
            store.fail(
                run_id,
                {
                    "code": error.code.value,
                    "message": error.message,
                    "stage": error.stage,
                    "reason": error.reason,
                    "request_id": request_id,
                },
            )
        except Exception:
            recorder.emit(
                "response",
                "failed",
                "Query execution failed",
                {"code": ErrorCode.DEPENDENCY_UNAVAILABLE.value},
                None,
            )
            store.fail(
                run_id,
                {
                    "code": ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                    "message": "Query execution failed",
                    "request_id": request_id,
                },
            )
        else:
            if result.model:
                _record_generation_probe(
                    model_service,
                    result.model,
                    status="ready",
                )
            store.finish(run_id, _result_payload(result, request_id))


def _result_payload(result: QueryExecutionResult, request_id: str) -> dict[str, object]:
    """Project the application result into a compact, source-safe demo shape."""

    return {
        "request_id": request_id,
        "decision": result.decision.value,
        "answer": result.answer,
        "no_answer_reason": (
            result.no_answer_reason.value if result.no_answer_reason else None
        ),
        "sources": [
            {
                "source_id": item.source_id,
                "document_id": item.document_id,
                "parent_id": item.parent_id,
                "title": item.title,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "excerpt": _compact_excerpt(item.context_text),
                # These fields are exposed only by the development/demo
                # transport so the inspector can show the canonical child
                # and its bounded parent context without reconstructing text
                # in the browser.
                "chunk_text": _bounded_text(item.text),
                "parent_context": _bounded_text(item.parent_text or item.text),
                "parent_context_available": item.parent_text is not None,
                "used_in_prompt": True,
                "dense_rank": item.dense_rank,
                "sparse_rank": item.sparse_rank,
                "fusion_rank": item.fusion_rank,
                "rerank_rank": item.rank if item.rerank_score is not None else None,
                "rerank_score": item.rerank_score,
            }
            for item in result.sources
        ],
        "retrieval": {
            "mode": result.retrieval.mode,
            "dense_candidates": result.retrieval.dense_candidates,
            "sparse_candidates": result.retrieval.sparse_candidates,
            "rrf_candidates": result.retrieval.rrf_candidates,
            "reranked_candidates": result.retrieval.reranked_candidates,
            "candidate_limit": result.retrieval.candidate_limit,
            "fusion_limit": result.retrieval.fusion_limit,
            "rerank_limit": result.retrieval.rerank_limit,
            "reranker_enabled": result.retrieval.reranker_enabled,
            "reranker_skipped_reason": result.retrieval.reranker_skipped_reason,
            "dense_model": result.retrieval.dense_model,
            "sparse_model": result.retrieval.sparse_model,
            "reranker_model": result.retrieval.reranker_model,
            "dense_distribution": [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "count": item.count,
                }
                for item in result.retrieval.dense_distribution
            ],
            "sparse_distribution": [
                {
                    "document_id": item.document_id,
                    "title": item.title,
                    "count": item.count,
                }
                for item in result.retrieval.sparse_distribution
            ],
        },
        "answerability": {
            "decision": result.answerability.decision.value,
            "reason_code": (
                result.answerability.reason.value
                if result.answerability.reason
                else None
            ),
            "top_score": result.answerability.top_score,
            "score_margin": result.answerability.score_margin,
            "coverage_ratio": result.answerability.coverage_ratio,
            "required_qualifiers": list(
                result.answerability.required_qualifiers
            ),
            "matched_qualifiers": list(
                result.answerability.matched_qualifiers
            ),
            "missing_qualifiers": list(
                result.answerability.missing_qualifiers
            ),
            "qualifier_coverage_satisfied": (
                result.answerability.qualifier_coverage_satisfied
            ),
        },
        "model": {
            "provider": result.provider,
            "model": result.model,
        },
        "latency": {
            "embedding_ms": result.retrieval.embedding_ms,
            "search_ms": result.retrieval.search_ms,
            "rerank_ms": result.retrieval.rerank_ms,
            "llm_ms": result.llm_ms,
            "total_ms": result.total_ms,
        },
        "warnings": [warning.code.value for warning in result.warnings],
        "debug_candidates": [
            {
                "source_id": item.source_id,
                "document_id": item.document_id,
                "title": item.title,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "excerpt": item.excerpt,
                "dense_rank": item.dense_rank,
                "sparse_rank": item.sparse_rank,
                "fusion_rank": item.fusion_rank,
                "rerank_rank": item.rerank_rank,
                "dense_score": item.dense_score,
                "sparse_score": item.sparse_score,
                "fused_score": item.fused_score,
                "rerank_score": item.rerank_score,
                "selected_as_evidence": item.selected_as_evidence,
                "used_in_prompt": False,
                "rank_delta": item.rank_delta,
                "matched_terms": list(item.matched_terms),
            }
            for item in result.retrieval.debug_candidates
        ],
    }


def _compact_excerpt(text: str, limit: int = 260) -> str:
    """Keep the final demo evidence card bounded."""

    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _bounded_text(text: str, limit: int = 4000) -> str:
    """Expose bounded canonical evidence text to the local inspector only."""

    normalized = text.strip()
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _record_generation_probe(
    model_service: object | None,
    model_id: str,
    *,
    status: str,
    reason: str | None = None,
) -> None:
    recorder = getattr(model_service, "record_generation_probe", None)
    if callable(recorder):
        recorder(model_id, status=status, reason=reason)
