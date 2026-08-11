"""Development-only polling transport for the engineering inspection UI."""

import asyncio
from typing import cast
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Header, Query, Request, status
from pydantic import BaseModel, Field

from ...application.gold_diagnostic import GoldDiagnosticService
from ...application.query_service import QueryExecutionResult
from ...domain.entities import RetrievalMode
from ...domain.answer_check import AnswerCheckMode
from ...domain.errors import ErrorCode, ServiceError
from ...domain.retrieval import RetrievedChunk
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
    expected_answer: str | None = Field(default=None, max_length=4000)
    answer_check_mode: AnswerCheckMode = AnswerCheckMode.FACT_AWARE
    semantic_threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    gold_case_id: str | None = Field(default=None, max_length=128)
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


class TrustedEvidenceSelectionRequest(BaseModel):
    """Trusted evaluator metadata for an already completed demo run."""

    source_ids: list[str] = Field(min_length=1, max_length=20)
    expected_answer: str | None = Field(default=None, max_length=4000)
    question: str | None = Field(default=None, max_length=4000)


@router.get("/gold/cases")
async def list_gold_cases(request: Request) -> dict[str, object]:
    """Return committed curated cases for the optional ASK example selector."""

    service = _gold_service(request)
    return {"cases": list(service.list_cases())}


@router.get("/gold/evidence")
async def browse_gold_evidence(
    request: Request,
    document_ids: list[str] = Query(default=[]),
    page: int | None = Query(default=None, ge=1, le=10000),
    text: str = Query(default="", max_length=400),
    limit: int = Query(default=50, ge=1, le=100),
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> dict[str, object]:
    """Browse active chunks for optional trusted expected-evidence labels."""

    service = _gold_service(request)
    scope = resolve_request_scope(
        body_tenant_id=None,
        header_tenant_id=header_tenant_id,
        body_acl_tags=(),
        header_acl_tags=header_acl_tags,
    )
    items = await service.browse_evidence(
        document_ids=document_ids,
        page=page,
        text=text,
        tenant_id=scope.tenant_id,
        acl_tags=scope.acl_tags,
        limit=limit,
    )
    return {"items": [_trusted_source_payload(item) for item in items]}


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
            gold_service=_gold_service(request),
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


@router.post(
    "/query-runs/{run_id}/trusted-evidence",
    responses={**openapi_error_responses()},
)
async def recompute_trusted_evidence_diagnostic(
    request: Request,
    run_id: str,
    payload: TrustedEvidenceSelectionRequest,
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> dict[str, object]:
    """Recompute stage attribution from one existing run, without re-running it."""

    store: LiveQueryTraceStore = request.app.state.demo_trace_store
    try:
        snapshot = store.snapshot(run_id)
        runtime_result = store.runtime_result(run_id)
    except KeyError as error:
        raise ServiceError(
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message="Demo query trace was not found or expired",
        ) from error
    if snapshot.get("status") not in {"completed", "failed"}:
        raise ServiceError(
            code=ErrorCode.INVALID_REQUEST,
            message="Trusted evidence can be selected after the query run completes",
        )
    if runtime_result is not None and not isinstance(runtime_result, QueryExecutionResult):
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Completed query result is no longer available for diagnosis",
        )
    error_payload = snapshot.get("error")
    if runtime_result is None and not isinstance(error_payload, dict):
        raise ServiceError(
            code=ErrorCode.FEATURE_NOT_READY,
            message="Completed query result is no longer available for diagnosis",
        )
    scope = resolve_request_scope(
        body_tenant_id=None,
        header_tenant_id=header_tenant_id,
        body_acl_tags=(),
        header_acl_tags=header_acl_tags,
    )
    expected_answer = payload.expected_answer
    result_payload = snapshot.get("result")
    if not expected_answer and isinstance(result_payload, dict):
        expected_check = result_payload.get("expected_check")
        if isinstance(expected_check, dict):
            expected = expected_check.get("expected")
            if isinstance(expected, dict) and isinstance(expected.get("answer"), str):
                expected_answer = expected["answer"]
    if not expected_answer or not expected_answer.strip():
        raise ServiceError(
            code=ErrorCode.INVALID_REQUEST,
            message="expected_answer is required for trusted evidence attribution",
        )
    question = payload.question or ""
    if not question and isinstance(result_payload, dict):
        expected_check = result_payload.get("expected_check")
        if isinstance(expected_check, dict) and isinstance(
            expected_check.get("question"), str
        ):
            question = expected_check["question"]
    service = _gold_service(request)
    diagnostic = await service.compare_existing_trusted_sources(
        question=question,
        expected_answer=expected_answer,
        source_ids=payload.source_ids,
        result=runtime_result,
        events=cast(list[dict[str, object]], snapshot.get("events", [])),
        error_payload=cast(dict[str, object] | None, error_payload),
        tenant_id=scope.tenant_id,
        acl_tags=scope.acl_tags,
    )
    store.merge_result(run_id, {"expected_check": diagnostic})
    return diagnostic


async def _run_query(
    *,
    query_service: object,
    model_service: object | None,
    gold_service: GoldDiagnosticService,
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
        trace_events: list[dict[str, object]] = []

        def trace(
            stage: str,
            status: str,
            summary: str,
            details: dict[str, object] | None = None,
            duration_ms: float | None = None,
        ) -> None:
            trace_events.append(
                {
                    "stage": stage,
                    "status": status,
                    "summary": summary,
                    "details": details or {},
                    "duration_ms": duration_ms,
                }
            )
            recorder.emit(stage, status, summary, details, duration_ms)

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
                trace=trace,
            )
        except ServiceError as error:
            if error.stage == "llm" and payload.generation_model:
                _record_generation_probe(
                    model_service,
                    payload.generation_model,
                    status="last_probe_failed",
                    reason=error.reason,
                )
            trace(
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
            error_payload: dict[str, object] = {
                "code": error.code.value,
                "message": error.message,
                "stage": error.stage,
                "reason": error.reason,
                "request_id": request_id,
            }
            diagnostic = await _compare_expected_run(
                gold_service=gold_service,
                payload=payload,
                result=None,
                events=trace_events,
                error_payload=error_payload,
                tenant_id=tenant_id,
            )
            if diagnostic is not None:
                error_payload["expected_check"] = diagnostic
            store.fail(run_id, error_payload)
        except Exception:
            trace(
                "response",
                "failed",
                "Query execution failed",
                {"code": ErrorCode.DEPENDENCY_UNAVAILABLE.value},
                None,
            )
            error_payload = {
                "code": ErrorCode.DEPENDENCY_UNAVAILABLE.value,
                "message": "Query execution failed",
                "request_id": request_id,
            }
            diagnostic = await _compare_expected_run(
                gold_service=gold_service,
                payload=payload,
                result=None,
                events=trace_events,
                error_payload=error_payload,
                tenant_id=tenant_id,
            )
            if diagnostic is not None:
                error_payload["expected_check"] = diagnostic
            store.fail(run_id, error_payload)
        else:
            if result.model:
                _record_generation_probe(
                    model_service,
                    result.model,
                    status="ready",
                )
            result_payload = _result_payload(result, request_id)
            diagnostic = await _compare_expected_run(
                gold_service=gold_service,
                payload=payload,
                result=result,
                events=trace_events,
                error_payload=None,
                tenant_id=tenant_id,
            )
            if diagnostic is not None:
                result_payload["expected_check"] = diagnostic
            store.finish(run_id, result_payload, runtime_result=result)


async def _compare_expected_run(
    *,
    gold_service: GoldDiagnosticService,
    payload: DemoQueryRunRequest,
    result: QueryExecutionResult | None,
    events: list[dict[str, object]],
    error_payload: dict[str, object] | None,
    tenant_id: str,
) -> dict[str, object] | None:
    """Compare an already executed ASK run without starting another query."""

    if payload.gold_case_id:
        return await gold_service.compare_existing_case(
            case_id=payload.gold_case_id,
            result=result,
            events=events,
            error_payload=error_payload,
            mode=payload.retrieval_mode,
            tenant_id=tenant_id,
        )
    if payload.expected_answer and payload.expected_answer.strip():
        return await gold_service.compare_existing_custom(
            question=payload.question,
            expected_answer=payload.expected_answer,
            result=result,
            events=events,
            error_payload=error_payload,
            answer_check_mode=payload.answer_check_mode,
            semantic_threshold=payload.semantic_threshold,
        )
    return None


def _result_payload(result: QueryExecutionResult, request_id: str) -> dict[str, object]:
    """Project the application result into a compact, source-safe demo shape."""

    prompt_pack = result.prompt_pack
    included_ids = set(prompt_pack.included_source_ids) if prompt_pack else set()
    fragment_by_id = {
        fragment.source_id: fragment.as_dict()
        for fragment in prompt_pack.fragments
    } if prompt_pack else {}

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
                "selected_as_evidence": True,
                "used_in_prompt": item.source_id in included_ids,
                "prompt_fragment": fragment_by_id.get(item.source_id),
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
        "prompt_pack": prompt_pack.as_dict() if prompt_pack else None,
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
                "parent_id": item.parent_id,
                "version_id": item.version_id,
                "title": item.title,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "excerpt": item.excerpt,
                "chunk_text": _bounded_text(item.chunk_text),
                "parent_context": _bounded_text(item.parent_context),
                "chunking_profile": item.chunking_profile,
                "dense_rank": item.dense_rank,
                "sparse_rank": item.sparse_rank,
                "fusion_rank": item.fusion_rank,
                "rerank_rank": item.rerank_rank,
                "dense_score": item.dense_score,
                "sparse_score": item.sparse_score,
                "fused_score": item.fused_score,
                "rerank_score": item.rerank_score,
                "selected_as_evidence": item.selected_as_evidence,
                "used_in_prompt": item.source_id in included_ids,
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


def _trusted_source_payload(item: RetrievedChunk) -> dict[str, object]:
    """Serialize one active chunk for the optional trusted-evidence picker."""

    return {
        "source_id": item.source_id,
        "document_id": item.document_id,
        "title": item.title,
        "page_start": item.page_start,
        "page_end": item.page_end,
        "parent_id": item.parent_id,
        "chunk_text": _bounded_text(item.text),
        "parent_context": _bounded_text(item.parent_text or item.text),
        "tenant_id": item.tenant_id,
        "acl_tags": list(item.acl_tags),
        "chunking_profile": item.chunking_profile,
    }


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
