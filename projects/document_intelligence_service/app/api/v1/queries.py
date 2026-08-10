"""Answer-generation query contract routes."""

from typing import Annotated

from fastapi import APIRouter, Header, Request, status

from ..errors import openapi_error_responses
from ...observability.request_id import get_request_id
from ._not_ready import feature_not_ready
from .scope import resolve_request_scope
from .contracts import (
    LatencyBreakdown,
    ModelInfo,
    OutputWarningResponse,
    QueryRequest,
    QueryResponse,
    NoAnswerInfo,
    RetrievalInfo,
    RetrievalDebugCandidateResponse,
    RetrievalDebugResponse,
    SourceResponse,
    DocumentCandidateDistributionResponse,
)

router = APIRouter(prefix="/queries", tags=["query"])
legacy_router = APIRouter(prefix="/query", tags=["query"])


@legacy_router.post(
    "",
    response_model=QueryResponse,
    include_in_schema=True,
    responses={
        status.HTTP_200_OK: {"model": QueryResponse},
        **openapi_error_responses(),
    },
)
@router.post(
    "",
    response_model=QueryResponse,
    responses={
        status.HTTP_200_OK: {"model": QueryResponse},
        **openapi_error_responses(),
    },
)
async def query(
    http_request: Request,
    request: QueryRequest,
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> QueryResponse:
    """Answer from filtered evidence or return a structured no-answer."""

    query_service = getattr(http_request.app.state, "query_service", None)
    if query_service is None:
        feature_not_ready("Query")
    scope = resolve_request_scope(
        body_tenant_id=request.tenant_id,
        header_tenant_id=header_tenant_id,
        body_acl_tags=request.acl_tags,
        header_acl_tags=header_acl_tags,
    )
    query_kwargs: dict[str, object] = {
        "question": request.question,
        "mode": request.retrieval_mode,
        "top_k": request.top_k,
        "document_ids": request.document_ids,
        "tenant_id": scope.tenant_id,
        "acl_tags": scope.acl_tags,
    }
    if request.reranker_enabled is not None:
        query_kwargs["reranker_enabled"] = request.reranker_enabled
    result = await query_service.execute(**query_kwargs)
    return QueryResponse(
        decision=result.decision,
        answer=result.answer,
        no_answer_reason=result.no_answer_reason,
        no_answer=(
            NoAnswerInfo(
                reason_code=result.no_answer_reason,
                message=_no_answer_message(result.no_answer_reason),
                searched_document_ids=list(request.document_ids),
            )
            if result.no_answer_reason is not None
            else None
        ),
        sources=[
            SourceResponse(
                source_id=candidate.source_id,
                document_id=candidate.document_id,
                version_id=candidate.version_id,
                chunk_id=candidate.source_id,
                parent_id=candidate.parent_id,
                page=candidate.page_start,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                title=candidate.title or None,
                snippet=candidate.text[:500],
                excerpt=candidate.text[:500],
                score=(
                    candidate.rerank_score
                    if candidate.rerank_score is not None
                    else candidate.score
                ),
                dense_score=candidate.dense_score,
                sparse_score=candidate.sparse_score,
                rerank_score=candidate.rerank_score,
                dense_rank=candidate.dense_rank if request.include_debug else None,
                sparse_rank=candidate.sparse_rank if request.include_debug else None,
                fusion_rank=candidate.fusion_rank if request.include_debug else None,
                rerank_rank=(
                    candidate.rank
                    if request.include_debug and candidate.rerank_score is not None
                    else None
                ),
                selected_as_evidence=request.include_debug,
            )
            for candidate in result.sources
        ],
        retrieval=RetrievalInfo(
            mode=request.retrieval_mode,
            dense_candidates=result.retrieval.dense_candidates,
            sparse_candidates=result.retrieval.sparse_candidates,
            rrf_candidates=result.retrieval.rrf_candidates,
            reranked_candidates=result.retrieval.reranked_candidates,
            candidate_limit=(
                result.retrieval.candidate_limit if request.include_debug else 0
            ),
            fusion_limit=(
                result.retrieval.fusion_limit if request.include_debug else 0
            ),
            rerank_limit=(
                result.retrieval.rerank_limit if request.include_debug else 0
            ),
            reranker_enabled=result.retrieval.reranker_enabled,
            reranker_skipped_reason=(
                result.retrieval.reranker_skipped_reason
                if request.include_debug
                else None
            ),
            dense_distribution=[
                DocumentCandidateDistributionResponse(
                    document_id=item.document_id,
                    title=item.title,
                    count=item.count,
                )
                for item in result.retrieval.dense_distribution
            ] if request.include_debug else [],
            sparse_distribution=[
                DocumentCandidateDistributionResponse(
                    document_id=item.document_id,
                    title=item.title,
                    count=item.count,
                )
                for item in result.retrieval.sparse_distribution
            ] if request.include_debug else [],
            dense_model=(
                result.retrieval.dense_model if request.include_debug else None
            ),
            sparse_model=(
                result.retrieval.sparse_model if request.include_debug else None
            ),
            reranker_model=(
                result.retrieval.reranker_model if request.include_debug else None
            ),
        ),
        model=ModelInfo(provider=result.provider, model=result.model),
        warnings=[
            OutputWarningResponse(
                code=warning.code,
                message=warning.message,
                values=list(warning.values),
            )
            for warning in result.warnings
        ],
        latency=LatencyBreakdown(
            embedding_ms=result.retrieval.embedding_ms,
            search_ms=result.retrieval.search_ms,
            rerank_ms=result.retrieval.rerank_ms,
            llm_ms=result.llm_ms,
            total_ms=result.total_ms,
        ),
        debug=(
            RetrievalDebugResponse(
                candidates=[
                    RetrievalDebugCandidateResponse(
                        source_id=item.source_id,
                        retrieval_rank=item.retrieval_rank,
                        rerank_rank=item.rerank_rank,
                        dense_rank=item.dense_rank,
                        sparse_rank=item.sparse_rank,
                        dense_score=item.dense_score,
                        sparse_score=item.sparse_score,
                        fused_score=item.fused_score,
                        rerank_score=item.rerank_score,
                        matched_terms=list(item.matched_terms),
                        document_id=item.document_id if request.include_debug else "",
                        title=item.title if request.include_debug else "",
                        page_start=item.page_start if request.include_debug else None,
                        page_end=item.page_end if request.include_debug else None,
                        excerpt=item.excerpt if request.include_debug else "",
                        fusion_rank=item.fusion_rank if request.include_debug else None,
                        selected_as_evidence=(
                            item.selected_as_evidence
                            if request.include_debug
                            else False
                        ),
                        rank_delta=item.rank_delta if request.include_debug else None,
                    )
                    for item in result.retrieval.debug_candidates
                ]
            )
            if request.include_debug
            else None
        ),
        request_id=get_request_id(),
    )


def _no_answer_message(reason: object) -> str:
    """Explain the policy gate without exposing scores or internal prompts."""

    messages = {
        "NO_EVIDENCE": "Aranan kapsamda kanıt bulunamadı; LLM çağrısı atlandı.",
        "LOW_RELEVANCE": "Bulunan kanıt relevance eşiğini geçmedi; LLM çağrısı atlandı.",
        "INSUFFICIENT_COVERAGE": "Kanıt sorunun gerekli kısmını kapsamıyor; LLM çağrısı atlandı.",
        "SECURITY_POLICY": "İstek veya kanıt güvenlik politikası nedeniyle reddedildi; LLM çağrısı atlandı.",
    }
    return messages.get(
        getattr(reason, "value", str(reason)),
        "Yeterli kanıt bulunamadı; LLM çağrısı atlandı.",
    )
