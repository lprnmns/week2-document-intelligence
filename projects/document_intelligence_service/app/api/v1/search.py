"""Evidence-only retrieval contract routes."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Header, Request, status

from ..errors import openapi_error_responses
from ...observability.request_id import get_request_id
from ...domain.errors import ErrorCode, ServiceError
from ._not_ready import feature_not_ready
from .scope import resolve_request_scope
from .contracts import (
    LatencyBreakdown,
    RetrievalDebugCandidateResponse,
    RetrievalDebugResponse,
    RetrievalInfo,
    SearchRequest,
    SearchResponse,
    SourceResponse,
    DocumentCandidateDistributionResponse,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    response_model_exclude_defaults=True,
    responses={
        status.HTTP_200_OK: {"model": SearchResponse},
        **openapi_error_responses(),
    },
)
async def search(
    request: Request,
    payload: SearchRequest,
    header_tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    header_acl_tags: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> SearchResponse:
    """Return retrieval evidence without calling the LLM."""

    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    if retrieval_service is None:
        feature_not_ready("Search")
    scope = resolve_request_scope(
        body_tenant_id=payload.tenant_id,
        header_tenant_id=header_tenant_id,
        body_acl_tags=payload.acl_tags,
        header_acl_tags=header_acl_tags,
    )
    search_kwargs: dict[str, object] = {
        "question": payload.question,
        "mode": payload.retrieval_mode,
        "top_k": payload.top_k,
        "document_ids": payload.document_ids,
        "tenant_id": scope.tenant_id,
        "acl_tags": scope.acl_tags,
    }
    if payload.reranker_enabled is not None:
        search_kwargs["reranker_enabled"] = payload.reranker_enabled
    try:
        result = await asyncio.to_thread(
            retrieval_service.search,
            **search_kwargs,
        )
    except ServiceError:
        raise
    except Exception as error:
        raise ServiceError(
            code=ErrorCode.DEPENDENCY_UNAVAILABLE,
            message="Retrieval dependency is unavailable",
        ) from error
    return SearchResponse(
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
                score=candidate.score,
                dense_score=candidate.dense_score,
                sparse_score=candidate.sparse_score,
                rerank_score=candidate.rerank_score,
                dense_rank=candidate.dense_rank if payload.include_debug else None,
                sparse_rank=candidate.sparse_rank if payload.include_debug else None,
                fusion_rank=candidate.fusion_rank if payload.include_debug else None,
                rerank_rank=(
                    candidate.rank
                    if payload.include_debug and candidate.rerank_score is not None
                    else None
                ),
                selected_as_evidence=payload.include_debug,
            )
            for candidate in result.candidates
        ],
        retrieval=RetrievalInfo(
            mode=payload.retrieval_mode,
            dense_candidates=result.dense_candidates,
            sparse_candidates=result.sparse_candidates,
            rrf_candidates=result.rrf_candidates,
            reranked_candidates=result.reranked_candidates,
            candidate_limit=result.candidate_limit if payload.include_debug else 0,
            fusion_limit=result.fusion_limit if payload.include_debug else 0,
            rerank_limit=result.rerank_limit if payload.include_debug else 0,
            reranker_enabled=result.reranker_enabled,
            reranker_skipped_reason=(
                result.reranker_skipped_reason if payload.include_debug else None
            ),
            dense_distribution=[
                DocumentCandidateDistributionResponse(
                    document_id=item.document_id,
                    title=item.title,
                    count=item.count,
                )
                for item in result.dense_distribution
            ] if payload.include_debug else [],
            sparse_distribution=[
                DocumentCandidateDistributionResponse(
                    document_id=item.document_id,
                    title=item.title,
                    count=item.count,
                )
                for item in result.sparse_distribution
            ] if payload.include_debug else [],
            dense_model=result.dense_model if payload.include_debug else None,
            sparse_model=result.sparse_model if payload.include_debug else None,
            reranker_model=result.reranker_model if payload.include_debug else None,
        ),
        latency=LatencyBreakdown(
            embedding_ms=result.embedding_ms,
            search_ms=result.search_ms,
            rerank_ms=result.rerank_ms,
            llm_ms=0,
            total_ms=result.embedding_ms + result.search_ms + result.rerank_ms,
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
                        document_id=item.document_id if payload.include_debug else "",
                        title=item.title if payload.include_debug else "",
                        page_start=item.page_start if payload.include_debug else None,
                        page_end=item.page_end if payload.include_debug else None,
                        excerpt=item.excerpt if payload.include_debug else "",
                        fusion_rank=item.fusion_rank if payload.include_debug else None,
                        selected_as_evidence=(
                            item.selected_as_evidence if payload.include_debug else False
                        ),
                        rank_delta=item.rank_delta if payload.include_debug else None,
                    )
                    for item in result.debug_candidates
                ]
            )
            if payload.include_debug
            else None
        ),
        request_id=get_request_id(),
    )
