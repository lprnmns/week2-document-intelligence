"""Tests for dense, sparse and hybrid retrieval orchestration."""

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

from projects.document_intelligence_service.app.application.retrieval_service import (
    RetrievalService,
)
from projects.document_intelligence_service.app.domain.entities import RetrievalMode
from projects.document_intelligence_service.app.domain.retrieval import RetrievedChunk
from projects.document_intelligence_service.app.domain.vectors import SparseVector


def candidate(source_id: str) -> RetrievedChunk:
    """Create a compact candidate fixture."""

    return RetrievedChunk(
        source_id=source_id,
        document_id="doc-1",
        version_id="ver-1",
        parent_id="parent-1",
        title="RAG",
        text=f"text-{source_id}",
        page_start=1,
        page_end=1,
        score=0.5,
        rank=1,
    )


class FakeDenseEmbedder:
    """Return one deterministic query vector."""

    dimension = 2

    def __init__(self) -> None:
        self.warmup_called = False

    def warmup(self) -> None:
        self.warmup_called = True

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


class FakeSparseEmbedder:
    """Return one deterministic query sparse vector."""

    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        return tuple(SparseVector(indices=(1,), values=(1.0,)) for _ in texts)


class FakeRetriever:
    """Expose intentionally different dense and sparse rankings."""

    def search_dense(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> tuple[RetrievedChunk, ...]:
        del query_vector, limit, document_ids, tenant_id, acl_tags
        return (candidate("dense-top"), candidate("shared"))

    def search_sparse(
        self,
        *,
        query_vector: SparseVector,
        limit: int,
        document_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> tuple[RetrievedChunk, ...]:
        del query_vector, limit, document_ids, tenant_id, acl_tags
        return (candidate("shared"), candidate("sparse-only"))


class FakeReranker:
    """Reverse candidates to prove the optional rerank stage is used."""

    def __init__(self) -> None:
        self.seen_count = 0
        self.warmup_called = False

    def warmup(self) -> None:
        self.warmup_called = True

    def rerank(
        self,
        *,
        question: str,
        candidates: Sequence[RetrievedChunk],
        limit: int,
    ) -> tuple[RetrievedChunk, ...]:
        del question
        self.seen_count = len(candidates)
        return tuple(
            replace(candidate, score=1.0, rerank_score=1.0, rank=index)
            for index, candidate in enumerate(
                tuple(candidates)[::-1][:limit], start=1
            )
        )


def make_service(reranker: FakeReranker | None = None) -> RetrievalService:
    """Build the service with fake infrastructure ports."""

    return RetrievalService(
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        retriever=FakeRetriever(),
        candidate_limit=30,
        rrf_k=60,
        fusion_limit=20,
        reranker=reranker,
    )


def test_warmup_calls_available_model_boundaries() -> None:
    dense = FakeDenseEmbedder()
    reranker = FakeReranker()
    service = RetrievalService(
        dense_embedder=dense,
        sparse_embedder=FakeSparseEmbedder(),
        retriever=FakeRetriever(),
        reranker=reranker,
    )

    service.warmup()

    assert dense.warmup_called
    assert reranker.warmup_called


def test_dense_mode_returns_only_dense_candidates() -> None:
    result = make_service().search(
        question="Qdrant ne işe yarar?",
        mode=RetrievalMode.DENSE,
        top_k=1,
    )

    assert result.mode == "dense"
    assert result.dense_candidates == 2
    assert result.sparse_candidates == 0
    assert result.rrf_candidates == 0
    assert [item.source_id for item in result.candidates] == ["dense-top"]


def test_hybrid_mode_uses_rank_based_rrf_not_raw_score_addition() -> None:
    result = make_service().search(
        question="Qdrant ne işe yarar?",
        mode=RetrievalMode.HYBRID,
        top_k=3,
    )

    assert result.dense_candidates == 2
    assert result.sparse_candidates == 2
    assert result.rrf_candidates == 3
    assert [item.source_id for item in result.candidates] == [
        "shared",
        "dense-top",
        "sparse-only",
    ]
    assert result.candidates[0].dense_rank == 2
    assert result.candidates[0].sparse_rank == 1
    assert result.candidates[0].fused_score is not None
    assert [item.source_id for item in result.debug_candidates] == [
        "shared",
        "dense-top",
        "sparse-only",
    ]
    assert result.debug_candidates[0].dense_rank == 2
    assert result.debug_candidates[0].sparse_rank == 1


def test_optional_reranker_receives_fused_candidates_and_returns_final_window() -> None:
    reranker = FakeReranker()
    result = make_service(reranker).search(
        question="Qdrant ne işe yarar?",
        mode=RetrievalMode.HYBRID,
        top_k=2,
    )

    assert reranker.seen_count == 3
    assert result.reranked_candidates == 2
    assert [item.source_id for item in result.candidate_window] == [
        "shared",
        "dense-top",
        "sparse-only",
    ]
    assert [item.source_id for item in result.candidates] == [
        "sparse-only",
        "dense-top",
    ]
    assert result.candidates[0].rerank_score == 1.0


def test_reranker_off_bypasses_adapter_and_marks_stage_skipped() -> None:
    reranker = FakeReranker()
    result = make_service(reranker).search(
        question="Qdrant ne işe yarar?",
        mode=RetrievalMode.HYBRID,
        top_k=2,
        reranker_enabled=False,
    )

    assert reranker.seen_count == 0
    assert result.reranker_enabled is False
    assert result.reranker_skipped_reason == "configuration"
    assert result.rerank_limit == 0
    assert all(item.rerank_rank is None for item in result.debug_candidates)


def test_rank_trace_retains_fusion_rank_after_reranker_moves_candidates() -> None:
    result = make_service(FakeReranker()).search(
        question="Qdrant ne işe yarar?",
        mode=RetrievalMode.HYBRID,
        top_k=2,
        reranker_enabled=True,
    )

    by_source = {item.source_id: item for item in result.debug_candidates}
    assert by_source["shared"].fusion_rank == 1
    assert by_source["sparse-only"].fusion_rank == 3
    assert by_source["sparse-only"].rank_delta == -2


def test_live_candidate_trace_is_emitted_after_application_access_filter() -> None:
    class LeakyRetriever(FakeRetriever):
        def search_dense(
            self,
            *,
            query_vector: Sequence[float],
            limit: int,
            document_ids: Sequence[str],
            tenant_id: str = "default",
            acl_tags: Sequence[str] = ("public",),
        ) -> tuple[RetrievedChunk, ...]:
            del query_vector, limit, document_ids, tenant_id, acl_tags
            return (
                candidate("authorized"),
                replace(candidate("wrong-tenant"), tenant_id="other-tenant"),
                replace(candidate("wrong-document"), document_id="outside-scope"),
            )

    events: list[tuple[str, str, dict[str, object] | None]] = []
    service = RetrievalService(
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        retriever=LeakyRetriever(),
        reranker_default_enabled=False,
    )

    service.search(
        question="Qdrant ne işe yarar?",
        mode=RetrievalMode.DENSE,
        top_k=2,
        document_ids=("doc-1",),
        tenant_id="default",
        acl_tags=("public",),
        trace=lambda stage, status, _summary, details, _duration: events.append(
            (stage, status, details)
        ),
    )

    dense_pass = cast(dict[str, object], next(
        details
        for stage, status, details in events
        if stage == "dense_retrieval" and status == "passed"
    ))
    assert dense_pass["count"] == 1
    assert dense_pass["distribution"] == [
        {"document_id": "doc-1", "title": "RAG", "count": 1}
    ]
    candidates = cast(list[dict[str, object]], dense_pass["candidates"])
    assert candidates[0]["source_id"] == "authorized"
    assert candidates[0]["dense_rank"] == 1
