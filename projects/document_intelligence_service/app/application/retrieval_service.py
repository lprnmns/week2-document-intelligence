"""Dense, sparse and hybrid retrieval orchestration."""

from dataclasses import dataclass, replace
from collections.abc import Callable, Sequence
import inspect
import re
from time import perf_counter
from typing import cast

from ..domain.entities import RetrievalMode
from ..domain.ingestion import normalize_acl_tags, normalize_tenant_id
from ..domain.retrieval import (
    RetrievedChunk,
    DocumentCandidateDistribution,
    RetrievalDebugCandidate,
    RetrievalResult,
)
from ..domain.vectors import SparseVector
from .ports import ChunkRetriever, DenseEmbedder, Reranker, SparseEmbedder

TraceCallback = Callable[
    [str, str, str, dict[str, object] | None, float | None],
    None,
]


@dataclass(slots=True)
class _FusionEntry:
    """Mutable application-local state while rank lists are fused."""

    candidate: RetrievedChunk
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None


class RetrievalService:
    """Coordinate query encoding, bounded candidate search and RRF fusion."""

    def __init__(
        self,
        *,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        retriever: ChunkRetriever,
        candidate_limit: int = 30,
        rrf_k: int = 60,
        fusion_limit: int = 20,
        reranker: Reranker | None = None,
        rerank_limit: int = 5,
        reranker_default_enabled: bool | None = None,
        dense_model: str | None = None,
        sparse_model: str | None = None,
        reranker_model: str | None = None,
    ) -> None:
        if (
            candidate_limit <= 0
            or rrf_k <= 0
            or fusion_limit <= 0
            or rerank_limit <= 0
        ):
            raise ValueError("retrieval limits must be greater than zero")
        self._dense_embedder = dense_embedder
        self._sparse_embedder = sparse_embedder
        self._retriever = retriever
        self._candidate_limit = min(candidate_limit, 50)
        self._rrf_k = rrf_k
        self._fusion_limit = min(fusion_limit, 50)
        self._reranker = reranker
        self._rerank_limit = min(rerank_limit, self._fusion_limit)
        self._reranker_default_enabled = (
            reranker is not None
            if reranker_default_enabled is None
            else reranker_default_enabled
        )
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._reranker_model = reranker_model

    def warmup(self) -> None:
        """Load model adapters before the first query when configured."""

        warmup_dense = getattr(self._dense_embedder, "warmup", None)
        if callable(warmup_dense):
            warmup_dense()
        if self._reranker_default_enabled:
            warmup_reranker = getattr(self._reranker, "warmup", None)
            if callable(warmup_reranker):
                warmup_reranker()

    def search(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[str] = (),
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        reranker_enabled: bool | None = None,
        trace: TraceCallback | None = None,
    ) -> RetrievalResult:
        """Search active evidence with one of the three supported modes."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        normalized_tenant = normalize_tenant_id(tenant_id)
        normalized_acl = normalize_acl_tags(tuple(acl_tags))
        active_version_ids = self._snapshot_active_version_ids(
            document_ids,
            normalized_tenant,
        )
        use_reranker = (
            self._reranker_default_enabled
            if reranker_enabled is None
            else reranker_enabled
        )
        if use_reranker and self._reranker is None:
            raise ValueError("reranker is not configured")

        embedding_started = perf_counter()
        dense_vector: tuple[float, ...] | None = None
        sparse_vector: SparseVector | None = None
        if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
            dense_vector = self._one_dense_vector(normalized_question)
        if mode in (RetrievalMode.BM25, RetrievalMode.HYBRID):
            sparse_vector = self._one_sparse_vector(normalized_question)
        embedding_ms = (perf_counter() - embedding_started) * 1000
        _emit_trace(
            trace,
            "query_representation",
            "passed",
            "Dense and lexical query representations prepared",
            {
                "dense": dense_vector is not None,
                "sparse": sparse_vector is not None,
                "dense_model": self._dense_model,
                "sparse_model": self._sparse_model,
                "dense_dimension": len(dense_vector) if dense_vector is not None else None,
                "sparse_terms": len(sparse_vector.indices) if sparse_vector is not None else None,
            },
            embedding_ms,
        )

        search_started = perf_counter()
        limit = min(max(self._candidate_limit, top_k), 50)
        dense_candidates: tuple[RetrievedChunk, ...] = ()
        sparse_candidates: tuple[RetrievedChunk, ...] = ()
        if dense_vector is not None:
            dense_started = perf_counter()
            _emit_trace(
                trace,
                "dense_retrieval",
                "running",
                "Dense candidates are being searched inside the normalized scope",
                {"limit": limit},
                None,
            )
            try:
                dense_candidates = self._invoke_retriever(
                    self._retriever.search_dense,
                    {
                        "query_vector": dense_vector,
                        "limit": limit,
                        "document_ids": document_ids,
                        "tenant_id": normalized_tenant,
                        "acl_tags": normalized_acl,
                    },
                    active_version_ids,
                )
            except Exception:
                _emit_trace(
                    trace,
                    "dense_retrieval",
                    "failed",
                    "Dense retrieval adapter failed",
                    {"reason": "adapter_unavailable"},
                    (perf_counter() - dense_started) * 1000,
                )
                raise
            # The adapter must pre-filter, but the application boundary also
            # re-checks the returned source metadata before publishing any
            # candidate details to the live trace.
            dense_candidates = self._filter_access(
                dense_candidates,
                document_ids=document_ids,
                tenant_id=normalized_tenant,
                acl_tags=normalized_acl,
            )
            _emit_trace(
                trace,
                "dense_retrieval",
                "passed",
                f"Dense returned {len(dense_candidates)} authorized candidates",
                {
                    "count": len(dense_candidates),
                    "distribution": _distribution_payload(dense_candidates),
                    "candidates": _candidate_trace_payload(
                        dense_candidates,
                        fallback_rank_field="dense_rank",
                    ),
                },
                (perf_counter() - dense_started) * 1000,
            )
        else:
            _emit_trace(
                trace,
                "dense_retrieval",
                "skipped",
                "Dense retrieval skipped by retrieval mode",
                {"mode": mode.value},
                0.0,
            )
        if sparse_vector is not None:
            sparse_started = perf_counter()
            _emit_trace(
                trace,
                "sparse_retrieval",
                "running",
                "BM25/sparse candidates are being searched inside the normalized scope",
                {"limit": limit},
                None,
            )
            try:
                sparse_candidates = self._invoke_retriever(
                    self._retriever.search_sparse,
                    {
                        "query_vector": sparse_vector,
                        "limit": limit,
                        "document_ids": document_ids,
                        "tenant_id": normalized_tenant,
                        "acl_tags": normalized_acl,
                    },
                    active_version_ids,
                )
            except Exception:
                _emit_trace(
                    trace,
                    "sparse_retrieval",
                    "failed",
                    "BM25/sparse retrieval adapter failed",
                    {"reason": "adapter_unavailable"},
                    (perf_counter() - sparse_started) * 1000,
                )
                raise
            sparse_candidates = self._filter_access(
                sparse_candidates,
                document_ids=document_ids,
                tenant_id=normalized_tenant,
                acl_tags=normalized_acl,
            )
            _emit_trace(
                trace,
                "sparse_retrieval",
                "passed",
                f"BM25 returned {len(sparse_candidates)} authorized candidates",
                {
                    "count": len(sparse_candidates),
                    "distribution": _distribution_payload(sparse_candidates),
                    "candidates": _candidate_trace_payload(
                        sparse_candidates,
                        fallback_rank_field="sparse_rank",
                    ),
                },
                (perf_counter() - sparse_started) * 1000,
            )
        else:
            _emit_trace(
                trace,
                "sparse_retrieval",
                "skipped",
                "BM25 retrieval skipped by retrieval mode",
                {"mode": mode.value},
                0.0,
            )

        if mode is RetrievalMode.HYBRID:
            candidates, rrf_count = self._fuse(
                dense_candidates,
                sparse_candidates,
                top_k=top_k,
            )
            _emit_trace(
                trace,
                "rrf_fusion",
                "passed",
                f"RRF fused {rrf_count} unique candidates; kept {len(candidates)}",
                {
                    "rrf_k": self._rrf_k,
                    "input_dense": len(dense_candidates),
                    "input_sparse": len(sparse_candidates),
                    "output": len(candidates),
                    "candidates": _candidate_trace_payload(candidates),
                },
                None,
            )
        elif mode is RetrievalMode.DENSE:
            window = self._candidate_window(top_k)
            candidates = tuple(
                replace(
                    candidate,
                    rank=index,
                    fusion_rank=index,
                    dense_score=(
                        candidate.dense_score
                        if candidate.dense_score is not None
                        else candidate.score
                    ),
                )
                for index, candidate in enumerate(dense_candidates[:window], start=1)
            )
            rrf_count = 0
            _emit_trace(
                trace,
                "rrf_fusion",
                "skipped",
                "RRF is not applicable in Dense-only mode",
                {"mode": mode.value},
                0.0,
            )
        else:
            window = self._candidate_window(top_k)
            candidates = tuple(
                replace(
                    candidate,
                    rank=index,
                    fusion_rank=index,
                    sparse_score=(
                        candidate.sparse_score
                        if candidate.sparse_score is not None
                        else candidate.score
                    ),
                )
                for index, candidate in enumerate(sparse_candidates[:window], start=1)
            )
            rrf_count = 0
            _emit_trace(
                trace,
                "rrf_fusion",
                "skipped",
                "RRF is not applicable in BM25-only mode",
                {"mode": mode.value},
                0.0,
            )
        search_ms = (perf_counter() - search_started) * 1000
        candidate_window = candidates
        reranked_count = 0
        rerank_ms = 0.0
        if use_reranker and candidates:
            _emit_trace(
                trace,
                "reranker",
                "running",
                "Bounded reranker is scoring the fusion candidate window",
                {
                    "candidate_input": len(candidate_window),
                    "candidate_limit": self._rerank_limit,
                    "model": self._reranker_model,
                },
                None,
            )
            rerank_started = perf_counter()
            assert self._reranker is not None
            candidates = self._reranker.rerank(
                question=normalized_question,
                candidates=candidates,
                limit=min(top_k, self._rerank_limit),
            )
            rerank_ms = (perf_counter() - rerank_started) * 1000
            reranked_count = len(candidates)
            _emit_trace(
                trace,
                "reranker",
                "passed",
                f"Reranker returned {len(candidates)} evidence candidates",
                {
                    "candidate_input": len(candidate_window),
                    "final_output": len(candidates),
                    "model": self._reranker_model,
                    "before": _candidate_trace_payload(candidate_window),
                    "after": _candidate_trace_payload(candidates),
                },
                rerank_ms,
            )
        else:
            candidates = candidates[:top_k]
            _emit_trace(
                trace,
                "reranker",
                "skipped",
                "Reranker skipped by configuration; fusion/retriever order used directly",
                {
                    "reason": "configuration",
                    "candidate_input": len(candidate_window),
                    "final_output": len(candidates),
                    "candidates": _candidate_trace_payload(candidates),
                },
                0.0,
            )

        debug_candidates = self._build_debug_candidates(
            normalized_question,
            dense_candidates,
            sparse_candidates,
            candidate_window,
            candidates,
            reranker_enabled=use_reranker,
        )

        return RetrievalResult(
            mode=mode.value,
            candidates=candidates,
            dense_candidates=len(dense_candidates),
            sparse_candidates=len(sparse_candidates),
            rrf_candidates=rrf_count,
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            reranked_candidates=reranked_count,
            rerank_ms=rerank_ms,
            candidate_window=candidate_window,
            debug_candidates=debug_candidates,
            candidate_limit=limit,
            fusion_limit=self._fusion_limit if mode is RetrievalMode.HYBRID else 0,
            rerank_limit=self._rerank_limit if use_reranker else 0,
            reranker_enabled=use_reranker,
            reranker_skipped_reason=None if use_reranker else "configuration",
            dense_distribution=_distribution(dense_candidates),
            sparse_distribution=_distribution(sparse_candidates),
            dense_model=self._dense_model,
            sparse_model=self._sparse_model,
            reranker_model=self._reranker_model if use_reranker else None,
        )

    @staticmethod
    def _invoke_retriever(
        method: Callable[..., tuple[RetrievedChunk, ...]],
        kwargs: dict[str, object],
        active_version_ids: tuple[str, ...] | None,
    ) -> tuple[RetrievedChunk, ...]:
        """Pass a captured version scope only to adapters that support it.

        Small test/demonstration adapters from earlier releases remain valid;
        the production Qdrant adapter opts into the additional boundary.
        """

        if active_version_ids is not None and "active_version_ids" in inspect.signature(method).parameters:
            kwargs["active_version_ids"] = active_version_ids
        return method(**kwargs)

    def _snapshot_active_version_ids(
        self,
        document_ids: Sequence[str],
        tenant_id: str,
    ) -> tuple[str, ...] | None:
        """Capture one registry scope for both Dense and BM25 branches."""

        snapshot = getattr(self._retriever, "snapshot_active_version_ids", None)
        if not callable(snapshot):
            return None
        resolved = snapshot(document_ids, tenant_id)
        if resolved is None:
            return None
        return tuple(dict.fromkeys(item for item in resolved if item))

    @staticmethod
    def _filter_access(
        candidates: Sequence[RetrievedChunk],
        *,
        document_ids: Sequence[str],
        tenant_id: str,
        acl_tags: Sequence[str],
    ) -> tuple[RetrievedChunk, ...]:
        """Re-check tenant, ACL and document scope on adapter results."""

        requested_documents = {item for item in document_ids if item}
        allowed_tags = {"public", *acl_tags}
        return tuple(
            candidate
            for candidate in candidates
            if candidate.tenant_id == tenant_id
            and (not requested_documents or candidate.document_id in requested_documents)
            and bool(set(candidate.acl_tags) & allowed_tags)
        )

    @staticmethod
    def _build_debug_candidates(
        question: str,
        dense_candidates: Sequence[RetrievedChunk],
        sparse_candidates: Sequence[RetrievedChunk],
        candidate_window: Sequence[RetrievedChunk],
        final_candidates: Sequence[RetrievedChunk],
        *,
        reranker_enabled: bool,
    ) -> tuple[RetrievalDebugCandidate, ...]:
        """Build bounded rank diagnostics for all participating stages."""

        question_terms = {
            token.casefold()
            for token in re.findall(r"\w+", question, flags=re.UNICODE)
            if len(token) >= 3
        }
        dense_by_source = {
            item.source_id: (rank, item)
            for rank, item in enumerate(dense_candidates, start=1)
        }
        sparse_by_source = {
            item.source_id: (rank, item)
            for rank, item in enumerate(sparse_candidates, start=1)
        }
        fusion_by_source = {item.source_id: item for item in candidate_window}
        final_by_source = {item.source_id: item for item in final_candidates}
        ordered_ids: list[str] = []
        for candidate in candidate_window:
            if candidate.source_id not in ordered_ids:
                ordered_ids.append(candidate.source_id)
        for candidate in (*dense_candidates, *sparse_candidates):
            if candidate.source_id not in ordered_ids:
                ordered_ids.append(candidate.source_id)
        debug: list[RetrievalDebugCandidate] = []
        for source_id in ordered_ids:
            item: RetrievedChunk | None = fusion_by_source.get(source_id)
            if item is None and source_id in dense_by_source:
                item = dense_by_source[source_id][1]
            if item is None and source_id in sparse_by_source:
                item = sparse_by_source[source_id][1]
            if item is None:
                continue
            final = final_by_source.get(source_id)
            rank_entry = dense_by_source.get(source_id) or sparse_by_source.get(source_id)
            retrieval_rank = (
                fusion_by_source[source_id].rank
                if source_id in fusion_by_source
                else rank_entry[0] if rank_entry is not None else None
            )
            evidence_terms = {
                token.casefold()
                for token in re.findall(r"\w+", item.context_text, flags=re.UNICODE)
                if len(token) >= 3
            }
            debug.append(
                RetrievalDebugCandidate(
                    source_id=item.source_id,
                    retrieval_rank=retrieval_rank,
                    rerank_rank=(
                        final.rank if final is not None and reranker_enabled else None
                    ),
                    dense_rank=(
                        dense_by_source[source_id][0]
                        if source_id in dense_by_source
                        else None
                    ),
                    sparse_rank=(
                        sparse_by_source[source_id][0]
                        if source_id in sparse_by_source
                        else None
                    ),
                    dense_score=(
                        dense_by_source[source_id][1].dense_score
                        if source_id in dense_by_source
                        else None
                    ),
                    sparse_score=(
                        sparse_by_source[source_id][1].sparse_score
                        if source_id in sparse_by_source
                        else None
                    ),
                    fused_score=(
                        fusion_by_source[source_id].fused_score
                        if source_id in fusion_by_source
                        else None
                    ),
                    rerank_score=final.rerank_score if final is not None else None,
                    matched_terms=tuple(sorted(question_terms & evidence_terms)),
                    document_id=item.document_id,
                    title=item.title,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    excerpt=_compact_excerpt(item.context_text),
                    fusion_rank=(
                        fusion_by_source[source_id].rank
                        if source_id in fusion_by_source
                        else None
                    ),
                    selected_as_evidence=final is not None,
                    rank_delta=(
                        final.rank - fusion_by_source[source_id].rank
                        if final is not None and reranker_enabled and source_id in fusion_by_source
                        else None
                    ),
                )
            )
        return tuple(debug)

    def _candidate_window(self, top_k: int) -> int:
        """Keep a bounded candidate window separate from final top-k."""

        return min(max(top_k, self._fusion_limit), 50)

    def _one_dense_vector(self, question: str) -> tuple[float, ...]:
        vectors = self._dense_embedder.embed_documents((question,))
        if len(vectors) != 1:
            raise ValueError("dense embedder returned an unexpected query batch")
        return vectors[0]

    def _one_sparse_vector(self, question: str) -> SparseVector:
        embed_query = getattr(self._sparse_embedder, "embed_query", None)
        if callable(embed_query):
            return cast(SparseVector, embed_query(question))
        vectors = self._sparse_embedder.embed_documents((question,))
        if len(vectors) != 1:
            raise ValueError("sparse embedder returned an unexpected query batch")
        return vectors[0]

    def _fuse(
        self,
        dense_candidates: Sequence[RetrievedChunk],
        sparse_candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> tuple[tuple[RetrievedChunk, ...], int]:
        entries: dict[str, _FusionEntry] = {}
        for rank, candidate in enumerate(dense_candidates, start=1):
            entry = entries.setdefault(
                candidate.source_id,
                _FusionEntry(candidate=candidate),
            )
            entry.dense_rank = rank
            entry.dense_score = (
                candidate.dense_score
                if candidate.dense_score is not None
                else candidate.score
            )
            entry.fused_score += 1.0 / (self._rrf_k + rank)
        for rank, candidate in enumerate(sparse_candidates, start=1):
            entry = entries.setdefault(
                candidate.source_id,
                _FusionEntry(candidate=candidate),
            )
            entry.sparse_rank = rank
            entry.sparse_score = (
                candidate.sparse_score
                if candidate.sparse_score is not None
                else candidate.score
            )
            entry.fused_score += 1.0 / (self._rrf_k + rank)

        limit = self._candidate_window(top_k)
        ordered = sorted(
            entries.values(),
            key=lambda entry: (-entry.fused_score, entry.candidate.source_id),
        )[:limit]
        fused = tuple(
            replace(
                entry.candidate,
                score=entry.fused_score,
                fused_score=entry.fused_score,
                rank=index,
                fusion_rank=index,
                dense_rank=entry.dense_rank,
                sparse_rank=entry.sparse_rank,
                dense_score=entry.dense_score,
                sparse_score=entry.sparse_score,
            )
            for index, entry in enumerate(ordered, start=1)
        )
        return fused, len(entries)


def _emit_trace(
    trace: TraceCallback | None,
    stage: str,
    status: str,
    summary: str,
    details: dict[str, object] | None,
    duration_ms: float | None,
) -> None:
    """Forward a development trace without allowing it to break retrieval."""

    if trace is None:
        return
    try:
        trace(stage, status, summary, details, duration_ms)
    except Exception:
        # Observability is deliberately best effort at this boundary. A broken
        # demo sink must never change retrieval correctness or authorization.
        return


def _distribution(
    candidates: Sequence[RetrievedChunk],
) -> tuple[DocumentCandidateDistribution, ...]:
    """Count candidates by document while retaining a human-readable title."""

    counts: dict[str, tuple[str, int]] = {}
    for candidate in candidates:
        title, count = counts.get(candidate.document_id, (candidate.title, 0))
        counts[candidate.document_id] = (title, count + 1)
    return tuple(
        DocumentCandidateDistribution(
            document_id=document_id,
            title=title,
            count=count,
        )
        for document_id, (title, count) in sorted(
            counts.items(),
            key=lambda item: (-item[1][1], item[0]),
        )
    )


def _distribution_payload(
    candidates: Sequence[RetrievedChunk],
) -> list[dict[str, object]]:
    """Return JSON-safe document counts for live trace details."""

    return [
        {
            "document_id": item.document_id,
            "title": item.title,
            "count": item.count,
        }
        for item in _distribution(candidates)
    ]


def _candidate_trace_payload(
    candidates: Sequence[RetrievedChunk],
    *,
    limit: int = 20,
    fallback_rank_field: str | None = None,
) -> list[dict[str, object]]:
    """Return bounded rank diagnostics for the live demo trace.

    This projection intentionally contains no full chunk, parent context or
    vector.  The application owns the ranking data; the demo transport only
    receives a compact, development-controlled view of it.
    """

    return [
        {
            "source_id": item.source_id,
            "document_id": item.document_id,
            "title": item.title,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "chunking_profile": item.chunking_profile,
            "excerpt": _compact_excerpt(item.context_text, limit=180),
            "rank": item.rank,
            "dense_rank": (
                item.dense_rank
                if item.dense_rank is not None or fallback_rank_field != "dense_rank"
                else index
            ),
            "sparse_rank": (
                item.sparse_rank
                if item.sparse_rank is not None or fallback_rank_field != "sparse_rank"
                else index
            ),
            "fusion_rank": item.fusion_rank,
            "rerank_score": item.rerank_score,
            "score": item.score,
            "fused_score": item.fused_score,
        }
        for index, item in enumerate(candidates[:limit], start=1)
    ]


def _compact_excerpt(text: str, limit: int = 220) -> str:
    """Bound trace excerpts so a debug page cannot become a document dump."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"
