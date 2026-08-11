"""Structured, privacy-conscious query trace events."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from threading import Lock
from time import monotonic
from typing import Protocol
from uuid import uuid4

from ..domain.evidence_validation import EvidenceWarning
from ..domain.entities import Decision, NoAnswerReason
from ..domain.retrieval import RetrievalResult
from .request_id import get_request_id


@dataclass(frozen=True, slots=True)
class QueryTraceEvent:
    """One JSON-serializable event for a completed RAG query."""

    event: str
    request_id: str
    question_sha256: str
    decision: Decision
    no_answer_reason: NoAnswerReason | None
    retrieval_mode: str
    dense_candidates: int
    sparse_candidates: int
    rrf_candidates: int
    reranked_candidates: int
    selected_evidence_count: int
    top_score: float | None
    score_margin: float | None
    coverage_ratio: float
    provider: str | None
    model: str | None
    warning_codes: tuple[str, ...]
    embedding_ms: float
    search_ms: float
    rerank_ms: float
    llm_ms: float
    total_ms: float
    candidate_limit: int = 0
    fusion_limit: int = 0
    rerank_limit: int = 0
    reranker_enabled: bool = False
    reranker_skipped_reason: str | None = None
    dense_model: str | None = None
    sparse_model: str | None = None
    reranker_model: str | None = None
    dense_distribution: tuple[dict[str, object], ...] = ()
    sparse_distribution: tuple[dict[str, object], ...] = ()
    policy_profile: str = "default"
    calibration_id: str | None = None
    score_threshold: float | None = None
    coverage_threshold: float = 0.0
    required_qualifiers: tuple[str, ...] = ()
    matched_qualifiers: tuple[str, ...] = ()
    missing_qualifiers: tuple[str, ...] = ()
    qualifier_coverage_satisfied: bool = True

    @classmethod
    def from_query_result(
        cls,
        *,
        question: str,
        decision: Decision,
        no_answer_reason: NoAnswerReason | None,
        retrieval: RetrievalResult,
        selected_evidence_count: int,
        top_score: float | None,
        score_margin: float | None,
        coverage_ratio: float,
        provider: str | None,
        model: str | None,
        warnings: tuple[EvidenceWarning, ...],
        llm_ms: float,
        total_ms: float,
        policy_profile: str = "default",
        calibration_id: str | None = None,
        score_threshold: float | None = None,
        coverage_threshold: float = 0.0,
        required_qualifiers: tuple[str, ...] = (),
        matched_qualifiers: tuple[str, ...] = (),
        missing_qualifiers: tuple[str, ...] = (),
        qualifier_coverage_satisfied: bool = True,
    ) -> "QueryTraceEvent":
        """Build an event without retaining the raw user question."""

        return cls(
            event="rag_query",
            request_id=get_request_id(),
            question_sha256=hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            decision=decision,
            no_answer_reason=no_answer_reason,
            retrieval_mode=retrieval.mode,
            dense_candidates=retrieval.dense_candidates,
            sparse_candidates=retrieval.sparse_candidates,
            rrf_candidates=retrieval.rrf_candidates,
            reranked_candidates=retrieval.reranked_candidates,
            selected_evidence_count=selected_evidence_count,
            top_score=top_score,
            score_margin=score_margin,
            coverage_ratio=coverage_ratio,
            provider=provider,
            model=model,
            warning_codes=tuple(warning.code.value for warning in warnings),
            embedding_ms=retrieval.embedding_ms,
            search_ms=retrieval.search_ms,
            rerank_ms=retrieval.rerank_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            candidate_limit=retrieval.candidate_limit,
            fusion_limit=retrieval.fusion_limit,
            rerank_limit=retrieval.rerank_limit,
            reranker_enabled=retrieval.reranker_enabled,
            reranker_skipped_reason=retrieval.reranker_skipped_reason,
            dense_model=retrieval.dense_model,
            sparse_model=retrieval.sparse_model,
            reranker_model=retrieval.reranker_model,
            dense_distribution=tuple(
                {
                    "document_id": item.document_id,
                    "count": item.count,
                }
                for item in retrieval.dense_distribution
            ),
            sparse_distribution=tuple(
                {
                    "document_id": item.document_id,
                    "count": item.count,
                }
                for item in retrieval.sparse_distribution
            ),
            policy_profile=policy_profile,
            calibration_id=calibration_id,
            score_threshold=score_threshold,
            coverage_threshold=coverage_threshold,
            required_qualifiers=required_qualifiers,
            matched_qualifiers=matched_qualifiers,
            missing_qualifiers=missing_qualifiers,
            qualifier_coverage_satisfied=qualifier_coverage_satisfied,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe mapping for log sinks and tests."""

        return {
            "event": self.event,
            "request_id": self.request_id,
            "question_sha256": self.question_sha256,
            "decision": self.decision.value,
            "no_answer_reason": (
                self.no_answer_reason.value
                if self.no_answer_reason is not None
                else None
            ),
            "retrieval": {
                "mode": self.retrieval_mode,
                "dense_candidates": self.dense_candidates,
                "sparse_candidates": self.sparse_candidates,
                "rrf_candidates": self.rrf_candidates,
                "reranked_candidates": self.reranked_candidates,
                "selected_evidence_count": self.selected_evidence_count,
                "candidate_limit": self.candidate_limit,
                "fusion_limit": self.fusion_limit,
                "rerank_limit": self.rerank_limit,
                "reranker_enabled": self.reranker_enabled,
                "reranker_skipped_reason": self.reranker_skipped_reason,
                "dense_model": self.dense_model,
                "sparse_model": self.sparse_model,
                "reranker_model": self.reranker_model,
                "dense_distribution": list(self.dense_distribution),
                "sparse_distribution": list(self.sparse_distribution),
            },
            "answerability": {
                "top_score": self.top_score,
                "score_margin": self.score_margin,
                "coverage_ratio": self.coverage_ratio,
                "policy_profile": self.policy_profile,
                "calibration_id": self.calibration_id,
                "score_threshold": self.score_threshold,
                "coverage_threshold": self.coverage_threshold,
                "required_qualifiers": list(self.required_qualifiers),
                "matched_qualifiers": list(self.matched_qualifiers),
                "missing_qualifiers": list(self.missing_qualifiers),
                "qualifier_coverage_satisfied": (
                    self.qualifier_coverage_satisfied
                ),
            },
            "model": {"provider": self.provider, "name": self.model},
            "warning_codes": list(self.warning_codes),
            "latency_ms": {
                "embedding": self.embedding_ms,
                "search": self.search_ms,
                "rerank": self.rerank_ms,
                "llm": self.llm_ms,
                "total": self.total_ms,
            },
            "spans": [
                {"name": "embed", "duration_ms": self.embedding_ms},
                {"name": "search", "duration_ms": self.search_ms},
                {"name": "rerank", "duration_ms": self.rerank_ms},
                {"name": "llm", "duration_ms": self.llm_ms},
            ],
        }


class QueryTraceSink(Protocol):
    """Application boundary for query observability sinks."""

    def record(self, event: QueryTraceEvent) -> None:
        """Persist or forward one completed query event."""


class JsonQueryTraceSink:
    """Write one compact JSON object per query to a standard logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(
            "document_intelligence_service.query"
        )

    def record(self, event: QueryTraceEvent) -> None:
        """Emit structured JSON without logging the raw question or evidence."""

        self._logger.info(
            "%s",
            json.dumps(
                event.as_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )


@dataclass(frozen=True, slots=True)
class LiveTraceEvent:
    """One bounded event exposed only through the development demo transport."""

    sequence: int
    timestamp: str
    stage: str
    status: str
    summary: str
    duration_ms: float | None
    details: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return the frontend-safe event shape."""

        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "status": self.status,
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


class LiveTraceRecorder:
    """Small callback object passed through the real query use-case."""

    def __init__(self, store: "LiveQueryTraceStore", run_id: str) -> None:
        self._store = store
        self._run_id = run_id

    def emit(
        self,
        stage: str,
        status: str,
        summary: str,
        details: dict[str, object] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Append an event without exposing a raw prompt or full chunk."""

        self._store.emit(
            self._run_id,
            stage=stage,
            status=status,
            summary=summary,
            details=details,
            duration_ms=duration_ms,
        )


@dataclass(slots=True)
class _LiveTraceRun:
    run_id: str
    request_id: str
    status: str
    created_at: str
    started_at: str
    finished_at: str | None
    events: list[LiveTraceEvent]
    result: dict[str, object] | None
    runtime_result: object | None
    error: dict[str, object] | None
    last_access: float


class LiveQueryTraceStore:
    """In-memory, bounded query-run store for the local Demo UI.

    It is intentionally separate from the canonical `/v1/queries` response.
    The same `QueryService` executes the request; this store only transports
    progress and a compact final projection for a development/demo client.
    """

    def __init__(self, *, ttl_seconds: float = 900.0, max_runs: int = 32) -> None:
        if ttl_seconds <= 0 or max_runs <= 0:
            raise ValueError("trace store limits must be greater than zero")
        self._ttl_seconds = ttl_seconds
        self._max_runs = max_runs
        self._runs: dict[str, _LiveTraceRun] = {}
        self._lock = Lock()

    def create(self, *, request_id: str) -> str:
        """Create one bounded run and return a non-sensitive run identifier."""

        now = datetime.now(timezone.utc).isoformat()
        run_id = f"trace_{uuid4().hex}"
        with self._lock:
            self._cleanup_locked()
            self._runs[run_id] = _LiveTraceRun(
                run_id=run_id,
                request_id=request_id,
                status="pending",
                created_at=now,
                started_at=now,
                finished_at=None,
                events=[],
                result=None,
                runtime_result=None,
                error=None,
                last_access=monotonic(),
            )
            while len(self._runs) > self._max_runs:
                oldest = min(self._runs, key=lambda key: self._runs[key].last_access)
                del self._runs[oldest]
        return run_id

    def recorder(self, run_id: str) -> LiveTraceRecorder:
        """Return the event callback for one existing run."""

        self._require(run_id)
        return LiveTraceRecorder(self, run_id)

    def emit(
        self,
        run_id: str,
        *,
        stage: str,
        status: str,
        summary: str,
        details: dict[str, object] | None,
        duration_ms: float | None,
    ) -> None:
        """Append one event and make the run visible to polling clients."""

        safe_details = _sanitize_trace_details(details or {})
        event = LiveTraceEvent(
            sequence=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            status=status,
            summary=summary,
            duration_ms=duration_ms,
            details=safe_details,
        )
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = "running" if status in {"running", "pending"} else run.status
            run.events.append(
                LiveTraceEvent(
                    sequence=len(run.events) + 1,
                    timestamp=event.timestamp,
                    stage=event.stage,
                    status=event.status,
                    summary=event.summary,
                    duration_ms=event.duration_ms,
                    details=event.details,
                )
            )
            run.last_access = monotonic()

    def finish(
        self,
        run_id: str,
        result: dict[str, object],
        *,
        runtime_result: object | None = None,
    ) -> None:
        """Publish a compact final result after the application use-case ends."""

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc).isoformat()
            run.result = _sanitize_trace_details(result)
            run.runtime_result = runtime_result
            run.last_access = monotonic()

    def runtime_result(self, run_id: str) -> object | None:
        """Return the in-process result for a post-run diagnostic comparison.

        This is deliberately not part of the JSON snapshot.  The trusted
        evidence picker compares the existing run without re-running retrieval
        or generation, while the public/demo transport still exposes only the
        bounded serialized projection.
        """

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            run.last_access = monotonic()
            return run.runtime_result

    def merge_result(self, run_id: str, updates: dict[str, object]) -> None:
        """Merge bounded post-run diagnostic metadata into a completed result."""

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.result is None:
                raise ValueError("query run has no completed result")
            run.result.update(_sanitize_trace_details(updates))
            run.last_access = monotonic()

    def fail(self, run_id: str, error: dict[str, object]) -> None:
        """Publish a safe terminal error without stack traces or host paths."""

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.status = "failed"
            run.finished_at = datetime.now(timezone.utc).isoformat()
            run.error = _sanitize_trace_details(error)
            run.last_access = monotonic()

    def snapshot(self, run_id: str) -> dict[str, object]:
        """Return an immutable JSON-safe snapshot for polling."""

        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            run.last_access = monotonic()
            return {
                "run_id": run.run_id,
                "request_id": run.request_id,
                "status": run.status,
                "created_at": run.created_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "events": [event.as_dict() for event in run.events],
                "result": run.result,
                "error": run.error,
            }

    def _require(self, run_id: str) -> None:
        with self._lock:
            self._cleanup_locked()
            if run_id not in self._runs:
                raise KeyError(run_id)

    def _cleanup_locked(self) -> None:
        cutoff = monotonic() - self._ttl_seconds
        expired = [
            run_id
            for run_id, run in self._runs.items()
            if run.last_access < cutoff
        ]
        for run_id in expired:
            del self._runs[run_id]


def _sanitize_trace_details(details: dict[str, object]) -> dict[str, object]:
    """Bound recursive values so demo traces remain readable and safe."""

    def convert(value: object, depth: int = 0, key: str | None = None) -> object:
        if depth > 3:
            return "[truncated]"
        if isinstance(value, str):
            # The normal trace budget remains 500 characters. The local
            # evidence inspector is allowed to reveal the already-bounded
            # child/parent evidence fields returned by the demo projection.
            limit = 4000 if key in {"chunk_text", "parent_context", "included_text"} else 500
            return value[:limit]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, dict):
            return {
                str(child_key)[:80]: convert(item, depth + 1, str(child_key))
                for child_key, item in list(value.items())[:40]
            }
        if isinstance(value, (list, tuple)):
            return [convert(item, depth + 1, key) for item in value[:40]]
        return str(value)[:200]

    return {
        str(key)[:80]: convert(value, key=str(key))
        for key, value in list(details.items())[:40]
    }
