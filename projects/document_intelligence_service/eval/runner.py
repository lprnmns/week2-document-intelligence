"""Retrieval benchmark runner for the layered document service."""

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import inspect
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from ..app.application.query_service import assess_answerability
    from ..app.domain.answerability import AnswerabilityPolicy
    from ..app.domain.entities import RetrievalMode
    from ..app.domain.retrieval import RetrievedChunk, RetrievalResult
else:  # pragma: no cover - package selection depends on the image layout
    _app_prefix = (
        "projects.document_intelligence_service.app"
        if __package__ and __package__.startswith("projects.")
        else "app"
    )
    _query_service = importlib.import_module(
        f"{_app_prefix}.application.query_service"
    )
    _answerability = importlib.import_module(f"{_app_prefix}.domain.answerability")
    _entities = importlib.import_module(f"{_app_prefix}.domain.entities")
    _retrieval = importlib.import_module(f"{_app_prefix}.domain.retrieval")
    assess_answerability = _query_service.assess_answerability
    AnswerabilityPolicy = _answerability.AnswerabilityPolicy
    RetrievalMode = _entities.RetrievalMode
    RetrievedChunk = _retrieval.RetrievedChunk
    RetrievalResult = _retrieval.RetrievalResult
from .contracts import EvidenceLike, GoldenCase
from .metrics import (
    LatencyMetrics,
    NoAnswerMetrics,
    RetrievalMetrics,
    evaluate_retrieval,
    evaluate_no_answer,
    latency_metrics,
)


class RetrievalPort(Protocol):
    """Application boundary needed by an offline retrieval benchmark."""

    def search(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[str] = (),
        reranker_enabled: bool | None = None,
    ) -> RetrievalResult:
        """Run one bounded retrieval query."""


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    """Raw per-query trace retained for error analysis."""

    case_id: str
    final_candidates: tuple[RetrievedChunk, ...]
    candidate_window: tuple[RetrievedChunk, ...]
    total_ms: float
    embedding_ms: float
    search_ms: float
    rerank_ms: float
    category: str | None = None
    status: str = "ok"
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkRun:
    """Metrics plus raw observations for one retrieval strategy."""

    strategy: str
    cases_run: int
    warmup_count: int
    metrics: RetrievalMetrics
    total_latency: LatencyMetrics
    embedding_latency: LatencyMetrics
    search_latency: LatencyMetrics
    rerank_latency: LatencyMetrics | None
    observations: tuple[RetrievalObservation, ...]


@dataclass(frozen=True, slots=True)
class AnswerabilityObservation:
    """Raw pre-LLM gate decision for one golden case."""

    case_id: str
    decision: str
    reason: str | None
    top_score: float | None
    score_margin: float | None
    coverage_ratio: float
    total_ms: float
    required_qualifiers: tuple[str, ...] = ()
    matched_qualifiers: tuple[str, ...] = ()
    missing_qualifiers: tuple[str, ...] = ()
    qualifier_coverage_satisfied: bool = True
    category: str | None = None
    status: str = "ok"
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerabilityBenchmarkRun:
    """No-answer confusion metrics and gate latency."""

    cases_run: int
    warmup_count: int
    metrics: NoAnswerMetrics
    total_latency: LatencyMetrics
    observations: tuple[AnswerabilityObservation, ...]


def run_retrieval_benchmark(
    *,
    retrieval_service: RetrievalPort,
    cases: Sequence[GoldenCase],
    mode: RetrievalMode,
    top_k: int = 5,
    warmup_questions: Sequence[str] = (),
    reranker_enabled: bool | None = None,
) -> RetrievalBenchmarkRun:
    """Run cases in a fixed order and retain stage-level raw traces.

    Warm-up questions are deliberately separate from golden cases: they warm
    model/cache state but never enter quality metrics. A caller can randomize
    strategy order outside this function while keeping query order equal.
    """

    if not cases:
        raise ValueError("benchmark needs at least one golden case")
    for question in warmup_questions:
        _search(
            retrieval_service,
            question=question,
            mode=mode,
            top_k=top_k,
            reranker_enabled=reranker_enabled,
        )

    observations: list[RetrievalObservation] = []
    final_results: dict[str, Sequence[EvidenceLike]] = {}
    candidate_results: dict[str, Sequence[EvidenceLike]] = {}
    for case in cases:
        started = perf_counter()
        try:
            result = _search(
                retrieval_service,
                question=case.question,
                mode=mode,
                top_k=top_k,
                document_ids=case.relevant_document_ids,
                reranker_enabled=reranker_enabled,
            )
            observation = RetrievalObservation(
                case_id=case.case_id,
                final_candidates=result.candidates,
                candidate_window=result.candidate_window or result.candidates,
                total_ms=(perf_counter() - started) * 1000,
                embedding_ms=result.embedding_ms,
                search_ms=result.search_ms,
                rerank_ms=result.rerank_ms,
                category=case.category,
            )
        except Exception as error:
            observation = RetrievalObservation(
                case_id=case.case_id,
                final_candidates=(),
                candidate_window=(),
                total_ms=(perf_counter() - started) * 1000,
                embedding_ms=0.0,
                search_ms=0.0,
                rerank_ms=0.0,
                category=case.category,
                status=_failure_status(error),
                error_code=type(error).__name__,
                error_message=str(error),
            )
        observations.append(observation)
        final_results[case.case_id] = cast(
            Sequence[EvidenceLike], observation.final_candidates
        )
        candidate_results[case.case_id] = cast(
            Sequence[EvidenceLike], observation.candidate_window
        )

    rerank_values = tuple(
        observation.rerank_ms
        for observation in observations
        if observation.rerank_ms > 0
    )
    return RetrievalBenchmarkRun(
        strategy=mode.value,
        cases_run=len(cases),
        warmup_count=len(warmup_questions),
        metrics=evaluate_retrieval(
            cases,
            final_results,
            candidate_results=candidate_results,
        ),
        total_latency=latency_metrics(
            tuple(observation.total_ms for observation in observations)
        ),
        embedding_latency=latency_metrics(
            tuple(observation.embedding_ms for observation in observations)
        ),
        search_latency=latency_metrics(
            tuple(observation.search_ms for observation in observations)
        ),
        rerank_latency=latency_metrics(rerank_values) if rerank_values else None,
        observations=tuple(observations),
    )


def run_answerability_benchmark(
    *,
    retrieval_service: RetrievalPort,
    answerability: AnswerabilityPolicy,
    cases: Sequence[GoldenCase],
    mode: RetrievalMode,
    top_k: int = 5,
    warmup_questions: Sequence[str] = (),
    reranker_enabled: bool | None = None,
) -> AnswerabilityBenchmarkRun:
    """Measure the live no-answer gate without calling an answer generator."""

    if not cases:
        raise ValueError("answerability benchmark needs at least one golden case")
    for question in warmup_questions:
        _search(
            retrieval_service,
            question=question,
            mode=mode,
            top_k=top_k,
            reranker_enabled=reranker_enabled,
        )

    observations: list[AnswerabilityObservation] = []
    predictions: dict[str, bool] = {}
    for case in cases:
        started = perf_counter()
        try:
            retrieval = _search(
                retrieval_service,
                question=case.question,
                mode=mode,
                top_k=top_k,
                document_ids=case.relevant_document_ids,
                reranker_enabled=reranker_enabled,
            )
            decision = assess_answerability(
                question=case.question,
                retrieval=retrieval,
                answerability=answerability,
            )
            predictions[case.case_id] = decision.decision.value == "answered"
            observations.append(
                AnswerabilityObservation(
                    case_id=case.case_id,
                    decision=decision.decision.value,
                    reason=decision.reason.value if decision.reason is not None else None,
                    top_score=decision.top_score,
                    score_margin=decision.score_margin,
                    coverage_ratio=decision.coverage_ratio,
                    total_ms=(perf_counter() - started) * 1000,
                    required_qualifiers=decision.required_qualifiers,
                    matched_qualifiers=decision.matched_qualifiers,
                    missing_qualifiers=decision.missing_qualifiers,
                    qualifier_coverage_satisfied=(
                        decision.qualifier_coverage_satisfied
                    ),
                    category=case.category,
                )
            )
        except Exception as error:
            predictions[case.case_id] = False
            observations.append(
                AnswerabilityObservation(
                    case_id=case.case_id,
                    decision="error",
                    reason=None,
                    top_score=None,
                    score_margin=None,
                    coverage_ratio=0.0,
                    total_ms=(perf_counter() - started) * 1000,
                    category=case.category,
                    status=_failure_status(error),
                    error_code=type(error).__name__,
                    error_message=str(error),
                )
            )
    return AnswerabilityBenchmarkRun(
        cases_run=len(cases),
        warmup_count=len(warmup_questions),
        metrics=evaluate_no_answer(cases, predictions),
        total_latency=latency_metrics(
            tuple(observation.total_ms for observation in observations)
        ),
        observations=tuple(observations),
    )


def _failure_status(error: Exception) -> str:
    """Normalize raw benchmark failures into reportable operational classes."""

    if isinstance(error, TimeoutError):
        return "timeout"
    name = type(error).__name__.casefold()
    if "dependency" in name or "connection" in name or "qdrant" in name:
        return "dependency_error"
    return "error"


def _search(
    retrieval_service: RetrievalPort,
    *,
    question: str,
    mode: RetrievalMode,
    top_k: int,
    document_ids: Sequence[str] = (),
    reranker_enabled: bool | None,
) -> RetrievalResult:
    """Pass the ablation flag when the concrete port supports it."""

    kwargs: dict[str, object] = {
        "question": question,
        "mode": mode,
        "top_k": top_k,
        "document_ids": document_ids,
    }
    parameters = inspect.signature(retrieval_service.search).parameters
    if reranker_enabled is not None and (
        "reranker_enabled" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    ):
        kwargs["reranker_enabled"] = reranker_enabled
    return cast(RetrievalResult, cast(Any, retrieval_service).search(**kwargs))
