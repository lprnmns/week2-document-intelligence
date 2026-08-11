"""Query orchestration: retrieve, gate, then optionally generate."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import inspect
import logging
import re
from time import perf_counter
from typing import Any, cast

from ..domain.answerability import (
    AnswerabilityDecision,
    AnswerabilityPolicy,
    AnswerabilityPolicySet,
    AnswerabilitySignals,
    qualifier_coverage,
)
from ..domain.entities import Decision, NoAnswerReason, RetrievalMode
from ..domain.errors import ErrorCode, ServiceError
from ..domain.evidence_safety import EvidenceSafetyPolicy
from ..domain.evidence_validation import (
    EvidenceWarning,
    validate_answer_against_evidence,
)
from ..domain.generation import AnswerGenerationError
from ..domain.prompt import PromptPackResult
from ..domain.prompt_safety import PromptSafetyPolicy
from ..domain.retrieval import RetrievedChunk, RetrievalResult
from ..observability.query_trace import (
    JsonQueryTraceSink,
    QueryTraceEvent,
    QueryTraceSink,
)
from ..observability.metrics import MetricsRegistry
from .ports import AnswerGenerator
from .retrieval_service import RetrievalService


LOGGER = logging.getLogger("document_intelligence_service.query")

TraceCallback = Callable[
    [str, str, str, dict[str, object] | None, float | None],
    None,
]


@dataclass(frozen=True, slots=True)
class QueryExecutionResult:
    """Application result mapped to the public query response by the API."""

    decision: Decision
    answer: str | None
    no_answer_reason: NoAnswerReason | None
    sources: tuple[RetrievedChunk, ...]
    retrieval: RetrievalResult
    provider: str | None
    model: str | None
    llm_ms: float
    total_ms: float
    answerability: AnswerabilityDecision
    warnings: tuple[EvidenceWarning, ...]
    prompt_pack: PromptPackResult | None = None


class QueryService:
    """Coordinate retrieval, pre-LLM rejection and grounded generation."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        answerability: AnswerabilityPolicy | AnswerabilityPolicySet,
        answer_generator: AnswerGenerator,
        prompt_safety: PromptSafetyPolicy | None = None,
        evidence_safety: EvidenceSafetyPolicy | None = None,
        trace_sink: QueryTraceSink | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._answerability = answerability
        self._answer_generator = answer_generator
        self._prompt_safety = prompt_safety or PromptSafetyPolicy()
        self._evidence_safety = evidence_safety or EvidenceSafetyPolicy()
        self._trace_sink = trace_sink or JsonQueryTraceSink()
        self._metrics = metrics

    async def execute(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[str] = (),
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        reranker_enabled: bool | None = None,
        generation_model: str | None = None,
        trace: TraceCallback | None = None,
    ) -> QueryExecutionResult:
        """Run the bounded query sequence and skip generation when unsafe."""

        started = perf_counter()
        _emit_trace(
            trace,
            "request_validation",
            "running",
            "Request accepted by the query use-case",
            {"top_k": top_k, "retrieval_mode": mode.value},
            None,
        )
        _emit_trace(
            trace,
            "request_validation",
            "passed",
            "Question and bounded query controls are valid",
            {"top_k": top_k, "retrieval_mode": mode.value},
            None,
        )
        _emit_trace(
            trace,
            "scope_normalization",
            "passed",
            "Tenant, ACL and document scope normalized before retrieval",
            {
                "tenant_id": tenant_id,
                "document_count": len(tuple(dict.fromkeys(document_ids))),
                "acl_count": len(tuple(acl_tags)),
                "active_versions_only": True,
            },
            None,
        )
        if self._prompt_safety.evaluate(question).blocked:
            retrieval = _empty_retrieval(mode)
            _emit_trace(
                trace,
                "prompt_safety",
                "failed",
                "Question blocked by the direct prompt-injection policy",
                {"reason": NoAnswerReason.SECURITY_POLICY.value},
                0.0,
            )
            _emit_trace(
                trace,
                "query_representation",
                "skipped",
                "Query representations skipped because the request was blocked",
                {"reason": NoAnswerReason.SECURITY_POLICY.value},
                0.0,
            )
            _emit_skipped_retrieval_stages(trace, mode)
            gate = assess_answerability(
                question=question,
                retrieval=retrieval,
                answerability=self._answerability,
                prompt_safety=self._prompt_safety,
            )
            _emit_trace(
                trace,
                "answerability",
                "failed",
                "Answerability failed; LLM will not be called",
                _answerability_details(
                    question,
                    retrieval,
                    gate,
                    self._answerability,
                ),
                0.0,
            )
            _emit_trace(
                trace,
                "llm",
                "skipped",
                "LLM skipped because the security gate failed",
                {"reason": gate.reason.value if gate.reason else "unknown"},
                0.0,
            )
            _emit_trace(
                trace,
                "response",
                "passed",
                "Structured no-answer response prepared",
                {"decision": gate.decision.value},
                (perf_counter() - started) * 1000,
            )
            return self._record_and_return(
                question,
                QueryExecutionResult(
                    decision=gate.decision,
                    answer=None,
                    no_answer_reason=gate.reason,
                    sources=(),
                    retrieval=retrieval,
                    provider=None,
                    model=None,
                    llm_ms=0.0,
                    total_ms=(perf_counter() - started) * 1000,
                    answerability=gate,
                    warnings=(),
                ),
            )
        _emit_trace(
            trace,
            "prompt_safety",
            "passed",
            "Question passed the direct prompt-injection policy",
            {},
            0.0,
        )
        try:
            retrieval = await asyncio.to_thread(
                self._retrieval_service.search,
                question=question,
                mode=mode,
                top_k=top_k,
                document_ids=document_ids,
                tenant_id=tenant_id,
                acl_tags=acl_tags,
                reranker_enabled=reranker_enabled,
                trace=trace,
            )
        except ServiceError:
            raise
        except Exception as error:
            if self._metrics is not None:
                self._metrics.increment(
                    "rag_dependency_errors_total",
                    {"dependency": "retrieval"},
                )
            raise ServiceError(
                code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                message="Retrieval dependency is unavailable",
            ) from error
        _emit_trace(
            trace,
            "evidence_selection",
            "running",
            "Applying evidence safety and selecting the final candidate set",
            {"candidate_window": len(retrieval.candidate_window)},
            None,
        )
        retrieval, blocked_evidence = _apply_evidence_safety(
            retrieval,
            policy=self._evidence_safety,
        )
        gate = assess_answerability(
            question=question,
            retrieval=retrieval,
            answerability=self._answerability,
            prompt_safety=self._prompt_safety,
            evidence_safety_blocked=blocked_evidence and not retrieval.candidates,
        )
        _emit_trace(
            trace,
            "evidence_selection",
            "passed" if retrieval.candidates else "failed",
            f"Evidence selection kept {len(retrieval.candidates)} candidates",
            {
                "selected_count": len(retrieval.candidates),
                "blocked_candidates": blocked_evidence,
                "evidence": _evidence_payload(retrieval),
            },
            None,
        )
        _emit_trace(
            trace,
            "answerability",
            "passed" if gate.decision is Decision.ANSWERED else "failed",
            (
                "Answerability passed; evidence is sufficient for generation"
                if gate.decision is Decision.ANSWERED
                else "Answerability failed; LLM will not be called"
            ),
            _answerability_details(
                question,
                retrieval,
                gate,
                self._answerability,
            ),
            None,
        )
        if gate.decision is Decision.NO_ANSWER:
            _emit_trace(
                trace,
                "prompt_build",
                "skipped",
                "Prompt construction skipped by answerability gate",
                {"reason": gate.reason.value if gate.reason else "unknown"},
                0.0,
            )
            _emit_trace(
                trace,
                "llm",
                "skipped",
                "LLM skipped because answerability failed",
                {"reason": gate.reason.value if gate.reason else "unknown"},
                0.0,
            )
            _emit_trace(
                trace,
                "response",
                "passed",
                "Structured no-answer response prepared",
                {"decision": gate.decision.value},
                (perf_counter() - started) * 1000,
            )
            return self._record_and_return(
                question,
                QueryExecutionResult(
                    decision=gate.decision,
                    answer=None,
                    no_answer_reason=gate.reason,
                    sources=(),
                    retrieval=retrieval,
                    provider=None,
                    model=None,
                    llm_ms=0.0,
                    total_ms=(perf_counter() - started) * 1000,
                    answerability=gate,
                    warnings=(),
                ),
            )

        prompt_started = perf_counter()
        prompt_pack = _prepare_prompt_pack(
            self._answer_generator,
            question=question,
            evidence=retrieval.candidates,
        )
        prompt_duration_ms = (perf_counter() - prompt_started) * 1000
        if prompt_pack is None:
            _emit_trace(
                trace,
                "prompt_build",
                "observed",
                "Prompt builder did not expose a membership result",
                {
                    "selected_source_ids": [
                        item.source_id for item in retrieval.candidates
                    ],
                    "selected_count": len(retrieval.candidates),
                    "membership_observed": False,
                },
                prompt_duration_ms,
            )
        else:
            _emit_trace(
                trace,
                "prompt_build",
                "passed" if prompt_pack.included_source_ids else "failed",
                "Grounded prompt packed from canonical evidence fragments",
                prompt_pack.as_dict(),
                prompt_duration_ms,
            )
        _emit_trace(
            trace,
            "llm",
            "running",
            "Generation model is answering from selected evidence",
            {
                "evidence_count": len(retrieval.candidates),
                "requested_model": generation_model,
            },
            None,
        )
        try:
            generate_with_model = getattr(
                self._answer_generator,
                "generate_with_model",
                None,
            )
            if generation_model is not None and callable(generate_with_model):
                kwargs: dict[str, object] = {
                    "model": generation_model,
                    "question": question,
                    "evidence": retrieval.candidates,
                }
                if prompt_pack is not None and _accepts_keyword(
                    generate_with_model,
                    "prompt_pack",
                ):
                    kwargs["prompt_pack"] = prompt_pack
                generated = await cast(Any, generate_with_model)(**kwargs)
            elif generation_model is None:
                kwargs = {
                    "question": question,
                    "evidence": retrieval.candidates,
                }
                generate = self._answer_generator.generate
                if prompt_pack is not None and _accepts_keyword(generate, "prompt_pack"):
                    kwargs["prompt_pack"] = prompt_pack
                generated = await cast(Any, generate)(**kwargs)
            else:
                raise ServiceError(
                    code=ErrorCode.INVALID_REQUEST,
                    message="Generation model selection is not supported by the adapter",
                    stage="llm",
                )
        except AnswerGenerationError as exc:
            if self._metrics is not None:
                self._metrics.increment(
                    "rag_dependency_errors_total",
                    {"dependency": "ollama"},
                )
            LOGGER.warning(
                "query answer generation failed stage=llm reason=%s",
                exc.reason_code,
            )
            _emit_trace(
                trace,
                "llm",
                "failed",
                "Generation dependency failed",
                {"reason": exc.reason_code},
                None,
            )
            message = (
                "LLM cevap üretme süresi doldu; retrieval tamamlandı ancak "
                "cevap oluşturulamadı."
                if exc.reason_code == "TIMEOUT"
                else "Answer generation dependency is unavailable"
            )
            raise ServiceError(
                code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                message=message,
                stage="llm",
                reason=exc.reason_code,
            ) from exc
        _emit_trace(
            trace,
            "llm",
            "passed",
            "LLM CALL SUCCEEDED",
            {
                "provider": generated.provider,
                "model": generated.model,
                "evidence_count": len(retrieval.candidates),
            },
            generated.latency_ms,
        )
        validation = validate_answer_against_evidence(
            answer=generated.answer,
            evidence=retrieval.candidates,
        )
        _emit_trace(
            trace,
            "response",
            "passed",
            "Structured answer and canonical sources prepared",
            {
                "decision": Decision.ANSWERED.value,
                "source_count": len(retrieval.candidates),
                "warning_codes": [warning.code.value for warning in validation.warnings],
            },
            (perf_counter() - started) * 1000,
        )
        return self._record_and_return(
            question,
            QueryExecutionResult(
                decision=Decision.ANSWERED,
                answer=generated.answer,
                no_answer_reason=None,
                sources=retrieval.candidates,
                retrieval=retrieval,
                provider=generated.provider,
                model=generated.model,
                llm_ms=generated.latency_ms,
                total_ms=(perf_counter() - started) * 1000,
                answerability=gate,
                warnings=validation.warnings,
                prompt_pack=prompt_pack or getattr(generated, "prompt_pack", None),
            ),
        )

    def _record_and_return(
        self,
        question: str,
        result: QueryExecutionResult,
    ) -> QueryExecutionResult:
        """Record a privacy-safe trace before returning an application result."""

        self._trace_sink.record(
            QueryTraceEvent.from_query_result(
                question=question,
                decision=result.decision,
                no_answer_reason=result.no_answer_reason,
                retrieval=result.retrieval,
                selected_evidence_count=len(result.sources),
                top_score=result.answerability.top_score,
                score_margin=result.answerability.score_margin,
                coverage_ratio=result.answerability.coverage_ratio,
                provider=result.provider,
                model=result.model,
                warnings=result.warnings,
                llm_ms=result.llm_ms,
                total_ms=result.total_ms,
                policy_profile=result.answerability.policy_profile,
                calibration_id=result.answerability.calibration_id,
                score_threshold=result.answerability.score_threshold,
                coverage_threshold=result.answerability.coverage_threshold,
                required_qualifiers=result.answerability.required_qualifiers,
                matched_qualifiers=result.answerability.matched_qualifiers,
                missing_qualifiers=result.answerability.missing_qualifiers,
                qualifier_coverage_satisfied=(
                    result.answerability.qualifier_coverage_satisfied
                ),
            )
        )
        if self._metrics is not None:
            labels = {
                "decision": result.decision.value,
                "mode": result.retrieval.mode,
            }
            self._metrics.increment("rag_query_total", labels)
            self._metrics.observe(
                "rag_query_duration_ms",
                result.total_ms,
                {"mode": result.retrieval.mode},
            )
            for stage, duration in (
                ("embed", result.retrieval.embedding_ms),
                ("search", result.retrieval.search_ms),
                ("rerank", result.retrieval.rerank_ms),
                ("llm", result.llm_ms),
            ):
                self._metrics.observe(
                    "rag_query_duration_ms",
                    duration,
                    {"mode": result.retrieval.mode, "stage": stage},
                )
            self._metrics.observe(
                "rag_retrieval_candidate_count",
                float(result.retrieval.dense_candidates + result.retrieval.sparse_candidates),
                {"mode": result.retrieval.mode},
            )
            if result.no_answer_reason is not None:
                self._metrics.increment(
                    "rag_no_answer_total",
                    {"reason_code": result.no_answer_reason.value},
                )
        return result

    def _signals(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> tuple[AnswerabilitySignals, str]:
        """Extract comparable score and coverage signals from retrieval trace."""

        return build_answerability_signals(question, retrieval)

    @staticmethod
    def _score_kind(retrieval: RetrievalResult) -> str:
        if any(candidate.rerank_score is not None for candidate in retrieval.candidates):
            return "rerank"
        if retrieval.mode == RetrievalMode.BM25.value:
            return "sparse"
        if retrieval.mode == RetrievalMode.DENSE.value:
            return "dense"
        if any(candidate.dense_score is not None for candidate in retrieval.candidates):
            return "dense"
        return "sparse"

    @staticmethod
    def _score_for(candidate: RetrievedChunk, score_kind: str) -> float:
        if score_kind == "rerank" and candidate.rerank_score is not None:
            return candidate.rerank_score
        if score_kind == "sparse" and candidate.sparse_score is not None:
            return candidate.sparse_score
        if score_kind == "dense" and candidate.dense_score is not None:
            return candidate.dense_score
        return candidate.score

    @staticmethod
    def _coverage_ratio(
        question: str,
        candidates: Sequence[RetrievedChunk],
    ) -> float:
        """Measure lexical overlap as a diagnostic, not a semantic truth test."""

        terms = {
            token.casefold()
            for token in re.findall(r"\w+", question, flags=re.UNICODE)
            if len(token) >= 3
        }
        if not terms:
            return 1.0
        evidence_terms = {
            token.casefold()
            for candidate in candidates
            for token in re.findall(r"\w+", candidate.context_text, flags=re.UNICODE)
        }
        return len(terms & evidence_terms) / len(terms)


def assess_answerability(
    *,
    question: str,
    retrieval: RetrievalResult,
    answerability: AnswerabilityPolicy | AnswerabilityPolicySet,
    prompt_safety: PromptSafetyPolicy | None = None,
    evidence_safety_blocked: bool = False,
) -> AnswerabilityDecision:
    """Apply the pre-LLM gate to a previously captured retrieval result."""

    safety = prompt_safety or PromptSafetyPolicy()
    policy = _select_answerability_policy(answerability, retrieval)
    if safety.evaluate(question).blocked:
        return AnswerabilityDecision(
            decision=Decision.NO_ANSWER,
            reason=NoAnswerReason.SECURITY_POLICY,
            top_score=None,
            score_margin=None,
            coverage_ratio=0.0,
            policy_profile=policy.profile_name,
            calibration_id=policy.calibration_id,
            coverage_threshold=policy.min_coverage,
        )
    if evidence_safety_blocked:
        return AnswerabilityDecision(
            decision=Decision.NO_ANSWER,
            reason=NoAnswerReason.SECURITY_POLICY,
            top_score=None,
            score_margin=None,
            coverage_ratio=0.0,
            policy_profile=policy.profile_name,
            calibration_id=policy.calibration_id,
            coverage_threshold=policy.min_coverage,
        )
    signals, score_kind = build_answerability_signals(question, retrieval)
    return policy.decide(signals=signals, score_kind=score_kind)


def build_answerability_signals(
    question: str,
    retrieval: RetrievalResult,
) -> tuple[AnswerabilitySignals, str]:
    """Build the same trace signals used by the live query path."""

    score_kind = QueryService._score_kind(retrieval)
    scores = tuple(
        QueryService._score_for(candidate, score_kind)
        for candidate in retrieval.candidates
    )
    # Hybrid candidates are ordered by RRF, not by the score used by the
    # answerability policy. Sort the comparable score values before deriving
    # top-score and margin; otherwise a valid dense result can look like a
    # negative-margin near miss merely because sparse ranking placed first.
    ranked_scores = tuple(sorted(scores, reverse=True))
    top_score = ranked_scores[0] if ranked_scores else None
    margin = (
        ranked_scores[0] - ranked_scores[1]
        if len(ranked_scores) > 1
        else None
    )
    qualifiers = qualifier_coverage(
        question,
        tuple(candidate.context_text for candidate in retrieval.candidates),
    )
    return (
        AnswerabilitySignals(
            evidence_count=len(retrieval.candidates),
            top_score=top_score,
            score_margin=margin,
            coverage_ratio=QueryService._coverage_ratio(
                question,
                retrieval.candidates,
            ),
            required_qualifiers=qualifiers.required,
            matched_qualifiers=qualifiers.matched,
            missing_qualifiers=qualifiers.missing,
            qualifier_coverage_satisfied=qualifiers.satisfied,
        ),
        score_kind,
    )


def _empty_retrieval(mode: RetrievalMode) -> RetrievalResult:
    """Create a zero-cost trace for a query blocked before retrieval."""

    return RetrievalResult(
        mode=mode.value,
        candidates=(),
        dense_candidates=0,
        sparse_candidates=0,
        rrf_candidates=0,
        embedding_ms=0.0,
        search_ms=0.0,
    )


def _emit_trace(
    trace: TraceCallback | None,
    stage: str,
    status: str,
    summary: str,
    details: dict[str, object] | None,
    duration_ms: float | None,
) -> None:
    """Send a best-effort live event without affecting the use-case."""

    if trace is None:
        return
    try:
        trace(stage, status, summary, details, duration_ms)
    except Exception:
        return


def _prepare_prompt_pack(
    generator: object,
    *,
    question: str,
    evidence: Sequence[RetrievedChunk],
) -> PromptPackResult | None:
    """Ask the concrete generator for its real pack result when supported."""

    pack_prompt = getattr(generator, "pack_prompt", None)
    if not callable(pack_prompt):
        return None
    packed = pack_prompt(question=question, evidence=evidence)
    return packed if isinstance(packed, PromptPackResult) else None


def _accepts_keyword(callable_object: object, name: str) -> bool:
    """Keep injected test generators compatible while using the real contract."""

    try:
        parameters = inspect.signature(cast(Any, callable_object)).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def _emit_skipped_retrieval_stages(
    trace: TraceCallback | None,
    mode: RetrievalMode,
) -> None:
    """Make a pre-retrieval security rejection explicit in the live console."""

    for stage in ("dense_retrieval", "sparse_retrieval", "rrf_fusion", "reranker"):
        _emit_trace(
            trace,
            stage,
            "skipped",
            "Stage skipped because the request was blocked before retrieval",
            {"mode": mode.value, "reason": NoAnswerReason.SECURITY_POLICY.value},
            0.0,
        )


def _answerability_details(
    question: str,
    retrieval: RetrievalResult,
    decision: AnswerabilityDecision,
    policy: AnswerabilityPolicy | AnswerabilityPolicySet,
) -> dict[str, object]:
    """Expose gate signals and pass/fail reasons without exposing raw text."""

    signals, score_kind = build_answerability_signals(question, retrieval)
    selected_policy = _select_answerability_policy(policy, retrieval)
    minimum = selected_policy._minimum_for(score_kind)
    return {
        "decision": decision.decision.value,
        "reason_code": decision.reason.value if decision.reason else None,
        "policy": {
            "profile": selected_policy.profile_name,
            "calibration_id": selected_policy.calibration_id,
        },
        "score_kind": score_kind,
        "signals": {
            "evidence_exists": {
                "status": "PASS" if len(retrieval.candidates) else "FAIL",
                "value": len(retrieval.candidates),
            },
            "top_score_threshold": {
                "status": (
                    "PASS"
                    if decision.top_score is not None and decision.top_score >= minimum
                    else "FAIL"
                ),
                "value": decision.top_score,
                "threshold": minimum,
            },
            "coverage": {
                "status": (
                    "PASS"
                    if decision.coverage_ratio >= selected_policy.min_coverage
                    else "FAIL"
                ),
                "value": decision.coverage_ratio,
                "threshold": selected_policy.min_coverage,
            },
            "qualifier_coverage": {
                "status": (
                    "PASS"
                    if signals.qualifier_coverage_satisfied
                    else "FAIL"
                ),
                "required": list(signals.required_qualifiers),
                "matched": list(signals.matched_qualifiers),
                "missing": list(signals.missing_qualifiers),
            },
            "margin": {
                "status": (
                    "PASS"
                    if decision.score_margin is None
                    or decision.score_margin >= selected_policy.min_margin
                    else "FAIL"
                ),
                "value": decision.score_margin,
                "threshold": selected_policy.min_margin,
            },
            "filters": {"status": "PASS", "value": True},
        },
    }


def _select_answerability_policy(
    answerability: AnswerabilityPolicy | AnswerabilityPolicySet,
    retrieval: RetrievalResult,
) -> AnswerabilityPolicy:
    """Resolve the policy from explicit candidate metadata when available."""

    if isinstance(answerability, AnswerabilityPolicySet):
        return answerability.select(
            tuple(
                candidate.chunking_profile
                for candidate in retrieval.candidates
                if candidate.chunking_profile is not None
            )
        )
    return answerability


def _evidence_payload(retrieval: RetrievalResult) -> list[dict[str, object]]:
    """Return only the bounded evidence facts needed by the demo trace."""

    return [
        {
            "source_id": item.source_id,
            "document_id": item.document_id,
            "parent_id": item.parent_id,
            "title": item.title,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "chunking_profile": item.chunking_profile,
            "excerpt": _compact_excerpt(item.context_text),
            "chunk_text": _bounded_text(item.text),
            "parent_context": _bounded_text(item.parent_text or item.text),
            "parent_context_available": item.parent_text is not None,
            "selected_as_evidence": True,
            "used_in_prompt": False,
            "dense_rank": item.dense_rank,
            "sparse_rank": item.sparse_rank,
            "fusion_rank": item.rank,
            "rerank_rank": item.rank if item.rerank_score is not None else None,
            "rerank_score": item.rerank_score,
        }
        for item in retrieval.candidates
    ]


def _compact_excerpt(text: str, limit: int = 220) -> str:
    """Keep live evidence cards readable and bounded."""

    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _bounded_text(text: str, limit: int = 4000) -> str:
    """Keep live evidence text bounded without changing retrieval behavior."""

    normalized = text.strip()
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _apply_evidence_safety(
    retrieval: RetrievalResult,
    *,
    policy: EvidenceSafetyPolicy,
) -> tuple[RetrievalResult, bool]:
    """Filter unsafe final/context candidates before answerability or LLM."""

    final_result = policy.filter(retrieval.candidates)
    window = retrieval.candidate_window or retrieval.candidates
    window_result = policy.filter(window)
    blocked = bool(
        final_result.blocked_source_ids or window_result.blocked_source_ids
    )
    if not blocked:
        return retrieval, False

    safe_final_ids = {item.source_id for item in final_result.safe_evidence}
    safe_window_ids = {item.source_id for item in window_result.safe_evidence}
    return (
        replace(
            retrieval,
            candidates=tuple(
                item
                for item in retrieval.candidates
                if item.source_id in safe_final_ids
            ),
            candidate_window=tuple(
                item for item in window if item.source_id in safe_window_ids
            ),
        ),
        True,
    )
