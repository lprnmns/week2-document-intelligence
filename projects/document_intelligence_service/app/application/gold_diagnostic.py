"""Gold-aware, deterministic RAG stage diagnosis for curated mentor cases."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import asyncio
import json
from pathlib import Path

from ..domain.entities import Decision, DocumentStatus, RetrievalMode
from ..domain.answer_check import (
    AnswerCheckMode,
    AnswerCheckVerdict,
    check_answer,
)
from ..domain.errors import ServiceError
from ..domain.gold_diagnostic import (
    DiagnosticRootCause,
    DiagnosticVerdict,
    GoldCase,
    GoldLocator,
    compare_claims,
    compare_decision,
    load_gold_cases,
    manifest_documents,
)
from ..domain.ingestion import IngestionReceipt
from ..domain.retrieval import RetrievedChunk, RetrievalDebugCandidate
from .document_service import DocumentService
from .ingestion_service import IngestionService
from .ports import GoldEvidenceLookup
from .query_service import QueryExecutionResult, QueryService

TraceEvent = dict[str, object]
TraceCallback = Callable[
    [str, str, str, dict[str, object] | None, float | None],
    None,
]


@dataclass(frozen=True, slots=True)
class ResolvedGoldEvidence:
    """A manifest locator bound to the current active document version."""

    locator: GoldLocator
    document_id: str
    version_id: str
    source: RetrievedChunk
    alternatives: tuple[RetrievedChunk, ...] = ()

    @property
    def accepted_sources(self) -> tuple[RetrievedChunk, ...]:
        """Return overlap-equivalent canonical children accepted by the locator."""

        return (self.source, *self.alternatives)

    def as_dict(self) -> dict[str, object]:
        """Return canonical source metadata without embedding vectors."""

        return {
            "document_key": self.locator.document_key,
            "must_contain": self.locator.must_contain,
            "source_id": self.source.source_id,
            "acceptable_source_ids": [
                item.source_id for item in self.accepted_sources
            ],
            "document_id": self.document_id,
            "version_id": self.version_id,
            "title": self.source.title,
            "page_start": self.source.page_start,
            "page_end": self.source.page_end,
            "parent_id": self.source.parent_id,
            "text": self.source.text,
        }


class GoldDatasetInvalid(ValueError):
    """Raised when a trusted gold locator cannot be resolved unambiguously."""


class GoldEvidenceResolver:
    """Resolve manifest locators against active versions only."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        document_service: DocumentService,
        lookup: GoldEvidenceLookup,
    ) -> None:
        self._documents = manifest_documents(manifest_path)
        self._document_service = document_service
        self._lookup = lookup

    async def resolve(
        self,
        locators: Sequence[GoldLocator],
        *,
        tenant_id: str = "default",
    ) -> tuple[ResolvedGoldEvidence, ...]:
        """Resolve every locator or fail the whole diagnostic dataset safely."""

        documents = await self._document_service.list_documents(
            limit=100,
            cursor=None,
            tenant_id=tenant_id,
        )
        resolved: list[ResolvedGoldEvidence] = []
        for locator in locators:
            filename = self._documents.get(locator.document_key)
            if filename is None:
                raise GoldDatasetInvalid(
                    f"unknown gold document key: {locator.document_key}"
                )
            matches = tuple(
                document
                for document in documents.items
                if document.title.casefold() == filename.casefold()
                and document.status is DocumentStatus.ACTIVE
                and document.active_version_id
            )
            if len(matches) != 1:
                raise GoldDatasetInvalid(
                    f"gold document {filename!r} has {len(matches)} active matches"
                )
            document = matches[0]
            assert document.active_version_id is not None
            candidates = self._lookup.find(
                document_id=document.document_id,
                version_id=document.active_version_id,
                page=locator.page,
                must_contain=locator.must_contain,
                tenant_id=tenant_id,
            )
            if not candidates:
                raise GoldDatasetInvalid(
                    f"gold locator {filename} p.{locator.page!s} is not unique "
                    f"({len(candidates)} matches)"
                )
            parent_ids = {item.parent_id for item in candidates}
            if len(parent_ids) != 1:
                raise GoldDatasetInvalid(
                    f"gold locator {filename} p.{locator.page!s} is not unique "
                    f"across parents ({len(candidates)} matches)"
                )
            ordered = tuple(
                sorted(candidates, key=lambda item: (len(item.text), item.source_id))
            )
            resolved.append(
                ResolvedGoldEvidence(
                    locator=locator,
                    document_id=document.document_id,
                    version_id=document.active_version_id,
                    source=ordered[0],
                    alternatives=ordered[1:],
                )
            )
        return tuple(resolved)

    async def resolve_document_ids(
        self,
        keys: Sequence[str],
        *,
        tenant_id: str = "default",
    ) -> tuple[str, ...]:
        """Resolve demo document keys to currently active logical IDs."""

        documents = await self._document_service.list_documents(
            limit=100,
            cursor=None,
            tenant_id=tenant_id,
        )
        ids: list[str] = []
        for key in keys:
            filename = self._documents.get(key)
            if filename is None:
                raise GoldDatasetInvalid(f"unknown gold document key: {key}")
            matches = tuple(
                document.document_id
                for document in documents.items
                if document.title.casefold() == filename.casefold()
                and document.status is DocumentStatus.ACTIVE
                and document.active_version_id
            )
            if len(matches) != 1:
                raise GoldDatasetInvalid(
                    f"demo document {filename!r} is not uniquely active"
                )
            ids.append(matches[0])
        return tuple(dict.fromkeys(ids))

    async def browse(
        self,
        *,
        document_ids: Sequence[str],
        page: int | None = None,
        text: str = "",
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        limit: int = 50,
    ) -> tuple[RetrievedChunk, ...]:
        """Browse only active chunks inside the caller's selected scope."""

        scoped_ids = tuple(document_ids)
        if not scoped_ids:
            documents = await self._document_service.list_documents(
                limit=100,
                cursor=None,
                tenant_id=tenant_id,
            )
            scoped_ids = tuple(
                item.document_id
                for item in documents.items
                if item.status is DocumentStatus.ACTIVE and item.active_version_id
            )
        return self._lookup.browse(
            document_ids=scoped_ids,
            page=page,
            text=text,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
            limit=limit,
        )

    async def resolve_source_ids(
        self,
        source_ids: Sequence[str],
        *,
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> tuple[RetrievedChunk, ...]:
        """Resolve trusted source IDs without bypassing active/ACL filters."""

        return self._lookup.find_source_ids(
            source_ids=source_ids,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
        )


class GoldDiagnosticService:
    """Run curated cases through the real query path and attribute first loss."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        asset_dir: Path,
        document_service: DocumentService,
        ingestion_service: IngestionService,
        query_service: QueryService,
        resolver: GoldEvidenceResolver,
        retrieval_service: object | None = None,
        similarity: Callable[[str, str], float] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.asset_dir = asset_dir
        self._document_service = document_service
        self._ingestion_service = ingestion_service
        self._query_service = query_service
        self._resolver = resolver
        self._retrieval_service = retrieval_service
        self._similarity = similarity
        self._cases = load_gold_cases(manifest_path)
        self._cases_by_id = {case.case_id: case for case in self._cases}
        self._document_map = manifest_documents(manifest_path)

    def list_cases(self) -> tuple[dict[str, object], ...]:
        """Return curated case metadata for the prepared ASK selector."""

        return tuple(case.as_dict() for case in self._cases)

    async def prepare_corpus(
        self,
        *,
        tenant_id: str = "default",
        acl_tags: tuple[str, ...] = ("public",),
    ) -> tuple[dict[str, object], ...]:
        """Ingest synthetic assets through the normal document acceptance use-case."""

        receipts: list[dict[str, object]] = []
        for key, filename in self._document_map.items():
            path = self.asset_dir / filename
            if not path.is_file():
                raise GoldDatasetInvalid(f"demo asset is missing: {filename}")
            receipt = await self._ingestion_service.accept_receipt(
                content=path.read_bytes(),
                filename=filename,
                content_type="application/pdf",
                idempotency_key=f"gold-diagnostic-demo:{key}",
                tenant_id=tenant_id,
                acl_tags=acl_tags,
            )
            receipts.append(_receipt_dict(key, filename, receipt))
        return tuple(receipts)

    async def run_case(
        self,
        *,
        case_id: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        reranker_enabled: bool = False,
        generation_model: str | None = None,
        retrieval_only: bool = False,
        tenant_id: str = "default",
        acl_tags: tuple[str, ...] = ("public",),
    ) -> dict[str, object]:
        """Run one trusted case; semantic similarity cannot affect the verdict."""

        case = self._cases_by_id.get(case_id)
        if case is None:
            raise GoldDatasetInvalid(f"unknown diagnostic case: {case_id}")
        locators = (
            *case.gold_evidence,
            *case.forbidden_evidence,
            *case.adversarial_evidence,
        )
        try:
            resolved = await self._resolver.resolve(locators, tenant_id=tenant_id)
            all_demo_ids = await self._resolver.resolve_document_ids(
                tuple(self._document_map),
                tenant_id=tenant_id,
            )
            scoped_ids = (
                await self._resolver.resolve_document_ids(
                    case.scope_document_keys,
                    tenant_id=tenant_id,
                )
                if case.scope_document_keys
                else all_demo_ids
            )
        except GoldDatasetInvalid as error:
            return _invalid_result(case, str(error))

        gold_count = len(case.gold_evidence)
        resolved_gold = tuple(resolved[:gold_count])
        # Gold locators are labels, never an implicit retrieval filter. A
        # curated case searches the complete prepared Atlas corpus unless the
        # manifest explicitly declares a scope.
        document_ids = tuple(dict.fromkeys(scoped_ids))
        events: list[TraceEvent] = []

        def trace(
            stage: str,
            status: str,
            summary: str,
            details: dict[str, object] | None,
            duration_ms: float | None,
        ) -> None:
            events.append(
                {
                    "stage": stage,
                    "status": status,
                    "summary": summary,
                    "details": details or {},
                    "duration_ms": duration_ms,
                }
            )

        query_result: QueryExecutionResult | None = None
        error_payload: dict[str, object] | None = None
        if retrieval_only:
            error_payload = {
                "code": "RETRIEVAL_ONLY",
                "message": "Generation was intentionally not invoked for this diagnostic run",
                "stage": "generation",
            }
            query_result, retrieval_events = await self._run_retrieval_only(
                question=case.question,
                mode=mode,
                top_k=top_k,
                document_ids=document_ids,
                tenant_id=tenant_id,
                acl_tags=acl_tags,
                reranker_enabled=reranker_enabled,
            )
            events.extend(retrieval_events)
        else:
            try:
                query_result = await self._query_service.execute(
                    question=case.question,
                    mode=mode,
                    top_k=top_k,
                    document_ids=document_ids,
                    tenant_id=tenant_id,
                    acl_tags=acl_tags,
                    reranker_enabled=reranker_enabled,
                    generation_model=generation_model,
                    trace=trace,
                )
            except ServiceError as error:
                error_payload = {
                    "code": error.code.value,
                    "message": error.message,
                    "stage": error.stage,
                    "reason": error.reason,
                }

        actual_decision, actual_reason, actual_answer = _actual_fields(
            query_result,
            error_payload,
        )
        claims = compare_claims(
            expected_claims=case.expected_claims,
            forbidden_claims=case.forbidden_claims,
            answer=actual_answer,
        )
        verdict = compare_decision(
            expected_decision=case.expected_decision,
            actual_decision=actual_decision,
            actual_reason=actual_reason,
            claims=claims,
            expected_reason=case.expected_reason,
        )
        journey = _journey(
            case=case,
            resolved_gold=resolved_gold,
            result=query_result,
            events=events,
            mode=mode,
            retrieval_only=retrieval_only,
        )
        fact_coverage = _fact_coverage(
            expected_answer=case.expected_answer,
            resolved_gold=resolved_gold,
            result=query_result,
            events=events,
            actual_answer=actual_answer,
        )
        root_cause, first_divergence, attribution_note = _attribute(
            case=case,
            verdict=verdict,
            actual_decision=actual_decision,
            actual_reason=actual_reason,
            actual_answer=actual_answer,
            result=query_result,
            journey=journey,
            events=events,
            error_payload=error_payload,
            fact_coverage=fact_coverage,
        )
        if retrieval_only and case.expected_decision == "ANSWERED":
            verdict = DiagnosticVerdict.REVIEW_REQUIRED
            if root_cause is DiagnosticRootCause.PASS:
                root_cause = DiagnosticRootCause.REVIEW_REQUIRED
                first_divergence = "generation"
                attribution_note = (
                    "Retrieval-only diagnosis stopped before generation; no generation "
                    "claim verdict is asserted."
                )
            else:
                attribution_note = (
                    f"{attribution_note} Generation was not invoked, so no generation "
                    "claim verdict is asserted."
                )
        similarity = (
            self._similarity(case.expected_answer, actual_answer)
            if self._similarity is not None and actual_answer
            else None
        )
        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected": {
                "decision": case.expected_decision,
                "answer": case.expected_answer,
                "claims": claims_expected(case),
                "gold_evidence": [item.as_dict() for item in resolved_gold],
            },
            "actual": {
                "decision": actual_decision,
                "answer": actual_answer,
                "reason": actual_reason,
                "sources": _sources_payload(query_result),
                "model": query_result.model if query_result else None,
                "provider": query_result.provider if query_result else None,
            },
            "verdict": verdict.value,
            "root_cause": root_cause.value,
            "first_divergence": first_divergence,
            "attribution_note": attribution_note,
            "claims": claims,
            "semantic_similarity": similarity,
            "semantic_similarity_note": (
                "Informational only; it cannot produce correctness or root-cause verdicts."
            ),
            "gold_evidence_journey": journey,
            "stage_coverage": _stage_coverage(
                journey,
                query_result.retrieval.mode if query_result else mode.value,
            ),
            "fact_coverage": fact_coverage,
            "trace_events": events,
            "error": error_payload,
            "retrieval_only": retrieval_only,
            "request_id": _request_id(events),
        }

    async def compare_existing_case(
        self,
        *,
        case_id: str,
        result: QueryExecutionResult | None,
        events: Sequence[TraceEvent],
        error_payload: Mapping[str, object] | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        retrieval_only: bool = False,
        tenant_id: str = "default",
    ) -> dict[str, object]:
        """Compare one already-completed query run with trusted gold.

        This method deliberately does not call QueryService.  It is the
        bridge used by ASK's optional prepared example so one user action has
        one retrieval/generation execution.
        """

        case = self._cases_by_id.get(case_id)
        if case is None:
            raise GoldDatasetInvalid(f"unknown diagnostic case: {case_id}")
        try:
            resolved = await self._resolver.resolve(
                (
                    *case.gold_evidence,
                    *case.forbidden_evidence,
                    *case.adversarial_evidence,
                ),
                tenant_id=tenant_id,
            )
        except GoldDatasetInvalid as error:
            return _invalid_result(case, str(error))
        gold_count = len(case.gold_evidence)
        resolved_gold = tuple(resolved[:gold_count])
        actual_decision, actual_reason, actual_answer = _actual_fields(
            result,
            error_payload,
        )
        claims = compare_claims(
            expected_claims=case.expected_claims,
            forbidden_claims=case.forbidden_claims,
            answer=actual_answer,
        )
        verdict = compare_decision(
            expected_decision=case.expected_decision,
            actual_decision=actual_decision,
            actual_reason=actual_reason,
            claims=claims,
            expected_reason=case.expected_reason,
        )
        journey = _journey(
            case=case,
            resolved_gold=resolved_gold,
            result=result,
            events=events,
            mode=mode,
            retrieval_only=retrieval_only,
        )
        fact_coverage = _fact_coverage(
            expected_answer=case.expected_answer,
            resolved_gold=resolved_gold,
            result=result,
            events=events,
            actual_answer=actual_answer,
        )
        root_cause, first_divergence, attribution_note = _attribute(
            case=case,
            verdict=verdict,
            actual_decision=actual_decision,
            actual_reason=actual_reason,
            actual_answer=actual_answer,
            result=result,
            journey=journey,
            events=events,
            error_payload=error_payload,
            fact_coverage=fact_coverage,
        )
        if retrieval_only and case.expected_decision == "ANSWERED":
            verdict = DiagnosticVerdict.REVIEW_REQUIRED
            if root_cause is DiagnosticRootCause.PASS:
                root_cause = DiagnosticRootCause.REVIEW_REQUIRED
                first_divergence = "generation"
                attribution_note = (
                    "Retrieval-only diagnosis stopped before generation; no generation "
                    "claim verdict is asserted."
                )
        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected": {
                "decision": case.expected_decision,
                "reason": case.expected_reason,
                "answer": case.expected_answer,
                "claims": claims_expected(case),
                "gold_evidence": [item.as_dict() for item in resolved_gold],
            },
            "actual": {
                "decision": actual_decision,
                "answer": actual_answer,
                "reason": actual_reason,
                "sources": _sources_payload(result),
                "model": result.model if result else None,
                "provider": result.provider if result else None,
            },
            "verdict": verdict.value,
            "root_cause": root_cause.value,
            "first_divergence": first_divergence,
            "attribution_note": attribution_note,
            "claims": claims,
            "semantic_similarity": None,
            "semantic_similarity_note": (
                "Informational only; it cannot produce correctness or root-cause verdicts."
            ),
            "gold_evidence_journey": journey,
            "stage_coverage": _stage_coverage(
                journey,
                result.retrieval.mode if result else mode.value,
            ),
            "fact_coverage": fact_coverage,
            "actual_selected_evidence": _sources_payload(result),
            "trace_events": list(events),
            "error": dict(error_payload) if error_payload else None,
            "retrieval_only": retrieval_only,
            "request_id": _request_id(events),
        }

    async def compare_existing_custom(
        self,
        *,
        question: str,
        expected_answer: str,
        result: QueryExecutionResult | None,
        events: Sequence[TraceEvent],
        error_payload: Mapping[str, object] | None = None,
        answer_check_mode: AnswerCheckMode | str = AnswerCheckMode.FACT_AWARE,
        semantic_threshold: float = 0.86,
    ) -> dict[str, object]:
        """Compare a manually supplied expected answer without gold attribution."""

        actual_decision, actual_reason, actual_answer = _actual_fields(
            result,
            error_payload,
        )
        answer_check = check_answer(
            expected=expected_answer,
            actual=actual_answer,
            mode=answer_check_mode,
            semantic_threshold=semantic_threshold,
        )
        return {
            "case_id": None,
            "category": "custom",
            "question": question,
            "expected": {"decision": "CUSTOM", "answer": expected_answer},
            "actual": {
                "decision": actual_decision,
                "answer": actual_answer,
                "reason": actual_reason,
                "sources": _sources_payload(result),
            },
            "verdict": answer_check.verdict.value,
            "root_cause": DiagnosticRootCause.UNATTRIBUTED.value,
            "first_divergence": None,
            "attribution_note": "UNATTRIBUTED — GOLD EVIDENCE REQUIRED",
            "claims": compare_claims(
                expected_claims=(),
                forbidden_claims=(),
                answer=actual_answer,
            ),
            "answer_check": answer_check.as_dict(),
            "semantic_similarity": answer_check.semantic_similarity,
            "semantic_similarity_note": (
                "Informational only; no trusted gold evidence was supplied."
            ),
            "gold_evidence_journey": [],
            "stage_coverage": {},
            "trace_events": list(events),
            "error": dict(error_payload) if error_payload else None,
            "retrieval_only": False,
            "request_id": _request_id(events),
        }

    async def browse_evidence(
        self,
        *,
        document_ids: Sequence[str],
        page: int | None = None,
        text: str = "",
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        limit: int = 50,
    ) -> tuple[RetrievedChunk, ...]:
        """Expose bounded active chunks for the optional trusted picker."""

        return await self._resolver.browse(
            document_ids=document_ids,
            page=page,
            text=text,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
            limit=limit,
        )

    async def compare_existing_trusted_sources(
        self,
        *,
        question: str,
        expected_answer: str,
        source_ids: Sequence[str],
        result: QueryExecutionResult | None,
        events: Sequence[TraceEvent],
        error_payload: Mapping[str, object] | None = None,
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        answer_check_mode: AnswerCheckMode | str = AnswerCheckMode.FACT_AWARE,
        semantic_threshold: float = 0.86,
    ) -> dict[str, object]:
        """Attribute one existing run against manually trusted source IDs.

        The trusted sources are evaluator metadata only.  This method never
        calls retrieval, answerability or generation and never changes the
        already executed prompt.  It only replays deterministic membership
        comparisons over the recorded trace.
        """

        trusted = await self._resolver.resolve_source_ids(
            source_ids,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
        )
        requested_ids = tuple(dict.fromkeys(item for item in source_ids if item))
        if len(trusted) != len(requested_ids):
            found = {item.source_id for item in trusted}
            missing = [item for item in requested_ids if item not in found]
            actual_decision, actual_reason, actual_answer = _actual_fields(
                result,
                error_payload,
            )
            return {
                "case_id": None,
                "category": "manual_trusted_evidence",
                "question": question,
                "expected": {"answer": expected_answer},
                "actual": {
                    "decision": actual_decision,
                    "answer": actual_answer,
                    "reason": actual_reason,
                    "sources": _sources_payload(result),
                },
                "verdict": DiagnosticVerdict.REVIEW_REQUIRED.value,
                "root_cause": DiagnosticRootCause.DATASET_GOLD_INVALID.value,
                "first_divergence": "trusted_evidence",
                "attribution_note": (
                    "One or more trusted source IDs were not found in the active "
                    f"tenant/ACL scope: {', '.join(missing)}."
                ),
                "trusted_source_ids": list(requested_ids),
                "gold_evidence_journey": [],
                "trace_events": list(events),
            }

        expected_check = check_answer(
            expected=expected_answer,
            actual=_actual_fields(result, error_payload)[2],
            mode=answer_check_mode,
            semantic_threshold=semantic_threshold,
        )
        verdict = _diagnostic_verdict_for_answer_check(expected_check.verdict)
        locators = tuple(
            GoldLocator(
                document_key="manual",
                page=item.page_start,
                must_contain=item.text[:120],
            )
            for item in trusted
        )
        resolved = tuple(
            ResolvedGoldEvidence(
                locator=locator,
                document_id=item.document_id,
                version_id=item.version_id,
                source=item,
            )
            for locator, item in zip(locators, trusted, strict=True)
        )
        case = GoldCase(
            case_id="manual_trusted_evidence",
            category="manual_trusted_evidence",
            question=question,
            expected_decision="ANSWERED",
            expected_answer=expected_answer,
            gold_evidence=locators,
        )
        journey = _journey(
            case=case,
            resolved_gold=resolved,
            result=result,
            events=events,
            mode=(
                RetrievalMode(result.retrieval.mode)
                if result is not None
                else RetrievalMode(_event_mode(events))
            ),
        )
        actual_decision, actual_reason, actual_answer = _actual_fields(
            result,
            error_payload,
        )
        fact_coverage = _fact_coverage(
            expected_answer=expected_answer,
            resolved_gold=resolved,
            result=result,
            events=events,
            actual_answer=actual_answer,
        )
        root_cause, first_divergence, attribution_note = _attribute(
            case=case,
            verdict=verdict,
            actual_decision=actual_decision,
            actual_reason=actual_reason,
            actual_answer=actual_answer,
            result=result,
            journey=journey,
            events=events,
            error_payload=error_payload,
            answer_check_verdict=expected_check.verdict,
            fact_coverage=fact_coverage,
        )
        return {
            "case_id": None,
            "category": "manual_trusted_evidence",
            "question": question,
            "expected": {
                "decision": "ANSWERED",
                "answer": expected_answer,
                "gold_evidence": [item.as_dict() for item in resolved],
            },
            "actual": {
                "decision": actual_decision,
                "answer": actual_answer,
                "reason": actual_reason,
                "sources": _sources_payload(result),
            },
            "verdict": verdict.value,
            "root_cause": root_cause.value,
            "first_divergence": first_divergence,
            "attribution_note": attribution_note,
            "answer_check": expected_check.as_dict(),
            "trusted_source_ids": [item.source_id for item in trusted],
            "gold_evidence_journey": journey,
            "stage_coverage": _stage_coverage(
                journey,
                result.retrieval.mode if result is not None else _event_mode(events),
            ),
            "fact_coverage": fact_coverage,
            "actual_selected_evidence": _sources_payload(result),
            "trace_events": list(events),
            "semantic_similarity": expected_check.semantic_similarity,
            "semantic_similarity_note": (
                "Informational only; trusted source membership, not similarity, "
                "drives stage attribution."
            ),
            "retrieval_only": False,
            "request_id": _request_id(events),
        }

    async def run_custom(
        self,
        *,
        question: str,
        expected_answer: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        reranker_enabled: bool = False,
        generation_model: str | None = None,
        tenant_id: str = "default",
        acl_tags: tuple[str, ...] = ("public",),
        answer_check_mode: AnswerCheckMode | str = AnswerCheckMode.FACT_AWARE,
        semantic_threshold: float = 0.86,
    ) -> dict[str, object]:
        """Run advanced custom comparison without pretending to have gold evidence."""

        events: list[TraceEvent] = []

        def trace(
            stage: str,
            status: str,
            summary: str,
            details: dict[str, object] | None,
            duration_ms: float | None,
        ) -> None:
            events.append(
                {
                    "stage": stage,
                    "status": status,
                    "summary": summary,
                    "details": details or {},
                    "duration_ms": duration_ms,
                }
            )

        try:
            result = await self._query_service.execute(
                question=question,
                mode=mode,
                top_k=top_k,
                document_ids=(),
                tenant_id=tenant_id,
                acl_tags=acl_tags,
                reranker_enabled=reranker_enabled,
                generation_model=generation_model,
                trace=trace,
            )
            actual = result.answer
            actual_decision = result.decision.value.upper()
            actual_reason = (
                result.no_answer_reason.value if result.no_answer_reason else None
            )
        except ServiceError as error:
            result = None
            actual = None
            actual_decision = "PIPELINE_FAILED"
            actual_reason = error.reason
        claims = compare_claims(
            expected_claims=(),
            forbidden_claims=(),
            answer=actual,
        )
        answer_check = check_answer(
            expected=expected_answer,
            actual=actual,
            mode=answer_check_mode,
            semantic_threshold=semantic_threshold,
        )
        return {
            "case_id": None,
            "category": "custom",
            "question": question,
            "expected": {"decision": "CUSTOM", "answer": expected_answer},
            "actual": {
                "decision": actual_decision,
                "answer": actual,
                "reason": actual_reason,
                "sources": _sources_payload(result),
            },
            "verdict": answer_check.verdict.value,
            "root_cause": DiagnosticRootCause.UNATTRIBUTED.value,
            "first_divergence": None,
            "attribution_note": "UNATTRIBUTED — GOLD EVIDENCE REQUIRED",
            "claims": claims,
            "answer_check": answer_check.as_dict(),
            "semantic_similarity": answer_check.semantic_similarity,
            "semantic_similarity_note": "Informational only; no trusted gold evidence was supplied.",
            "gold_evidence_journey": [],
            "stage_coverage": {},
            "trace_events": events,
            "request_id": _request_id(events),
        }

    async def recorded_reranker_case(self, case_id: str = "direct_08") -> dict[str, object]:
        """Return a committed historical reranker flip, never a live measurement."""

        repo_root = self.manifest_path.parents[2]
        path = repo_root / "projects" / "document_intelligence_service" / "eval" / "results" / "week2_stabilization_v1" / "reranker_flips.jsonl"
        if not path.is_file():
            return {
                "status": "unavailable",
                "case_id": case_id,
                "label": "RECORDED FROZEN EVALUATION CASE",
            }
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("case_id") == case_id:
                return {
                    "status": "recorded",
                    "label": "RECORDED FROZEN EVALUATION CASE",
                    "case_id": case_id,
                    "artifact_path": str(path.relative_to(repo_root)),
                    "data": payload,
                }
        return {
            "status": "not_found",
            "label": "RECORDED FROZEN EVALUATION CASE",
            "case_id": case_id,
        }

    async def _run_retrieval_only(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[str],
        tenant_id: str,
        acl_tags: Sequence[str],
        reranker_enabled: bool,
    ) -> tuple[QueryExecutionResult | None, tuple[TraceEvent, ...]]:
        """Use the real retrieval adapter and gate without invoking a model."""

        if self._retrieval_service is None:
            return None, ()
        search = getattr(self._retrieval_service, "search", None)
        if not callable(search):
            return None, ()
        from .query_service import _empty_retrieval, assess_answerability

        prompt_safety = getattr(self._query_service, "_prompt_safety", None)
        if prompt_safety is not None and prompt_safety.evaluate(question).blocked:
            retrieval = _empty_retrieval(mode)
            gate = assess_answerability(
                question=question,
                retrieval=retrieval,
                answerability=self._query_service._answerability,
                prompt_safety=prompt_safety,
            )
            return (
                QueryExecutionResult(
                    decision=gate.decision,
                    answer=None,
                    no_answer_reason=gate.reason,
                    sources=(),
                    retrieval=retrieval,
                    provider=None,
                    model=None,
                    llm_ms=0.0,
                    total_ms=0.0,
                    answerability=gate,
                    warnings=(),
                ),
                (),
            )
        retrieval = await asyncio.to_thread(
            search,
            question=question,
            mode=mode,
            top_k=top_k,
            document_ids=document_ids,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
                reranker_enabled=reranker_enabled,
        )
        evidence_safety = getattr(self._query_service, "_evidence_safety", None)
        blocked_evidence = False
        if evidence_safety is not None:
            safety = evidence_safety.filter(retrieval.candidates)
            window = retrieval.candidate_window or retrieval.candidates
            window_safety = evidence_safety.filter(window)
            blocked_evidence = bool(
                safety.blocked_source_ids or window_safety.blocked_source_ids
            )
            if blocked_evidence:
                safe_ids = {item.source_id for item in safety.safe_evidence}
                safe_window_ids = {
                    item.source_id for item in window_safety.safe_evidence
                }
                retrieval = replace(
                    retrieval,
                    candidates=tuple(
                        item
                        for item in retrieval.candidates
                        if item.source_id in safe_ids
                    ),
                    candidate_window=tuple(
                        item for item in window
                        if item.source_id in safe_window_ids
                    ),
                )
        gate = assess_answerability(
            question=question,
            retrieval=retrieval,
            answerability=self._query_service._answerability,
            prompt_safety=prompt_safety,
            evidence_safety_blocked=blocked_evidence and not retrieval.candidates,
        )
        result = QueryExecutionResult(
            decision=gate.decision,
            answer=None,
            no_answer_reason=gate.reason,
            sources=retrieval.candidates if gate.decision is Decision.ANSWERED else (),
            retrieval=retrieval,
            provider=None,
            model=None,
            llm_ms=0.0,
            total_ms=retrieval.embedding_ms + retrieval.search_ms + retrieval.rerank_ms,
            answerability=gate,
            warnings=(),
        )
        return result, ()


def _journey(
    *,
    case: GoldCase,
    resolved_gold: Sequence[ResolvedGoldEvidence],
    result: QueryExecutionResult | None,
    events: Sequence[TraceEvent],
    mode: RetrievalMode,
    retrieval_only: bool = False,
) -> list[dict[str, object]]:
    """Follow each trusted source through all observable pipeline stages."""

    debug = {item.source_id: item for item in result.retrieval.debug_candidates} if result else {}
    event_candidates = _event_candidates(events)
    evidence_ids = _event_source_ids(events, "evidence_selection")
    prompt_ids = _event_source_ids(events, "prompt_build")
    prompt_event = _latest_event(events, "prompt_build")
    prompt_observed = _prompt_membership_observed(prompt_event)
    prompt_skipped = prompt_event is not None and prompt_event.get("status") == "skipped"
    if result is not None and result.prompt_pack is not None:
        prompt_ids = set(result.prompt_pack.included_source_ids)
        prompt_observed = True
    # Only stage events are authoritative for evidence/prompt membership.
    # Final result sources are a projection, not proof that a source entered
    # the prompt builder.
    retrieval_mode = result.retrieval.mode if result is not None else mode.value
    dense_applicable = retrieval_mode in {
        RetrievalMode.DENSE.value,
        RetrievalMode.HYBRID.value,
    }
    sparse_applicable = retrieval_mode in {
        RetrievalMode.BM25.value,
        RetrievalMode.HYBRID.value,
    }
    rrf_applicable = retrieval_mode == RetrievalMode.HYBRID.value
    if retrieval_only and result is not None:
        # Retrieval-only diagnosis has no prompt_build event by design, but
        # its result sources are the explicit pre-generation evidence set.
        evidence_ids.update(item.source_id for item in result.sources)
    rows: list[dict[str, object]] = []
    for item in resolved_gold:
        candidate = _journey_candidate(item, debug, event_candidates)
        accepted_ids = {source.source_id for source in item.accepted_sources}
        rows.append(
            {
                "gold": item.as_dict(),
                "dense": _rank(candidate, "dense_rank") if dense_applicable else "N/A",
                "bm25": _rank(candidate, "sparse_rank") if sparse_applicable else "N/A",
                "rrf": _rank(candidate, "fusion_rank") if rrf_applicable else "N/A",
                "reranker": (
                    _rank(candidate, "rerank_rank")
                    if _reranker_was_enabled(result, events)
                    else "N/A"
                ),
                "evidence": bool(accepted_ids & evidence_ids),
                "prompt": (
                    "N/A"
                    if prompt_skipped
                    else bool(accepted_ids & prompt_ids)
                ) if prompt_observed else "UNKNOWN",
                "prompt_observed": prompt_observed,
                "selected_source": bool(accepted_ids & {
                    source.source_id for source in result.sources
                }) if result else False,
                "scores": _scores(candidate),
            }
        )
    return rows


def _journey_candidate(
    item: ResolvedGoldEvidence,
    debug: Mapping[str, RetrievalDebugCandidate],
    event_candidates: Mapping[str, Mapping[str, object]],
) -> RetrievalDebugCandidate | Mapping[str, object] | None:
    """Choose the strongest observed candidate among overlap-equivalent IDs."""

    observed = [
        debug[source.source_id]
        for source in item.accepted_sources
        if source.source_id in debug
    ]
    if observed:
        return min(
            observed,
            key=lambda candidate: (
                candidate.fusion_rank is None,
                candidate.fusion_rank or 0,
                candidate.source_id,
            ),
        )
    for source in item.accepted_sources:
        candidate = event_candidates.get(source.source_id)
        if candidate is not None:
            return candidate
    return None


def _fact_coverage(
    *,
    expected_answer: str,
    resolved_gold: Sequence[ResolvedGoldEvidence],
    result: QueryExecutionResult | None,
    events: Sequence[TraceEvent],
    actual_answer: str | None,
) -> dict[str, object]:
    """Compare hard expected facts against each real evidence boundary.

    Source membership is deliberately insufficient here: a source can be
    selected and still be truncated before the fact reaches the model.  The
    prompt boundary therefore uses the actual PromptPackResult fragments when
    available, with trace fragments as the only compatible fallback.
    """

    expected_check = check_answer(
        expected=expected_answer,
        actual="",
        mode=AnswerCheckMode.FACT_AWARE,
    )
    facts = expected_check.required_facts
    if not facts:
        return {"facts": [], "observed": False}

    trusted_text = " ".join(
        source.text
        for item in resolved_gold
        for source in item.accepted_sources
    )
    selected_text = " ".join(
        source.text for source in (result.sources if result is not None else ())
    )
    packed_text = ""
    packed_source_ids: set[str] = set()
    prompt_pack = result.prompt_pack if result is not None else None
    if prompt_pack is not None:
        packed_text = " ".join(
            fragment.included_text for fragment in prompt_pack.fragments
        )
        packed_source_ids = set(prompt_pack.included_source_ids)
    else:
        prompt_event = _latest_event(events, "prompt_build")
        details = prompt_event.get("details") if prompt_event else None
        if isinstance(details, Mapping):
            fragments = details.get("fragments")
            if isinstance(fragments, list):
                packed_text = " ".join(
                    fragment.get("included_text", "")
                    for fragment in fragments
                    if isinstance(fragment, Mapping)
                    and isinstance(fragment.get("included_text"), str)
                )
            raw_ids = details.get("included_source_ids")
            if isinstance(raw_ids, list):
                packed_source_ids = {
                    item for item in raw_ids if isinstance(item, str)
                }

    rows: list[dict[str, object]] = []
    for fact in facts:
        value = fact.value
        rows.append(
            {
                "type": fact.fact_type,
                "value": value,
                "in_trusted_evidence": _fact_present(value, trusted_text),
                "in_selected_evidence": _fact_present(value, selected_text),
                "in_packed_prompt": _fact_present(value, packed_text),
                "in_final_answer": _fact_present(value, actual_answer or ""),
            }
        )
    return {
        "facts": rows,
        "observed": prompt_pack is not None or bool(packed_source_ids),
        "packed_source_ids": sorted(packed_source_ids),
    }


def _fact_present(value: str, text: str) -> bool:
    """Use the same deterministic fact matcher as Answer Check."""

    if not text:
        return False
    checked = check_answer(
        expected=value,
        actual=text,
        mode=AnswerCheckMode.FACT_AWARE,
    )
    return bool(checked.required_facts) and all(
        fact.matched for fact in checked.required_facts
    )


def _attribute(
    *,
    case: GoldCase,
    verdict: DiagnosticVerdict,
    actual_decision: str,
    actual_reason: str | None,
    actual_answer: str | None,
    result: QueryExecutionResult | None,
    journey: Sequence[Mapping[str, object]],
    events: Sequence[TraceEvent],
    error_payload: Mapping[str, object] | None,
    answer_check_verdict: AnswerCheckVerdict | None = None,
    fact_coverage: Mapping[str, object] | None = None,
) -> tuple[DiagnosticRootCause, str | None, str]:
    """Return only a first divergence supported by trusted gold observations."""

    if case.expected_decision == "SECURITY_POLICY":
        if actual_decision == "NO_ANSWER" and actual_reason == "SECURITY_POLICY":
            return DiagnosticRootCause.SECURITY_POLICY_CORRECT, None, "Security policy decision matched the trusted case."
        return DiagnosticRootCause.REVIEW_REQUIRED, "prompt_safety", "The trusted security case did not produce the expected policy decision."
    if case.expected_decision == "NO_ANSWER":
        if actual_decision == "NO_ANSWER":
            if (
                case.expected_reason is not None
                and actual_reason != case.expected_reason
            ):
                return (
                    DiagnosticRootCause.REVIEW_REQUIRED,
                    "answerability",
                    f"Expected no-answer reason {case.expected_reason}, "
                    f"observed {actual_reason or 'none'}.",
                )
            return DiagnosticRootCause.NO_ANSWER_CORRECT, None, "The expected no-answer decision was preserved and generation was skipped."
        return DiagnosticRootCause.REVIEW_REQUIRED, "answerability", "A no-answer case produced an answer; no positive gold evidence was supplied for stage blame."

    if not journey:
        return DiagnosticRootCause.DATASET_GOLD_INVALID, "dataset", "No trusted gold evidence resolved for an answerable case."

    mode = result.retrieval.mode if result is not None else _event_mode(events)
    dense_count = sum(1 for row in journey if _observed_rank(row.get("dense")))
    sparse_count = sum(1 for row in journey if _observed_rank(row.get("bm25")))
    rrf_count = sum(1 for row in journey if _observed_rank(row.get("rrf")))
    total = len(journey)
    if mode == RetrievalMode.DENSE.value and dense_count < total:
        return (
            DiagnosticRootCause.RETRIEVAL_MISS,
            "candidate_retrieval",
            "Trusted gold evidence was absent from the Dense-only candidate window.",
        )
    if mode == RetrievalMode.BM25.value and sparse_count < total:
        return (
            DiagnosticRootCause.RETRIEVAL_MISS,
            "candidate_retrieval",
            "Trusted gold evidence was absent from the BM25-only candidate window.",
        )
    if (
        mode == RetrievalMode.HYBRID.value
        and dense_count < total
        and sparse_count == total
        and rrf_count == total
    ):
        recovery_note = "Dense branch missed at least one gold source, but BM25 and Hybrid RRF recovered it."
    elif (
        mode == RetrievalMode.HYBRID.value
        and sparse_count < total
        and dense_count == total
        and rrf_count == total
    ):
        recovery_note = "BM25 branch missed at least one gold source, but Dense and Hybrid RRF recovered it."
    else:
        recovery_note = ""

    if mode == RetrievalMode.HYBRID.value and rrf_count < total:
        if any(
            not _observed_rank(row.get("dense"))
            and not _observed_rank(row.get("bm25"))
            for row in journey
        ):
            return DiagnosticRootCause.RETRIEVAL_MISS, "candidate_retrieval", "Gold evidence was absent from both Dense and BM25 candidate branches."
        return DiagnosticRootCause.FUSION_LOSS, "rrf_fusion", "Gold evidence entered a branch but was absent from the RRF candidate window."
    reranker_enabled = result is not None and result.retrieval.reranker_enabled
    if reranker_enabled and any(row.get("reranker") == "—" for row in journey):
        return DiagnosticRootCause.RERANKER_LOSS, "reranker", "Gold evidence was in the RRF window but disappeared after reranking."
    if any(not bool(row.get("evidence")) for row in journey):
        blocked = _blocked_count(events)
        if blocked:
            return DiagnosticRootCause.EVIDENCE_SAFETY_BLOCK, "evidence", "Evidence safety blocked candidates before the final evidence set."
        return DiagnosticRootCause.EVIDENCE_SELECTION_LOSS, "evidence_selection", "Gold evidence survived retrieval but was not selected as final evidence."
    if result is not None and result.decision is Decision.NO_ANSWER:
        return DiagnosticRootCause.ANSWERABILITY_FALSE_NEGATIVE, "answerability", "Trusted gold evidence was selected, but the calibrated answerability gate rejected the query."
    if fact_coverage and fact_coverage.get("facts"):
        facts = fact_coverage["facts"]
        if isinstance(facts, list):
            if any(isinstance(fact, Mapping) and not fact.get("in_trusted_evidence") for fact in facts):
                return (
                    DiagnosticRootCause.DATASET_GOLD_INVALID,
                    "dataset",
                    "A required expected fact was not present in the trusted evidence label.",
                )
            if any(isinstance(fact, Mapping) and not fact.get("in_selected_evidence") for fact in facts):
                return (
                    DiagnosticRootCause.EVIDENCE_SELECTION_LOSS,
                    "evidence_selection",
                    "The trusted evidence contains a required fact, but the selected evidence set does not.",
                )
            if any(isinstance(fact, Mapping) and not fact.get("in_packed_prompt") for fact in facts):
                return (
                    DiagnosticRootCause.PROMPT_CONSTRUCTION_LOSS,
                    "prompt",
                    "A required fact was selected, but it is absent from the actual packed prompt fragment text.",
                )
            if any(isinstance(fact, Mapping) and not fact.get("in_final_answer") for fact in facts):
                return (
                    DiagnosticRootCause.GENERATION_CLAIM_MISMATCH,
                    "generation",
                    "A required fact reached the actual packed prompt, but the final answer omitted it.",
                )
    if any(
        row.get("prompt_observed") is True and row.get("prompt") is False
        for row in journey
    ):
        return (
            DiagnosticRootCause.PROMPT_CONSTRUCTION_LOSS,
            "prompt",
            "Trusted gold evidence was selected, but its required fact was absent from the actual packed prompt.",
        )
    if error_payload is not None and error_payload.get("stage") == "llm":
        return DiagnosticRootCause.GENERATION_DEPENDENCY_FAILURE, "generation", "Answerability passed and evidence was available, but the generation dependency failed."
    if (
        error_payload is not None
        and error_payload.get("code") != "RETRIEVAL_ONLY"
    ):
        return DiagnosticRootCause.REVIEW_REQUIRED, str(error_payload.get("stage") or "pipeline"), "The live run failed before a trusted stage attribution could be completed."
    if error_payload is not None and error_payload.get("code") == "RETRIEVAL_ONLY":
        return DiagnosticRootCause.PASS, None, "Retrieval, evidence selection and answerability passed; generation was intentionally not invoked."
    if any(row.get("prompt_observed") is False for row in journey):
        return DiagnosticRootCause.REVIEW_REQUIRED, "prompt", "The run did not expose actual prompt membership; generation attribution is not proven."
    if answer_check_verdict is not None:
        if answer_check_verdict is AnswerCheckVerdict.PASS:
            if recovery_note:
                return DiagnosticRootCause.PASS, None, recovery_note
            return DiagnosticRootCause.PASS, None, "Trusted evidence reached the prompt and the expected answer check passed."
        if answer_check_verdict is AnswerCheckVerdict.REVIEW_REQUIRED:
            return DiagnosticRootCause.REVIEW_REQUIRED, "generation", "Trusted evidence reached generation, but the answer comparison requires human review."
        return DiagnosticRootCause.GENERATION_CLAIM_MISMATCH, "generation", "Trusted evidence reached the actual prompt, but the final answer did not satisfy the expected answer check."
    if verdict is DiagnosticVerdict.PASS:
        if recovery_note:
            return DiagnosticRootCause.PASS, None, recovery_note
        return DiagnosticRootCause.PASS, None, "All trusted claims passed and the gold evidence reached the grounded answer path."
    if result is not None and result.decision is Decision.ANSWERED:
        claims = compare_claims(
            expected_claims=case.expected_claims,
            forbidden_claims=case.forbidden_claims,
            answer=actual_answer,
        )
        passed = claims.get("expected_claims_passed")
        expected = claims.get("expected_claim_count")
        if (
            isinstance(passed, int)
            and isinstance(expected, int)
            and passed == expected
        ):
            return DiagnosticRootCause.CORRECT_BUT_UNGROUNDED, "generation", "The structured answer claims matched, but the trusted gold evidence was not the sole supported prompt path."
        return DiagnosticRootCause.GENERATION_CLAIM_MISMATCH, "generation", "Evidence reached generation, but one or more trusted claims were missing or forbidden."
    return DiagnosticRootCause.REVIEW_REQUIRED, None, "The run requires human review; no deterministic first divergence was proven."


def _diagnostic_verdict_for_answer_check(verdict: AnswerCheckVerdict) -> DiagnosticVerdict:
    """Map the independent answer check into the diagnostic result space."""

    if verdict is AnswerCheckVerdict.PASS:
        return DiagnosticVerdict.PASS
    if verdict is AnswerCheckVerdict.REVIEW_REQUIRED:
        return DiagnosticVerdict.REVIEW_REQUIRED
    return DiagnosticVerdict.FAIL


def _stage_coverage(
    journey: Sequence[Mapping[str, object]],
    mode: str,
) -> dict[str, str]:
    del mode
    result: dict[str, str] = {}
    for stage in ("dense", "bm25", "rrf", "reranker", "evidence", "prompt"):
        applicable = [item for item in journey if item.get(stage) != "N/A"]
        result[stage] = (
            "N/A"
            if not applicable
            else f"{sum(1 for item in applicable if _stage_present(item, stage))}/{len(applicable)}"
        )
    return result


def _stage_present(item: Mapping[str, object], stage: str) -> bool:
    value = item.get(stage)
    if isinstance(value, bool):
        return value
    return value is not None and value not in {"—", "N/A", "OFF"}


def _event_mode(events: Sequence[TraceEvent]) -> str:
    for event in events:
        details = event.get("details")
        if isinstance(details, Mapping) and isinstance(details.get("mode"), str):
            return str(details["mode"])
    return RetrievalMode.HYBRID.value


def _event_source_ids(events: Sequence[TraceEvent], stage: str) -> set[str]:
    ids: set[str] = set()
    for event in events:
        if event.get("stage") != stage:
            continue
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        raw = details.get("included_source_ids")
        if isinstance(raw, list):
            ids.update(item for item in raw if isinstance(item, str))
        raw = details.get("evidence_ids")
        if isinstance(raw, list):
            ids.update(item for item in raw if isinstance(item, str))
        raw_evidence = details.get("evidence")
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if isinstance(item, Mapping) and isinstance(item.get("source_id"), str):
                    ids.add(item["source_id"])
    return ids


def _latest_event(events: Sequence[TraceEvent], stage: str) -> TraceEvent | None:
    """Return the final event for one stage without trusting result projections."""

    for event in reversed(events):
        if event.get("stage") == stage:
            return event
    return None


def _prompt_membership_observed(event: TraceEvent | None) -> bool:
    """Tell whether the trace contains actual packer membership metadata."""

    if event is None:
        return False
    details = event.get("details")
    if not isinstance(details, Mapping):
        return False
    if details.get("membership_observed") is True:
        return True
    # Compatibility for pre-V7 test doubles whose explicit evidence_ids were
    # already the prompt-builder output. Real V7 events use included_source_ids.
    return isinstance(details.get("included_source_ids"), list) or isinstance(
        details.get("evidence_ids"), list
    )


def _blocked_count(events: Sequence[TraceEvent]) -> int:
    for event in reversed(events):
        if event.get("stage") != "evidence_selection":
            continue
        details = event.get("details")
        if isinstance(details, Mapping) and isinstance(details.get("blocked_candidates"), int):
            return int(details["blocked_candidates"])
    return 0


def _rank(candidate: RetrievalDebugCandidate | Mapping[str, object] | None, field: str) -> int | str:
    if candidate is None:
        return "—"
    value = candidate.get(field) if isinstance(candidate, Mapping) else getattr(candidate, field)
    return value if isinstance(value, int) else "—"


def _observed_rank(value: object) -> bool:
    """Return whether a journey value represents a real integer rank."""

    return isinstance(value, int) and not isinstance(value, bool)


def _scores(
    candidate: RetrievalDebugCandidate | Mapping[str, object] | None,
) -> dict[str, float | None]:
    if candidate is None:
        return {"dense": None, "bm25": None, "rrf": None, "rerank": None}
    if isinstance(candidate, Mapping):
        return {
            "dense": _object_float(candidate.get("dense_score")),
            "bm25": _object_float(candidate.get("sparse_score")),
            "rrf": _object_float(candidate.get("fused_score")),
            "rerank": _object_float(candidate.get("rerank_score")),
        }
    return {
        "dense": candidate.dense_score,
        "bm25": candidate.sparse_score,
        "rrf": candidate.fused_score,
        "rerank": candidate.rerank_score,
    }


def _event_candidates(events: Sequence[TraceEvent]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for event in events:
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        for key in ("candidates", "before", "after", "evidence"):
            raw = details.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if isinstance(item, Mapping) and isinstance(item.get("source_id"), str):
                    existing = dict(result.get(item["source_id"], {}))
                    existing.update(item)
                    result[item["source_id"]] = existing
    return result


def _reranker_was_enabled(
    result: QueryExecutionResult | None,
    events: Sequence[TraceEvent],
) -> bool:
    if result is not None:
        return result.retrieval.reranker_enabled
    return any(
        event.get("stage") == "reranker"
        and event.get("status") in {"running", "passed"}
        for event in events
    )


def _object_float(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _actual_fields(
    result: QueryExecutionResult | None,
    error: Mapping[str, object] | None,
) -> tuple[str, str | None, str | None]:
    if result is not None:
        return (
            result.decision.value.upper(),
            result.no_answer_reason.value if result.no_answer_reason else None,
            result.answer,
        )
    if error is not None:
        return "PIPELINE_FAILED", str(error.get("reason")) if error.get("reason") else None, None
    return "PIPELINE_FAILED", None, None


def _sources_payload(result: QueryExecutionResult | None) -> list[dict[str, object]]:
    if result is None:
        return []
    return [
        {
            "source_id": item.source_id,
            "document_id": item.document_id,
            "title": item.title,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "parent_id": item.parent_id,
            "excerpt": " ".join(item.context_text.split())[:400],
        }
        for item in result.sources
    ]


def claims_expected(case: GoldCase) -> list[dict[str, object]]:
    return [
        {
            "claim_id": claim.claim_id,
            "type": claim.claim_type,
            "value": claim.value,
        }
        for claim in case.expected_claims
    ]


def _request_id(events: Sequence[TraceEvent]) -> str | None:
    # The regular demo transport owns request IDs. Direct diagnostic runs expose
    # the trace itself, so this remains optional and never becomes an oracle.
    for event in events:
        details = event.get("details")
        value = details.get("request_id") if isinstance(details, Mapping) else None
        if isinstance(value, str):
            return value
    return None


def _receipt_dict(key: str, filename: str, receipt: IngestionReceipt) -> dict[str, object]:
    return {
        "document_key": key,
        "filename": filename,
        "document_id": receipt.document_id,
        "version_id": receipt.version_id,
        "job_id": receipt.job_id,
        "status": receipt.status.value,
        "idempotent_hit": receipt.idempotent_hit,
    }


def _invalid_result(case: GoldCase, message: str) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected": {
            "decision": case.expected_decision,
            "answer": case.expected_answer,
            "claims": claims_expected(case),
            "gold_evidence": [],
        },
        "actual": {"decision": "NOT_RUN", "answer": None, "reason": message, "sources": []},
        "verdict": DiagnosticVerdict.REVIEW_REQUIRED.value,
        "root_cause": DiagnosticRootCause.DATASET_GOLD_INVALID.value,
        "first_divergence": "dataset",
        "attribution_note": message,
        "claims": compare_claims(
            expected_claims=case.expected_claims,
            forbidden_claims=case.forbidden_claims,
            answer=None,
        ),
        "semantic_similarity": None,
        "semantic_similarity_note": "Not computed because trusted gold could not be resolved.",
        "gold_evidence_journey": [],
        "stage_coverage": {},
        "trace_events": [],
        "error": {"code": DiagnosticRootCause.DATASET_GOLD_INVALID.value, "message": message},
        "retrieval_only": False,
        "request_id": None,
    }
