"""Deterministic Gold Diagnostic attribution tests."""

from collections.abc import Sequence
import asyncio
from typing import cast
from pathlib import Path

from projects.document_intelligence_service.app.application.gold_diagnostic import (
    GoldDatasetInvalid,
    GoldDiagnosticService,
    ResolvedGoldEvidence,
)
from projects.document_intelligence_service.app.application.query_service import (
    QueryExecutionResult,
)
from projects.document_intelligence_service.app.application.document_service import (
    DocumentService,
)
from projects.document_intelligence_service.app.application.ingestion_service import (
    IngestionService,
)
from projects.document_intelligence_service.app.domain.answerability import (
    AnswerabilityDecision,
    AnswerabilityPolicy,
)
from projects.document_intelligence_service.app.domain.entities import (
    Decision,
    NoAnswerReason,
    RetrievalMode,
)
from projects.document_intelligence_service.app.domain.errors import (
    ErrorCode,
    ServiceError,
)
from projects.document_intelligence_service.app.domain.gold_diagnostic import (
    DiagnosticRootCause,
    DiagnosticVerdict,
    GoldLocator,
    compare_decision,
    compare_claims,
    load_gold_cases,
)
from projects.document_intelligence_service.app.domain.retrieval import (
    RetrievedChunk,
    RetrievalDebugCandidate,
    RetrievalResult,
)


ROOT = Path(__file__).parents[4]
MANIFEST = ROOT / "data/evaluations/atlas_orion_demo/atlas_orion_diagnostic_cases.json"


def source(source_id: str = "ops:ver:parent:001:child:001") -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        document_id="doc_ops",
        version_id="ver_ops",
        parent_id="ops:ver:parent:001",
        title="atlas_orion_operations.pdf",
        text="Program owner: Deniz Aral. Emergency override code: ZX-417.",
        page_start=1,
        page_end=1,
        score=0.8,
        rank=1,
        dense_rank=1,
        sparse_rank=1,
        fusion_rank=1,
        dense_score=0.8,
        sparse_score=2.0,
        fused_score=0.03,
        chunking_profile="generic_v1",
    )


def debug(
    item: RetrievedChunk,
    *,
    dense_rank: int | None = 1,
    sparse_rank: int | None = 1,
    fusion_rank: int | None = 1,
    rerank_rank: int | None = None,
    selected: bool = True,
) -> RetrievalDebugCandidate:
    return RetrievalDebugCandidate(
        source_id=item.source_id,
        retrieval_rank=fusion_rank,
        rerank_rank=rerank_rank,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=item.dense_score,
        sparse_score=item.sparse_score,
        fused_score=item.fused_score,
        rerank_score=0.5 if rerank_rank else None,
        matched_terms=("orion",),
        document_id=item.document_id,
        title=item.title,
        page_start=1,
        page_end=1,
        excerpt=item.text,
        fusion_rank=fusion_rank,
        selected_as_evidence=selected,
        rank_delta=None,
    )


def result(
    *,
    item: RetrievedChunk,
    decision: Decision,
    debug_item: RetrievalDebugCandidate,
    mode: str = "hybrid",
    reranker_enabled: bool = False,
    sources: Sequence[RetrievedChunk] = (),
    reason: NoAnswerReason | None = None,
    answer: str | None = None,
) -> QueryExecutionResult:
    retrieval = RetrievalResult(
        mode=mode,
        candidates=tuple(sources),
        dense_candidates=1,
        sparse_candidates=1,
        rrf_candidates=1,
        embedding_ms=1.0,
        search_ms=2.0,
        reranked_candidates=len(sources) if reranker_enabled else 0,
        rerank_ms=1.0 if reranker_enabled else 0.0,
        candidate_window=(item,),
        debug_candidates=(debug_item,),
        reranker_enabled=reranker_enabled,
    )
    return QueryExecutionResult(
        decision=decision,
        answer=answer,
        no_answer_reason=reason,
        sources=tuple(sources),
        retrieval=retrieval,
        provider="test" if answer else None,
        model="test-model" if answer else None,
        llm_ms=5.0 if answer else 0.0,
        total_ms=9.0,
        answerability=AnswerabilityDecision(
            decision=decision,
            reason=reason,
            top_score=0.8,
            score_margin=0.2,
            coverage_ratio=1.0,
            policy_profile="generic_v1",
        ),
        warnings=(),
    )


class FakeResolver:
    def __init__(self, item: RetrievedChunk) -> None:
        self.item = item
        self.document_key_calls: list[tuple[str, ...]] = []

    async def resolve(
        self,
        locators: Sequence[GoldLocator],
        *,
        tenant_id: str = "default",
    ) -> tuple[ResolvedGoldEvidence, ...]:
        del tenant_id
        return tuple(
            ResolvedGoldEvidence(
                locator=locator,
                document_id=self.item.document_id,
                version_id=self.item.version_id,
                source=self.item,
            )
            for locator in locators
        )

    async def resolve_document_ids(
        self,
        keys: Sequence[str],
        *,
        tenant_id: str = "default",
    ) -> tuple[str, ...]:
        del tenant_id
        self.document_key_calls.append(tuple(keys))
        return (self.item.document_id,)


class InvalidResolver(FakeResolver):
    async def resolve(
        self,
        locators: Sequence[GoldLocator],
        *,
        tenant_id: str = "default",
    ) -> tuple[ResolvedGoldEvidence, ...]:
        del locators, tenant_id
        raise GoldDatasetInvalid("gold locator is ambiguous")


class FakeRetrievalService:
    def __init__(self, item: RetrievedChunk) -> None:
        self.item = item

    def search(self, **kwargs: object) -> RetrievalResult:
        del kwargs
        return RetrievalResult(
            mode="hybrid",
            candidates=(self.item,),
            dense_candidates=1,
            sparse_candidates=1,
            rrf_candidates=1,
            embedding_ms=1.0,
            search_ms=1.0,
            candidate_window=(self.item,),
            debug_candidates=(debug(self.item),),
        )


class FakeQueryService:
    def __init__(
        self,
        query_result: object | None,
        *,
        error: ServiceError | None = None,
        emit_events: bool = True,
    ) -> None:
        self._answerability = AnswerabilityPolicy(
            min_dense_score=0.0,
            min_sparse_score=0.0,
            min_coverage=0.0,
        )
        self.query_result = query_result
        self.error = error
        self.emit_events = emit_events

    async def execute(self, **kwargs: object) -> object:
        trace = kwargs.get("trace")
        if self.emit_events and callable(trace):
            item = (
                self.query_result.retrieval.debug_candidates[0]
                if isinstance(self.query_result, QueryExecutionResult)
                and self.query_result.retrieval.debug_candidates
                else None
            )
            source_id = item.source_id if item is not None else "ops:ver:parent:001:child:001"
            candidate = {
                "source_id": source_id,
                "dense_rank": item.dense_rank if item is not None else 1,
                "sparse_rank": item.sparse_rank if item is not None else 1,
                "fusion_rank": item.fusion_rank if item is not None else 1,
                "rerank_rank": item.rerank_rank if item is not None else None,
            }
            trace("dense_retrieval", "passed", "dense", {"candidates": [candidate]}, 1.0)
            trace("sparse_retrieval", "passed", "sparse", {"candidates": [candidate]}, 1.0)
            trace("rrf_fusion", "passed", "rrf", {"candidates": [candidate]}, 1.0)
            selected = item is None or item.selected_as_evidence
            trace(
                "evidence_selection",
                "passed" if selected else "failed",
                "evidence",
                {"evidence": [{"source_id": source_id}]} if selected else {"evidence": []},
                1.0,
            )
            trace(
                "prompt_build",
                "passed",
                "prompt",
                {"evidence_ids": [source_id]} if selected else {"evidence_ids": []},
                1.0,
            )
        if self.error is not None:
            trace = kwargs.get("trace")
            if callable(trace):
                trace("llm", "failed", "generation", {"reason": "EMPTY_RESPONSE"}, 1.0)
            raise self.error
        return self.query_result


def service_for(
    query_service: FakeQueryService,
    *,
    resolver: FakeResolver | None = None,
    retrieval_service: FakeRetrievalService | None = None,
) -> GoldDiagnosticService:
    return GoldDiagnosticService(
        manifest_path=MANIFEST,
        asset_dir=MANIFEST.parent,
        document_service=cast(
            DocumentService,
            object(),
        ),  # resolver owns catalog behavior in this unit test
        ingestion_service=cast(IngestionService, object()),
        query_service=query_service,  # type: ignore[arg-type]
        resolver=resolver or FakeResolver(source()),  # type: ignore[arg-type]
        retrieval_service=retrieval_service,
    )


def test_dense_branch_miss_is_reported_as_recovered() -> None:
    item = source()
    query = FakeQueryService(
        result(
            item=item,
            decision=Decision.ANSWERED,
            debug_item=debug(item, dense_rank=None),
            sources=(item,),
            answer="Deniz Aral.",
        )
    )
    diagnostic = service_for(query)
    report = asyncio.run(diagnostic.run_case(case_id="direct_owner"))
    assert report["root_cause"] == DiagnosticRootCause.PASS.value
    assert "Dense branch missed" in str(report["attribution_note"])


def test_retrieval_miss_is_provable_only_when_both_branches_miss() -> None:
    item = source()
    query = FakeQueryService(
        result(
            item=item,
            decision=Decision.NO_ANSWER,
            debug_item=debug(
                item,
                dense_rank=None,
                sparse_rank=None,
                fusion_rank=None,
                selected=False,
            ),
            reason=NoAnswerReason.NO_EVIDENCE,
        )
    )
    report = asyncio.run(service_for(query).run_case(case_id="direct_owner"))
    assert report["root_cause"] == DiagnosticRootCause.RETRIEVAL_MISS.value
    assert report["first_divergence"] == "candidate_retrieval"


def test_evidence_selection_loss_is_distinct_from_retrieval_loss() -> None:
    item = source()
    query = FakeQueryService(
        result(
            item=item,
            decision=Decision.NO_ANSWER,
            debug_item=debug(item, selected=False),
            reason=NoAnswerReason.NO_EVIDENCE,
        )
    )
    report = asyncio.run(service_for(query).run_case(case_id="direct_owner"))
    assert report["root_cause"] == DiagnosticRootCause.EVIDENCE_SELECTION_LOSS.value
    assert report["first_divergence"] == "evidence_selection"


def test_fusion_loss_is_the_first_provable_divergence() -> None:
    item = source()
    query = FakeQueryService(
        result(
            item=item,
            decision=Decision.NO_ANSWER,
            debug_item=debug(item, fusion_rank=None, selected=False),
            reason=NoAnswerReason.NO_EVIDENCE,
        )
    )
    report = asyncio.run(service_for(query).run_case(case_id="direct_owner"))
    assert report["root_cause"] == DiagnosticRootCause.FUSION_LOSS.value
    assert report["first_divergence"] == "rrf_fusion"


def test_reranker_loss_and_answerability_false_negative_are_distinct() -> None:
    item = source()
    rerank_report = asyncio.run(service_for(
        FakeQueryService(
            result(
                item=item,
                decision=Decision.NO_ANSWER,
                debug_item=debug(item, rerank_rank=None, selected=False),
                reranker_enabled=True,
                reason=NoAnswerReason.NO_EVIDENCE,
            )
        )
    ).run_case(case_id="direct_owner", reranker_enabled=True))
    assert rerank_report["root_cause"] == DiagnosticRootCause.RERANKER_LOSS.value

    gate_report = asyncio.run(service_for(
        FakeQueryService(
            result(
                item=item,
                decision=Decision.NO_ANSWER,
                debug_item=debug(item, selected=True),
                reason=NoAnswerReason.LOW_RELEVANCE,
            )
        ),
    ).run_case(case_id="direct_owner"))
    assert gate_report["root_cause"] == DiagnosticRootCause.ANSWERABILITY_FALSE_NEGATIVE.value


def test_generation_dependency_failure_keeps_gold_journey() -> None:
    query = FakeQueryService(
        None,
        error=ServiceError(
            code=ErrorCode.DEPENDENCY_UNAVAILABLE,
            message="generation failed",
            stage="llm",
            reason="EMPTY_RESPONSE",
        ),
    )
    report = asyncio.run(service_for(query).run_case(case_id="direct_owner"))
    assert report["root_cause"] == DiagnosticRootCause.GENERATION_DEPENDENCY_FAILURE.value
    assert report["gold_evidence_journey"]


def test_retrieval_only_keeps_pre_generation_pass_unattributed() -> None:
    item = source()
    report = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.ANSWERED,
                    debug_item=debug(item),
                    sources=(item,),
                    answer="Deniz Aral.",
                )
            ),
            retrieval_service=FakeRetrievalService(item),
        ).run_case(case_id="direct_owner", retrieval_only=True)
    )
    assert report["verdict"] == "REVIEW_REQUIRED"
    assert report["root_cause"] == DiagnosticRootCause.REVIEW_REQUIRED.value
    assert report["first_divergence"] == "generation"
    assert "no generation claim verdict" in str(report["attribution_note"])


def test_expected_no_answer_reason_mismatch_requires_review() -> None:
    item = source()
    no_answer = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.NO_ANSWER,
                    debug_item=debug(item, selected=False),
                    reason=NoAnswerReason.NO_EVIDENCE,
                )
            )
        ).run_case(case_id="no_answer_budget")
    )
    assert no_answer["root_cause"] == DiagnosticRootCause.NO_ANSWER_CORRECT.value

    security = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.NO_ANSWER,
                    debug_item=debug(item, selected=False),
                    reason=NoAnswerReason.SECURITY_POLICY,
                )
            )
        ).run_case(case_id="indirect_injection")
    )
    assert security["root_cause"] == DiagnosticRootCause.REVIEW_REQUIRED.value
    assert security["first_divergence"] == "answerability"


def test_custom_expected_answer_never_invents_stage_attribution() -> None:
    item = source()
    report = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.ANSWERED,
                    debug_item=debug(item),
                    sources=(item,),
                    answer="Deniz Aral.",
                )
            )
        ).run_custom(
            question="Who owns ORION?",
            expected_answer="Deniz Aral.",
        )
    )
    assert report["root_cause"] == DiagnosticRootCause.UNATTRIBUTED.value
    assert report["attribution_note"] == "UNATTRIBUTED — GOLD EVIDENCE REQUIRED"


def test_invalid_gold_locator_stops_before_pipeline_attribution() -> None:
    item = source()
    report = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.ANSWERED,
                    debug_item=debug(item),
                    sources=(item,),
                    answer="Deniz Aral.",
                )
            ),
            resolver=InvalidResolver(item),
        ).run_case(case_id="direct_owner")
    )
    assert report["root_cause"] == DiagnosticRootCause.DATASET_GOLD_INVALID.value
    assert report["first_divergence"] == "dataset"


def test_semantic_similarity_cannot_control_structured_verdict() -> None:
    claims = compare_claims(
        expected_claims=(),
        forbidden_claims=(),
        answer="semantically similar but factually different",
    )
    assert compare_decision(
        expected_decision="ANSWERED",
        actual_decision="NO_ANSWER",
        actual_reason="LOW_RELEVANCE",
        claims=claims,
    ).value == "FAIL"


def test_near_miss_expected_answer_passes_relation_and_negation_checks() -> None:
    case = next(
        item for item in load_gold_cases(MANIFEST) if item.case_id == "near_miss_legacy_code"
    )
    claims = compare_claims(
        expected_claims=case.expected_claims,
        forbidden_claims=case.forbidden_claims,
        answer=case.expected_answer,
    )
    assert claims["expected_claims_passed"] == claims["expected_claim_count"]
    assert claims["forbidden_claims_found"] == 0


def test_default_curated_case_scope_is_the_full_prepared_atlas_corpus() -> None:
    item = source()
    resolver = FakeResolver(item)
    asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.ANSWERED,
                    debug_item=debug(item),
                    sources=(item,),
                    answer="Deniz Aral.",
                )
            ),
            resolver=resolver,
        ).run_case(case_id="direct_owner")
    )
    assert resolver.document_key_calls
    assert set(resolver.document_key_calls[-1]) == {"ops", "tech", "notes", "untrusted"}


def test_wrong_year_case_requires_the_expected_no_answer_reason() -> None:
    item = source()
    report = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.NO_ANSWER,
                    debug_item=debug(item, selected=False),
                    sources=(),
                    reason=NoAnswerReason.INSUFFICIENT_COVERAGE,
                )
            )
        ).run_case(case_id="wrong_year_near_miss")
    )
    assert report["verdict"] == DiagnosticVerdict.PASS.value
    assert report["root_cause"] == DiagnosticRootCause.NO_ANSWER_CORRECT.value


def test_prompt_membership_is_not_inferred_from_final_sources() -> None:
    item = source()
    report = asyncio.run(
        service_for(
            FakeQueryService(
                result(
                    item=item,
                    decision=Decision.ANSWERED,
                    debug_item=debug(item),
                    sources=(item,),
                    answer="Deniz Aral.",
                ),
                emit_events=False,
            )
        ).run_case(case_id="direct_owner")
    )
    journey = cast(list[dict[str, object]], report["gold_evidence_journey"])
    assert journey[0]["prompt"] is False


def test_stage_journey_marks_non_applicable_retrieval_branches() -> None:
    item = source()
    for mode, expected in (
        (RetrievalMode.DENSE, {"dense": 1, "bm25": "N/A", "rrf": "N/A"}),
        (RetrievalMode.BM25, {"dense": "N/A", "bm25": 1, "rrf": "N/A"}),
    ):
        report = asyncio.run(
            service_for(
                FakeQueryService(
                    result(
                        item=item,
                        mode=mode.value,
                        decision=Decision.ANSWERED,
                        debug_item=debug(item),
                        sources=(item,),
                        answer="Deniz Aral.",
                    )
                )
            ).run_case(case_id="direct_owner", mode=mode)
        )
        journey = cast(list[dict[str, object]], report["gold_evidence_journey"])[0]
        for stage, value in expected.items():
            assert journey[stage] == value
        assert journey["reranker"] == "N/A"
