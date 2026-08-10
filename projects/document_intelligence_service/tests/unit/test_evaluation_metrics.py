"""Tests for backend-independent evaluation calculations."""

import pytest

from projects.document_intelligence_service.eval.contracts import GoldenCase
from projects.document_intelligence_service.eval.metrics import (
    evaluate_no_answer,
    evaluate_retrieval,
    latency_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class Evidence:
    """Small evidence fixture matching the evaluation protocol."""

    def __init__(self, source_id: str, title: str, rank: int = 1) -> None:
        self.source_id = source_id
        self.document_id = "mentor-week1"
        self.title = title
        self.rank = rank


def make_case(*, answerable: bool = True) -> GoldenCase:
    return GoldenCase(
        case_id="case-1",
        question="RAG nedir?",
        category="direct_fact",
        split="development",
        expected_answerable=answerable,
        relevant_sections=("rag",) if answerable else (),
    )


def test_section_metrics_deduplicate_multiple_child_chunks() -> None:
    case = make_case()
    results = (
        Evidence("chunk-a", "embedding"),
        Evidence("chunk-b", "rag"),
        Evidence("chunk-c", "rag"),
    )

    assert recall_at_k(case, results, 1) == 0.0
    assert recall_at_k(case, results, 2) == 1.0
    assert mrr_at_k(case, results, 5) == 0.5
    assert ndcg_at_k(case, results, 5) == 1.0


def test_multi_evidence_recall_and_ndcg_use_all_gold_sections() -> None:
    case = GoldenCase(
        case_id="multi",
        question="Sistem nasıl çalışır?",
        category="multi_evidence",
        split="validation",
        expected_answerable=True,
        relevant_sections=("rag", "local_model"),
        relevance_grades=(("rag", 2), ("local_model", 1)),
    )
    results = (
        Evidence("chunk-a", "local_model"),
        Evidence("chunk-b", "rag"),
    )

    assert recall_at_k(case, results, 1) == 0.5
    assert recall_at_k(case, results, 2) == 1.0
    assert mrr_at_k(case, results, 5) == 1.0
    assert 0.0 < ndcg_at_k(case, results, 2) < 1.0


def test_retrieval_aggregate_keeps_candidate_and_final_windows_separate() -> None:
    case = make_case()
    final = {case.case_id: (Evidence("chunk-x", "embedding"),)}
    candidates = {
        case.case_id: (
            Evidence("chunk-x", "embedding"),
            Evidence("chunk-y", "rag"),
        )
    }

    metrics = evaluate_retrieval((case,), final, candidate_results=candidates)

    assert metrics.recall_at_1 == 0.0
    assert metrics.candidate_recall_at_20 == 1.0


def test_no_answer_metrics_make_error_direction_explicit() -> None:
    answerable = make_case()
    no_answer = GoldenCase(
        case_id="case-2",
        question="Maaş ne kadar?",
        category="no_answer",
        split="test",
        expected_answerable=False,
    )

    metrics = evaluate_no_answer(
        (answerable, no_answer),
        {"case-1": False, "case-2": True},
    )

    assert metrics.no_answer_false_positive_count == 1
    assert metrics.no_answer_false_negative_count == 1
    assert metrics.no_answer_false_positive_rate == 1.0
    assert metrics.no_answer_false_negative_rate == 1.0


def test_latency_metrics_use_linear_percentiles() -> None:
    metrics = latency_metrics((10.0, 20.0, 30.0, 40.0, 100.0))

    assert metrics.p50_ms == 30.0
    assert metrics.p95_ms == pytest.approx(88.0)
    assert metrics.min_ms == 10.0
    assert metrics.max_ms == 100.0


def test_answerable_case_requires_evidence_target() -> None:
    with pytest.raises(ValueError, match="must define document"):
        GoldenCase.from_mapping(
            {
                "id": "invalid",
                "question": "Soru",
                "category": "direct_fact",
                "split": "development",
                "expected_answerable": True,
            }
        )
