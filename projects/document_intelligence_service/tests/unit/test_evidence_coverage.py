"""Tests for offline evidence phrase coverage diagnostics."""

from projects.document_intelligence_service.eval.contracts import GoldenCase
from projects.document_intelligence_service.eval.evidence_coverage import (
    build_evidence_coverage_report,
)


def test_evidence_coverage_matches_labeled_terms_without_calling_llm() -> None:
    case = GoldenCase.from_mapping(
        {
            "id": "q1",
            "question": "Soru",
            "category": "direct_fact",
            "split": "test",
            "expected_answerable": True,
            "relevant_sections": ["guide"],
            "expected_phrases": ["Qdrant", "kaynak"],
        }
    )
    report = build_evidence_coverage_report(
        cases=(case,),
        benchmark_report={
            "run": {
                "observations": [
                    {
                        "case_id": "q1",
                        "final_candidates": [
                            {"source_id": "s1", "text": "Qdrant kaynak saklar."}
                        ],
                    }
                ]
            }
        },
    )

    assert report["fully_covered_count"] == 1
    assert report["mean_coverage_ratio"] == 1.0
