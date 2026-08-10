"""Tests for validation-only threshold selection."""

import pytest

from projects.document_intelligence_service.eval.calibration import calibrate_threshold
from projects.document_intelligence_service.eval.contracts import GoldenCase


def case(case_id: str, *, answerable: bool, score_split: str = "validation") -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        question=case_id,
        category="direct_fact" if answerable else "no_answer",
        split=score_split,  # type: ignore[arg-type]
        expected_answerable=answerable,
        relevant_sections=("rag",) if answerable else (),
    )


def test_calibration_uses_validation_gap_and_penalizes_unsafe_answers() -> None:
    cases = (
        case("unknown-low", answerable=False),
        case("unknown-high", answerable=False),
        case("known-low", answerable=True),
        case("known-high", answerable=True),
    )

    result = calibrate_threshold(
        cases,
        {
            "unknown-low": 0.10,
            "unknown-high": 0.40,
            "known-low": 0.50,
            "known-high": 0.80,
        },
    )

    assert 0.40 < result.threshold < 0.50
    assert result.metrics.no_answer_false_positive_count == 0
    assert result.metrics.no_answer_false_negative_count == 0
    assert result.objective == 0.0


def test_calibration_rejects_non_validation_cases() -> None:
    with pytest.raises(ValueError, match="validation cases only"):
        calibrate_threshold(
            (case("dev", answerable=True, score_split="development"),),
            {"dev": 0.8},
        )
