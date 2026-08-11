"""Deterministic expected-answer regression tests."""

import json
from pathlib import Path

from projects.document_intelligence_service.app.domain.answer_check import (
    AnswerCheckVerdict,
    check_answer,
)


def test_deadline_time_is_required_when_expected() -> None:
    result = check_answer(
        expected="10 August 2026 23:59",
        actual="10 August 2026",
    )
    assert result.verdict is AnswerCheckVerdict.FAIL_INCOMPLETE
    assert [fact.value for fact in result.required_facts if not fact.matched] == ["23:59"]


def test_committed_synthetic_deadline_fixture_is_used_for_the_regression() -> None:
    root = Path(__file__).parents[4]
    case = json.loads(
        (
            root
            / "data/evaluations/diagnostic_fixtures/synthetic_deadline_expected_case.json"
        ).read_text(encoding="utf-8")
    )
    result = check_answer(
        expected=case["expected_answer"],
        actual="10 August 2026",
        mode=case["matching_mode"],
    )
    assert result.verdict is AnswerCheckVerdict.FAIL_INCOMPLETE


def test_semantic_similarity_cannot_override_missing_explicit_fact() -> None:
    result = check_answer(
        expected="10 August 2026 23:59",
        actual="10 August 2026",
        mode="semantic",
        semantic_similarity=0.99,
    )
    assert result.verdict is AnswerCheckVerdict.FAIL_INCOMPLETE
    assert result.semantic_used_for_verdict is False
