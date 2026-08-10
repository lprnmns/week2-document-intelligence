"""Tests for frozen test-split security gate evaluation."""

from projects.document_intelligence_service.eval.contracts import GoldenCase
from projects.document_intelligence_service.eval.security import evaluate_security_gate


def security_case(case_id: str, category: str) -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        question=case_id,
        category=category,  # type: ignore[arg-type]
        split="test",
        expected_answerable=False,
    )


def test_security_metrics_keep_injection_failure_visible() -> None:
    cases = (
        security_case("injection", "prompt_injection"),
        security_case("leakage", "leakage_acl"),
    )

    metrics = evaluate_security_gate(
        cases,
        {"injection": True, "leakage": False},
    )

    assert metrics.test_case_count == 2
    assert metrics.passed_count == 1
    assert metrics.failed_count == 1
    assert metrics.pass_rate == 0.5
    assert metrics.failures == ("injection",)
