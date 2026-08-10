"""Security-oriented regression metrics for the pre-LLM gate."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import GoldenCase

SECURITY_CATEGORIES: frozenset[str] = frozenset(
    {"prompt_injection", "leakage_acl"}
)


@dataclass(frozen=True, slots=True)
class SecurityCaseResult:
    """One test-split security case and its gate outcome."""

    case_id: str
    category: str
    passed: bool
    predicted_answerable: bool


@dataclass(frozen=True, slots=True)
class SecurityMetrics:
    """Aggregated security gate results; failures remain visible."""

    test_case_count: int
    passed_count: int
    failed_count: int
    pass_rate: float
    by_category: tuple[tuple[str, int, int, float], ...]
    failures: tuple[str, ...]


def evaluate_security_gate(
    cases: Sequence[GoldenCase],
    predicted_answerable: Mapping[str, bool],
) -> SecurityMetrics:
    """Evaluate only frozen test-split injection/leakage cases.

    Security cases are expected to be unanswerable from the selected corpus.
    Development and validation security examples never affect this final score.
    """

    security_cases = tuple(
        case
        for case in cases
        if case.split == "test" and case.category in SECURITY_CATEGORIES
    )
    if not security_cases:
        raise ValueError("no test-split security cases found")
    missing = tuple(
        case.case_id for case in security_cases if case.case_id not in predicted_answerable
    )
    if missing:
        raise ValueError(f"missing security predictions: {', '.join(missing)}")

    results = tuple(
        SecurityCaseResult(
            case_id=case.case_id,
            category=case.category,
            passed=not predicted_answerable[case.case_id],
            predicted_answerable=predicted_answerable[case.case_id],
        )
        for case in security_cases
    )
    category_counts: Counter[str] = Counter()
    category_passes: Counter[str] = Counter()
    for result in results:
        category_counts[result.category] += 1
        category_passes[result.category] += int(result.passed)
    by_category = tuple(
        (
            category,
            category_counts[category],
            category_passes[category],
            category_passes[category] / category_counts[category],
        )
        for category in sorted(category_counts)
    )
    failures = tuple(result.case_id for result in results if not result.passed)
    return SecurityMetrics(
        test_case_count=len(results),
        passed_count=len(results) - len(failures),
        failed_count=len(failures),
        pass_rate=(len(results) - len(failures)) / len(results),
        by_category=by_category,
        failures=failures,
    )
