"""Deterministic retrieval, answerability and latency metrics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2

from .contracts import EvidenceLike, GoldenCase


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Macro-averaged ranking metrics over answerable golden cases."""

    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    candidate_recall_at_20: float
    mrr_at_5: float
    mrr_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float


@dataclass(frozen=True, slots=True)
class NoAnswerMetrics:
    """Confusion counts for the answerable/no-answer decision."""

    case_count: int
    expected_answerable: int
    expected_no_answer: int
    predicted_answerable: int
    predicted_no_answer: int
    no_answer_false_positive_count: int
    no_answer_false_negative_count: int
    no_answer_false_positive_rate: float
    no_answer_false_negative_rate: float


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    """Percentiles for one measured stage or end-to-end query."""

    sample_count: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def evaluate_retrieval(
    cases: Sequence[GoldenCase],
    final_results: Mapping[str, Sequence[EvidenceLike]],
    *,
    candidate_results: Mapping[str, Sequence[EvidenceLike]] | None = None,
) -> RetrievalMetrics:
    """Compute ranking metrics without assuming a specific retriever backend."""

    answerable_cases = tuple(case for case in cases if case.expected_answerable)
    if not answerable_cases:
        raise ValueError("retrieval evaluation needs at least one answerable case")

    def average(metric: str, k: int) -> float:
        values: list[float] = []
        for case in answerable_cases:
            results = final_results.get(case.case_id, ())
            if metric == "recall":
                values.append(recall_at_k(case, results, k))
            elif metric == "mrr":
                values.append(mrr_at_k(case, results, k))
            else:
                values.append(ndcg_at_k(case, results, k))
        return sum(values) / len(values)

    candidate_source = candidate_results or final_results
    candidate_recall = sum(
        recall_at_k(case, candidate_source.get(case.case_id, ()), 20)
        for case in answerable_cases
    ) / len(answerable_cases)
    return RetrievalMetrics(
        query_count=len(answerable_cases),
        recall_at_1=average("recall", 1),
        recall_at_3=average("recall", 3),
        recall_at_5=average("recall", 5),
        candidate_recall_at_20=candidate_recall,
        mrr_at_5=average("mrr", 5),
        mrr_at_10=average("mrr", 10),
        ndcg_at_5=average("ndcg", 5),
        ndcg_at_10=average("ndcg", 10),
    )


def recall_at_k(
    case: GoldenCase,
    results: Sequence[EvidenceLike],
    k: int,
) -> float:
    """Return unique gold-target recall in the first ``k`` results."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    targets = case.target_keys()
    if not targets:
        return 0.0
    found = {
        key
        for item in results[:k]
        if (key := case.match_key(item)) is not None
    }
    return len(found & targets) / len(targets)


def mrr_at_k(
    case: GoldenCase,
    results: Sequence[EvidenceLike],
    k: int,
) -> float:
    """Return reciprocal rank of the first unique relevant target."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    seen: set[str] = set()
    for rank, item in enumerate(results[:k], start=1):
        key = case.match_key(item)
        if key is not None and key not in seen:
            return 1.0 / rank
        if key is not None:
            seen.add(key)
    return 0.0


def ndcg_at_k(
    case: GoldenCase,
    results: Sequence[EvidenceLike],
    k: int,
) -> float:
    """Return nDCG with duplicate child chunks counted once per target."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    observed: list[int] = []
    seen: set[str] = set()
    for item in results[:k]:
        key = case.match_key(item)
        if key is None or key in seen:
            continue
        seen.add(key)
        observed.append(case.grade_for(key))
    ideal = sorted((case.grade_for(key) for key in case.target_keys()), reverse=True)
    ideal_dcg = _dcg(ideal[:k])
    if ideal_dcg == 0:
        return 0.0
    return _dcg(observed) / ideal_dcg


def evaluate_no_answer(
    cases: Sequence[GoldenCase],
    predicted_answerable: Mapping[str, bool],
) -> NoAnswerMetrics:
    """Measure both kinds of no-answer mistake with explicit semantics.

    ``no_answer_false_positive`` means the system rejected an answerable case.
    ``no_answer_false_negative`` means the system answered an unanswerable case.
    The names are intentionally documented because teams often reverse these
    labels when treating ``no_answer`` as the positive class.
    """

    missing = tuple(case.case_id for case in cases if case.case_id not in predicted_answerable)
    if missing:
        raise ValueError(f"missing answerability predictions: {', '.join(missing)}")
    expected_answerable = sum(case.expected_answerable for case in cases)
    expected_no_answer = len(cases) - expected_answerable
    predicted_answered_count = sum(predicted_answerable[case.case_id] for case in cases)
    predicted_no_answer_count = len(cases) - predicted_answered_count
    false_positive = sum(
        case.expected_answerable and not predicted_answerable[case.case_id]
        for case in cases
    )
    false_negative = sum(
        not case.expected_answerable and predicted_answerable[case.case_id]
        for case in cases
    )
    return NoAnswerMetrics(
        case_count=len(cases),
        expected_answerable=expected_answerable,
        expected_no_answer=expected_no_answer,
        predicted_answerable=predicted_answered_count,
        predicted_no_answer=predicted_no_answer_count,
        no_answer_false_positive_count=false_positive,
        no_answer_false_negative_count=false_negative,
        no_answer_false_positive_rate=(
            false_positive / expected_answerable if expected_answerable else 0.0
        ),
        no_answer_false_negative_rate=(
            false_negative / expected_no_answer if expected_no_answer else 0.0
        ),
    )


def latency_metrics(values_ms: Sequence[float]) -> LatencyMetrics:
    """Return reproducible linear-interpolated p50 and p95 values."""

    if not values_ms:
        raise ValueError("latency sample cannot be empty")
    if any(value < 0 for value in values_ms):
        raise ValueError("latency values cannot be negative")
    ordered = sorted(values_ms)
    return LatencyMetrics(
        sample_count=len(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )


def _dcg(grades: Sequence[int]) -> float:
    return sum(grade / log2(rank + 2) for rank, grade in enumerate(grades))


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight
