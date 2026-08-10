"""Validation-only calibration for answerability score thresholds."""

from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import GoldenCase
from .metrics import NoAnswerMetrics, evaluate_no_answer


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    """Selected threshold and validation-only error accounting."""

    score_kind: str
    threshold: float
    rounded_threshold: float
    false_negative_cost: float
    validation_case_count: int
    candidate_threshold_count: int
    objective: float
    metrics: NoAnswerMetrics


def calibrate_threshold(
    cases: tuple[GoldenCase, ...],
    scores: Mapping[str, float],
    *,
    score_kind: str = "dense",
    false_negative_cost: float = 3.0,
) -> ThresholdCalibration:
    """Choose a threshold using validation cases only.

    A false negative means answering an unanswerable question, so its cost is
    explicit and higher than a false positive by default. Candidate thresholds
    are midpoints between observed scores; this avoids accidentally classifying
    a point on the boundary due to ``>=`` semantics.
    """

    if not cases:
        raise ValueError("calibration needs at least one validation case")
    if any(case.split != "validation" for case in cases):
        raise ValueError("threshold calibration accepts validation cases only")
    if false_negative_cost <= 0:
        raise ValueError("false_negative_cost must be greater than zero")
    missing = tuple(case.case_id for case in cases if case.case_id not in scores)
    if missing:
        raise ValueError(f"missing validation scores: {', '.join(missing)}")
    if any(score < 0 or score > 1 for score in scores.values()):
        raise ValueError("calibration scores must be between zero and one")

    unique_scores = sorted({scores[case.case_id] for case in cases})
    thresholds = {0.0, 1.0}
    thresholds.update(
        (left + right) / 2 for left, right in zip(unique_scores, unique_scores[1:])
    )

    best: tuple[tuple[float, int, int, float], float, NoAnswerMetrics] | None = None
    for threshold in sorted(thresholds):
        predictions = {
            case.case_id: scores[case.case_id] >= threshold for case in cases
        }
        metrics = evaluate_no_answer(cases, predictions)
        objective = (
            metrics.no_answer_false_positive_count
            + false_negative_cost * metrics.no_answer_false_negative_count
        )
        # Prefer lower objective, then fewer unsafe answers, then fewer false
        # rejections. The final tie-break is the higher threshold.
        ordering = (
            objective,
            metrics.no_answer_false_negative_count,
            metrics.no_answer_false_positive_count,
            -threshold,
        )
        if best is None or ordering < best[0]:
            best = (ordering, threshold, metrics)

    if best is None:  # pragma: no cover - thresholds always include 0 and 1
        raise RuntimeError("no calibration candidate was generated")
    _, threshold, metrics = best
    return ThresholdCalibration(
        score_kind=score_kind,
        threshold=threshold,
        rounded_threshold=round(threshold, 3),
        false_negative_cost=false_negative_cost,
        validation_case_count=len(cases),
        candidate_threshold_count=len(thresholds),
        objective=metrics.no_answer_false_positive_count
        + false_negative_cost * metrics.no_answer_false_negative_count,
        metrics=metrics,
    )
