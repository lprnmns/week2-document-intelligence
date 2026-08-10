"""Regression evidence for profile-scoped answerability calibration."""

import json
from pathlib import Path

from projects.document_intelligence_service.app.domain.answerability import (
    AnswerabilityPolicy,
    AnswerabilitySignals,
)
from projects.document_intelligence_service.app.domain.entities import (
    Decision,
    NoAnswerReason,
)
from projects.document_intelligence_service.eval.calibration import calibrate_threshold
from projects.document_intelligence_service.eval.contracts import load_jsonl


ROOT = Path(__file__).resolve().parents[4]
GENERIC_DATASET = ROOT / "data/evaluations/generic_document_answerability_v1.jsonl"
GENERIC_ARTIFACT = (
    ROOT
    / "projects/document_intelligence_service/eval/results/"
    / "generic_document_answerability_v1.json"
)
MENTOR_CALIBRATION = (
    ROOT
    / "projects/document_intelligence_service/eval/results/week2_stabilization_v1/"
    / "hybrid_threshold_calibration.json"
)


def test_mentor_calibration_artifact_remains_unchanged() -> None:
    artifact = json.loads(MENTOR_CALIBRATION.read_text(encoding="utf-8"))

    assert artifact["calibration_split"] == "validation"
    assert artifact["test_split_used"] is False
    assert artifact["calibration"]["rounded_threshold"] == 0.338
    assert artifact["calibration"]["metrics"][
        "no_answer_false_negative_count"
    ] == 0


def test_generic_calibration_is_validation_only_and_deterministic() -> None:
    artifact = json.loads(GENERIC_ARTIFACT.read_text(encoding="utf-8"))
    cases = load_jsonl(GENERIC_DATASET)
    validation = tuple(case for case in cases if case.split == "validation")
    observations = {
        row["case_id"]: row
        for row in artifact["validation_signal_observations"]
    }
    scores = {
        case.case_id: float(observations[case.case_id]["top_score"])
        for case in validation
    }

    first = calibrate_threshold(validation, scores, score_kind="dense")
    second = calibrate_threshold(validation, scores, score_kind="dense")

    assert artifact["selection_split"] == "validation"
    assert artifact["test_split_used_for_selection"] is False
    assert first == second
    assert first.rounded_threshold == artifact["frozen_runtime_policy"][
        "min_dense_score"
    ]
    assert artifact["calibration"]["dense_score"]["metrics"][
        "no_answer_false_negative_count"
    ] == 0


def test_generic_low_evidence_negative_is_rejected_by_frozen_policy() -> None:
    policy = AnswerabilityPolicy(
        min_dense_score=0.247,
        min_coverage=0.367,
        profile_name="generic_v1",
    )
    result = policy.decide(
        signals=AnswerabilitySignals(
            evidence_count=5,
            top_score=0.20984864,
            score_margin=0.060483,
            coverage_ratio=0.3333333333,
        ),
        score_kind="dense",
    )

    assert result.decision is Decision.NO_ANSWER
    assert result.reason is NoAnswerReason.LOW_RELEVANCE


def test_generic_near_miss_is_rejected_by_qualifier_coverage() -> None:
    artifact = json.loads(GENERIC_ARTIFACT.read_text(encoding="utf-8"))
    near_miss = next(
        row
        for row in artifact["test_observations"]
        if row["case_id"] == "generic_test_near_miss_year"
    )

    assert near_miss["decision"] == "no_answer"
    assert near_miss["reason"] == "INSUFFICIENT_COVERAGE"
    assert near_miss["missing_qualifiers"] == ["year:2024"]
    assert artifact["test_metrics"]["no_answer_false_negative_count"] == 0
