"""Calibrate answerability thresholds from a frozen validation split."""

from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .calibration import calibrate_threshold
from .contracts import load_jsonl, validate_case_set
from .security import SECURITY_CATEGORIES

DEFAULT_DATASET = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")
DEFAULT_GATE_RESULTS = Path(
    "projects/document_intelligence_service/eval/results/"
    "hybrid_answerability_gate.json"
)


def main() -> None:
    """Read gate traces, use validation only and write a calibration manifest."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gate-results", type=Path, default=DEFAULT_GATE_RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--false-negative-cost", type=float, default=3.0)
    args = parser.parse_args()

    cases = validate_case_set(load_jsonl(args.dataset), minimum_count=44)
    raw = json.loads(args.gate_results.read_text(encoding="utf-8"))
    observations = raw.get("run", {}).get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("gate results do not contain observation list")
    scores: dict[str, float] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("invalid gate observation")
        case_id = observation.get("case_id")
        top_score = observation.get("top_score")
        if not isinstance(case_id, str):
            raise ValueError("gate observation needs a string case_id")
        if top_score is None:
            # A pre-retrieval security block intentionally has no retrieval
            # score. Security policy is evaluated by its own frozen attack set,
            # not by the answerability threshold.
            continue
        if not isinstance(top_score, (int, float)):
            raise ValueError("gate observation top_score must be numeric or null")
        scores[case_id] = float(top_score)

    validation_cases = tuple(
        case
        for case in cases
        if case.split == "validation" and case.category not in SECURITY_CATEGORIES
    )
    calibration = calibrate_threshold(
        validation_cases,
        scores,
        false_negative_cost=args.false_negative_cost,
    )
    dataset_sha256 = hashlib.sha256(
        args.dataset.read_bytes()
    ).hexdigest()
    report = {
        "gate_git_sha": raw.get("git_sha"),
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "gate_results": str(args.gate_results),
        "calibration_split": "validation",
        "test_split_used": False,
        "calibration": asdict(calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
