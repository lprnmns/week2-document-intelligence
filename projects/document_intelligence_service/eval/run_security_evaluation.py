"""Evaluate test-split prompt-injection and leakage gate regressions."""

from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .contracts import load_jsonl, validate_case_set
from .security import evaluate_security_gate

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPOSITORY_ROOT / "data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl"
DEFAULT_GATE_RESULTS = (
    REPOSITORY_ROOT
    / "projects/document_intelligence_service/eval/results/week2_stabilization_v1/"
    / "hybrid_answerability_gate.json"
)


def main() -> None:
    """Read frozen gate decisions and produce a security-only report."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gate-results", type=Path, default=DEFAULT_GATE_RESULTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = validate_case_set(load_jsonl(args.dataset), minimum_count=44)
    raw = json.loads(args.gate_results.read_text(encoding="utf-8"))
    observations = raw.get("run", {}).get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("gate results do not contain observation list")
    predictions: dict[str, bool] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("invalid gate observation")
        case_id = observation.get("case_id")
        decision = observation.get("decision")
        if not isinstance(case_id, str) or not isinstance(decision, str):
            raise ValueError("gate observation needs case_id and decision")
        predictions[case_id] = decision == "answered"

    metrics = evaluate_security_gate(cases, predictions)
    report = {
        "gate_git_sha": raw.get("git_sha"),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "evaluated_split": "test",
        "evaluated_categories": ["prompt_injection", "leakage_acl"],
        "llm_called": False,
        "metrics": asdict(metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
