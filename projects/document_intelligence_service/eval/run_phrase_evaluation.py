"""Evaluate one saved local answer against expected phrase labels."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
from typing import Any

from .output_quality import evaluate_phrase_coverage


DEFAULT_CASES = Path("data/evaluations/mentor_program_pdf_rag_cases_v2.json")


def main() -> None:
    """Read a saved query output and persist deterministic phrase coverage."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    response = _read_object(args.response)
    question = _required_string(response, "question")
    answer = response.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("saved response answer must be a string or null")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("phrase cases must be a JSON list")
    matching = tuple(
        case
        for case in cases
        if isinstance(case, dict) and case.get("question") == question
    )
    if len(matching) != 1:
        raise ValueError(f"expected exactly one phrase case for question: {question}")
    case = matching[0]
    expected = _string_list(case, "expected_phrases")
    forbidden = _string_list(case, "forbidden_phrases")
    coverage = evaluate_phrase_coverage(
        answer=answer or "",
        expected_phrases=expected,
        forbidden_phrases=forbidden,
    )
    report = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "response_file": str(args.response),
        "cases_file": str(args.cases),
        "case_id": case.get("id"),
        "question": question,
        "answer": answer,
        "decision": response.get("decision"),
        "phrase_coverage": {
            "expected_phrases": list(coverage.expected_phrases),
            "matched_phrases": list(coverage.matched_phrases),
            "missing_phrases": list(coverage.missing_phrases),
            "forbidden_phrases": list(coverage.forbidden_phrases),
            "forbidden_found": list(coverage.forbidden_found),
            "coverage_ratio": coverage.coverage_ratio,
            "passed": coverage.passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _read_object(path: Path) -> dict[str, object]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("saved response must be a JSON object")
    return raw


def _required_string(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"saved response needs a non-empty {key}")
    return value.strip()


def _string_list(raw: dict[str, object], key: str) -> list[str]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"case {key} must be a list of strings")
    return [item for item in value if item.strip()]


if __name__ == "__main__":
    main()
