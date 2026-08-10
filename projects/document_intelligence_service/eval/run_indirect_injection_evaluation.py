"""Run a deterministic evidence-boundary smoke without calling an LLM."""

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from ..app.domain.evidence_safety import EvidenceSafetyPolicy
from ..app.domain.retrieval import RetrievedChunk

DEFAULT_CASES = Path("data/evaluations/indirect_injection_cases_v1.json")


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One evidence safety fixture result."""

    case_id: str
    kind: str
    expected_blocked: bool
    blocked: bool
    matched_rules: tuple[str, ...]
    passed: bool


def main() -> None:
    """Evaluate high-confidence indirect injection containment rules."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_cases: Any = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("indirect injection cases must be a JSON list")

    policy = EvidenceSafetyPolicy()
    results: list[CaseResult] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("each indirect injection case must be an object")
        case_id = _required_string(raw, "id")
        kind = _required_string(raw, "kind")
        text = _required_string(raw, "text")
        expected_blocked = raw.get("expected_blocked")
        if not isinstance(expected_blocked, bool):
            raise ValueError("expected_blocked must be boolean")
        evidence = RetrievedChunk(
            source_id=case_id,
            document_id="attack-fixture",
            version_id="v1",
            parent_id="attack-fixture",
            title=kind,
            text=text,
            page_start=1,
            page_end=1,
            score=1.0,
            rank=1,
        )
        safety = policy.filter((evidence,))
        blocked = bool(safety.blocked_source_ids)
        results.append(
            CaseResult(
                case_id=case_id,
                kind=kind,
                expected_blocked=expected_blocked,
                blocked=blocked,
                matched_rules=safety.matched_rules,
                passed=blocked == expected_blocked,
            )
        )

    report = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cases": str(args.cases),
        "llm_called": False,
        "control": "EvidenceSafetyPolicy before evidence enters the LLM prompt",
        "case_count": len(results),
        "passed_count": sum(item.passed for item in results),
        "blocked_attack_count": sum(
            item.blocked for item in results if item.kind == "indirect_injection"
        ),
        "attack_count": sum(
            item.kind == "indirect_injection" for item in results
        ),
        "benign_allowed_count": sum(
            not item.blocked for item in results if item.kind == "benign_evidence"
        ),
        "results": [asdict(item) for item in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _required_string(raw: dict[str, object], key: str) -> str:
    """Read one required non-empty string from a case object."""

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {key} must be a non-empty string")
    return value.strip()


if __name__ == "__main__":
    main()
