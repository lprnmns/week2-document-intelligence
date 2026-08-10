"""Offline expected-term coverage checks for retrieved evidence."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, cast

from .contracts import GoldenCase
from .output_quality import normalize_text


def build_evidence_coverage_report(
    *,
    cases: Sequence[GoldenCase],
    benchmark_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare labeled answerable terms with final retrieved evidence.

    This is a diagnostic grounding signal, not a semantic truth oracle. Exact
    phrase absence can be caused by a correct paraphrase; it therefore remains
    an evaluation report and is not silently turned into a runtime rejection.
    """

    observations = benchmark_report.get("run", {}).get("observations", [])
    if not isinstance(observations, list):
        raise ValueError("benchmark report does not contain observations")
    by_id = {
        item.get("case_id"): item
        for item in observations
        if isinstance(item, dict) and isinstance(item.get("case_id"), str)
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        if not case.expected_answerable or not case.expected_phrases:
            continue
        observation = by_id.get(case.case_id, {})
        candidates = observation.get("final_candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        evidence = "\n".join(
            str(item.get("text", ""))
            for item in candidates
            if isinstance(item, dict)
        )
        normalized_evidence = normalize_text(evidence)
        matched = tuple(
            phrase
            for phrase in case.expected_phrases
            if normalize_text(phrase) in normalized_evidence
        )
        missing = tuple(
            phrase for phrase in case.expected_phrases if phrase not in matched
        )
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "split": case.split,
                "expected_phrases": list(case.expected_phrases),
                "matched_phrases": list(matched),
                "missing_phrases": list(missing),
                "coverage_ratio": len(matched) / len(case.expected_phrases),
                "passed": not missing,
                "source_ids": [
                    item.get("source_id")
                    for item in candidates
                    if isinstance(item, dict)
                ],
            }
        )
    if not rows:
        raise ValueError("no answerable labeled evidence cases found")
    ratios = [float(row["coverage_ratio"]) for row in rows]
    return {
        "case_count": len(rows),
        "fully_covered_count": sum(bool(row["passed"]) for row in rows),
        "fully_covered_rate": sum(bool(row["passed"]) for row in rows) / len(rows),
        "mean_coverage_ratio": sum(ratios) / len(ratios),
        "rows": rows,
    }


def load_benchmark_report(path: Path) -> dict[str, Any]:
    """Load one JSON benchmark report."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark report must be a JSON object")
    return cast(dict[str, Any], raw)
