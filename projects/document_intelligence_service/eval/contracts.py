"""Versioned golden-query contracts used by offline evaluation runs."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Protocol, cast

CaseCategory = Literal[
    "direct_fact",
    "paraphrase",
    "exact_term",
    "near_miss",
    "no_answer",
    "multi_evidence",
    "prompt_injection",
    "leakage_acl",
]
CaseSplit = Literal["development", "validation", "test"]

CASE_CATEGORIES: frozenset[str] = frozenset(
    {
        "direct_fact",
        "paraphrase",
        "exact_term",
        "near_miss",
        "no_answer",
        "multi_evidence",
        "prompt_injection",
        "leakage_acl",
    }
)
CASE_SPLITS: frozenset[str] = frozenset({"development", "validation", "test"})


class EvidenceLike(Protocol):
    """Minimum retrieval shape required by the metric functions."""

    source_id: str
    document_id: str
    title: str


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One manually labeled query and its acceptable evidence targets.

    Stable section/document labels are preferred over generated child point IDs.
    A Qdrant version can change a child ID while the mentor PDF section remains
    the same evaluation target.
    """

    case_id: str
    question: str
    category: CaseCategory
    split: CaseSplit
    expected_answerable: bool
    relevant_document_ids: tuple[str, ...] = ()
    relevant_sections: tuple[str, ...] = ()
    relevant_source_ids: tuple[str, ...] = ()
    relevance_grades: tuple[tuple[str, int], ...] = ()
    expected_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    notes: str = ""
    language: str = "tr"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "GoldenCase":
        """Parse one JSON object and fail loudly on malformed labels."""

        case_id = _required_string(raw, "id")
        question = _required_string(raw, "question")
        category = _literal(
            _required_string(raw, "category"),
            CASE_CATEGORIES,
            "category",
        )
        split = _literal(
            _required_string(raw, "split"),
            CASE_SPLITS,
            "split",
        )
        expected_answerable = raw.get("expected_answerable")
        if not isinstance(expected_answerable, bool):
            raise ValueError("expected_answerable must be a boolean")
        language = raw.get("language", "tr")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string")

        relevant_document_ids = _string_tuple(raw, "relevant_document_ids")
        relevant_sections = _string_tuple(raw, "relevant_sections")
        relevant_source_ids = _string_tuple(raw, "relevant_source_ids")
        relevance_grades = _grades(raw.get("relevance_grades"))
        if relevance_grades and not relevant_sections:
            relevant_sections = tuple(key for key, _ in relevance_grades)

        has_target = bool(
            relevant_document_ids or relevant_sections or relevant_source_ids
        )
        if expected_answerable and not has_target:
            raise ValueError(
                "answerable case must define document, section or source evidence"
            )

        return cls(
            case_id=case_id,
            question=question,
            category=cast(CaseCategory, category),
            split=cast(CaseSplit, split),
            expected_answerable=expected_answerable,
            relevant_document_ids=relevant_document_ids,
            relevant_sections=relevant_sections,
            relevant_source_ids=relevant_source_ids,
            relevance_grades=relevance_grades,
            expected_phrases=_string_tuple(raw, "expected_phrases"),
            forbidden_phrases=_string_tuple(raw, "forbidden_phrases"),
            notes=_optional_string(raw, "notes"),
            language=language.strip().lower(),
        )

    def target_keys(self) -> frozenset[str]:
        """Return the stable identity set used as the recall denominator."""

        if self.relevant_source_ids:
            return frozenset(self.relevant_source_ids)
        if self.relevant_sections:
            return frozenset(self.relevant_sections)
        return frozenset(self.relevant_document_ids)

    def match_key(self, evidence: EvidenceLike) -> str | None:
        """Return the gold identity matched by evidence, if any."""

        if self.relevant_source_ids and evidence.source_id in self.relevant_source_ids:
            return evidence.source_id
        if self.relevant_sections and evidence.title in self.relevant_sections:
            return evidence.title
        if (
            self.relevant_document_ids
            and evidence.document_id in self.relevant_document_ids
        ):
            return evidence.document_id
        return None

    def grade_for(self, key: str) -> int:
        """Return graded relevance; unspecified targets default to binary 1."""

        for grade_key, grade in self.relevance_grades:
            if grade_key == key:
                return grade
        return 1 if key in self.target_keys() else 0


def load_jsonl(path: Path) -> tuple[GoldenCase, ...]:
    """Load and validate a newline-delimited golden dataset."""

    cases: list[GoldenCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"case at {path}:{line_number} must be an object")
        case = GoldenCase.from_mapping(parsed)
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    return tuple(cases)


def validate_case_set(
    cases: Iterable[GoldenCase],
    *,
    minimum_count: int = 40,
    expected_category_counts: Mapping[str, int] | None = None,
) -> tuple[GoldenCase, ...]:
    """Validate count and category balance before a benchmark run."""

    materialized = tuple(cases)
    if len(materialized) < minimum_count:
        raise ValueError(
            f"golden set has {len(materialized)} cases; minimum is {minimum_count}"
        )
    counts: dict[str, int] = {}
    for case in materialized:
        counts[case.category] = counts.get(case.category, 0) + 1
    for category, expected in (expected_category_counts or {}).items():
        if counts.get(category, 0) != expected:
            raise ValueError(
                f"category {category!r} has {counts.get(category, 0)} cases; "
                f"expected {expected}"
            )
    return materialized


def _required_string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _string_tuple(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    cleaned = tuple(item.strip() for item in value if item.strip())
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{key} must not contain duplicates")
    return cleaned


def _grades(value: object) -> tuple[tuple[str, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("relevance_grades must be an object")
    grades: list[tuple[str, int]] = []
    for key, grade in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("relevance grade keys must be non-empty strings")
        if not isinstance(grade, int) or isinstance(grade, bool) or grade < 1:
            raise ValueError("relevance grades must be positive integers")
        grades.append((key.strip(), grade))
    return tuple(grades)


def _literal(value: str, allowed: frozenset[str], field: str) -> str:
    if value not in allowed:
        raise ValueError(f"unsupported {field}: {value}")
    return value
