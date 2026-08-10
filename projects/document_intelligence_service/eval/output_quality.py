"""Deterministic phrase/claim coverage metrics for offline answer evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True, slots=True)
class PhraseCoverage:
    """Expected phrase coverage and forbidden phrase violations for one answer."""

    expected_phrases: tuple[str, ...]
    matched_phrases: tuple[str, ...]
    missing_phrases: tuple[str, ...]
    forbidden_phrases: tuple[str, ...]
    forbidden_found: tuple[str, ...]
    coverage_ratio: float
    passed: bool


def evaluate_phrase_coverage(
    *,
    answer: str,
    expected_phrases: Sequence[str],
    forbidden_phrases: Sequence[str] = (),
) -> PhraseCoverage:
    """Compare an answer with manually labeled phrase expectations.

    This is an offline evaluation signal, not a production truth oracle. A
    paraphrase can be correct without containing the exact phrase, so missing
    phrases require review rather than automatic answer rejection.
    """

    normalized_answer = normalize_text(answer)
    expected = tuple(expected_phrases)
    forbidden = tuple(forbidden_phrases)
    matched = tuple(
        phrase for phrase in expected if normalize_text(phrase) in normalized_answer
    )
    missing = tuple(phrase for phrase in expected if phrase not in matched)
    forbidden_found = tuple(
        phrase
        for phrase in forbidden
        if normalize_text(phrase) in normalized_answer
    )
    coverage = len(matched) / len(expected) if expected else 1.0
    return PhraseCoverage(
        expected_phrases=expected,
        matched_phrases=matched,
        missing_phrases=missing,
        forbidden_phrases=forbidden,
        forbidden_found=forbidden_found,
        coverage_ratio=coverage,
        passed=not missing and not forbidden_found,
    )


def normalize_text(text: str) -> str:
    """Normalize case, accents and punctuation for stable phrase matching."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^\w\s]", " ", without_marks).split())
