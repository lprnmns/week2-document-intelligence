"""Framework-independent validation of generated answers against evidence."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import re

from .retrieval import RetrievedChunk


class EvidenceWarningCode(StrEnum):
    """Stable warning codes exposed by output/evidence validation."""

    UNSUPPORTED_NUMBER = "UNSUPPORTED_NUMBER"


@dataclass(frozen=True, slots=True)
class EvidenceWarning:
    """One structured concern found in a generated answer."""

    code: EvidenceWarningCode
    message: str
    values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceValidationResult:
    """Validation output kept separate from the generated answer itself."""

    warnings: tuple[EvidenceWarning, ...]


@dataclass(frozen=True, slots=True)
class _NumericMention:
    """Internal normalized representation of one numeric answer mention."""

    raw: str
    value: Decimal


_NUMBER_RE = re.compile(
    r"(?<![\w])"
    r"(?P<number>[+-]?(?:\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?))"
    r"(?:\s*(?:%|ms|s|sn|gb|mb|kb|tb|b|gün|gun|hafta|ay|yıl|yil|"
    r"saat|dakika|token|tokens|point|points))?"
    r"(?![\w])",
    flags=re.IGNORECASE | re.UNICODE,
)


def validate_answer_against_evidence(
    *,
    answer: str,
    evidence: Sequence[RetrievedChunk],
) -> EvidenceValidationResult:
    """Warn when answer numbers cannot be found in supplied evidence.

    This is intentionally a narrow first guardrail. It does not claim to
    prove every natural-language statement, and it does not mutate or reject
    the answer. Numeric claims are selected because they are easy to inspect,
    often operationally important, and a common hallucination failure mode.
    """

    answer_mentions = _extract_numeric_mentions(answer)
    if not answer_mentions:
        return EvidenceValidationResult(warnings=())

    evidence_text = "\n".join(item.context_text for item in evidence)
    evidence_values = {
        mention.value for mention in _extract_numeric_mentions(evidence_text)
    }
    unsupported: list[str] = []
    seen_values: set[Decimal] = set()
    for mention in answer_mentions:
        if mention.value in evidence_values or mention.value in seen_values:
            continue
        seen_values.add(mention.value)
        unsupported.append(mention.raw)

    if not unsupported:
        return EvidenceValidationResult(warnings=())

    warning = EvidenceWarning(
        code=EvidenceWarningCode.UNSUPPORTED_NUMBER,
        message=(
            "Cevapta geçen bazı sayılar getirilen kanıtta bulunamadı; "
            "cevap insan incelemesine gönderilmelidir."
        ),
        values=tuple(unsupported),
    )
    return EvidenceValidationResult(warnings=(warning,))


def _extract_numeric_mentions(text: str) -> tuple[_NumericMention, ...]:
    """Extract decimal/integer mentions while ignoring list numbering."""

    mentions: list[_NumericMention] = []
    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        if _is_non_claim_number(text, start, end):
            continue
        raw = match.group("number")
        value = _parse_number(raw)
        if value is not None:
            mentions.append(_NumericMention(raw=raw, value=value))
    return tuple(mentions)


def _is_non_claim_number(text: str, start: int, end: int) -> bool:
    """Exclude common formatting identifiers, not factual quantities."""

    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    following = text[end : end + 1]
    if not prefix.strip() and following in {".", ")"}:
        return True
    if text[start - 1 : start] == "[" and text[end : end + 1] == "]":
        return True
    # Avoid interpreting the ``4b`` in a model identifier such as
    # ``gemma3:4b`` as a factual numeric claim.
    if text[start - 1 : start] == ":" and start > 1 and text[start - 2].isalnum():
        return True
    return False


def _parse_number(raw: str) -> Decimal | None:
    """Normalize the common Turkish decimal comma before comparison."""

    normalized = raw.replace(",", ".")
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None
