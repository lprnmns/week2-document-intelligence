"""Deterministic expected-answer checking independent of pipeline attribution."""

from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata


class AnswerCheckMode(StrEnum):
    """Supported manual expected-answer comparison modes."""

    FACT_AWARE = "fact_aware"
    EXACT = "exact"
    SEMANTIC = "semantic"


class AnswerCheckVerdict(StrEnum):
    """User-facing answer correctness outcomes."""

    PASS = "PASS"
    FAIL_INCOMPLETE = "FAIL_INCOMPLETE"
    FAIL_CONTRADICTORY = "FAIL_CONTRADICTORY"
    FAIL_EXACT_MISMATCH = "FAIL_EXACT_MISMATCH"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True, slots=True)
class RequiredFact:
    """One explicit high-confidence fact extracted from the expected answer."""

    fact_type: str
    value: str
    matched: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.fact_type,
            "value": self.value,
            "matched": self.matched,
        }


@dataclass(frozen=True, slots=True)
class AnswerCheckResult:
    """Bounded answer comparison result."""

    mode: AnswerCheckMode
    verdict: AnswerCheckVerdict
    expected: str
    actual: str | None
    required_facts: tuple[RequiredFact, ...]
    semantic_similarity: float | None
    semantic_threshold: float | None
    semantic_used_for_verdict: bool
    summary: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "verdict": self.verdict.value,
            "expected": self.expected,
            "actual": self.actual,
            "required_facts": [fact.as_dict() for fact in self.required_facts],
            "semantic_similarity": self.semantic_similarity,
            "semantic_threshold": self.semantic_threshold,
            "semantic_used_for_verdict": self.semantic_used_for_verdict,
            "summary": self.summary,
        }


_MONTHS = (
    "ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık|"
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_TIME_RE = re.compile(r"(?<!\w)(?:[01]?\d|2[0-3]):[0-5]\d(?!\w)")
_DATE_WORD_RE = re.compile(
    rf"(?<!\w)\d{{1,2}}\s+(?:{_MONTHS})(?:\s+\d{{4}})?(?!\w)",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"(?<!\w)\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?!\w)")
_YEAR_RE = re.compile(r"(?<!\w)(?:19|20)\d{2}(?!\w)")
_PERCENT_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*%(?!\w)")
_CURRENCY_RE = re.compile(
    r"(?<!\w)(?:[$€₺£]\s*)?\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?\s*(?:₺|TL|EUR|USD|€|\$|£)(?!\w)",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"(?<!\w)(?=[A-Za-z0-9-]*\d)[A-Za-z]{1,8}-[A-Za-z0-9-]{1,24}(?!\w)")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
_QUOTE_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]")


def check_answer(
    *,
    expected: str,
    actual: str | None,
    mode: AnswerCheckMode | str = AnswerCheckMode.FACT_AWARE,
    semantic_similarity: float | None = None,
    semantic_threshold: float = 0.86,
) -> AnswerCheckResult:
    """Compare one already-produced answer without calling retrieval or LLM."""

    selected_mode = AnswerCheckMode(mode)
    expected_text = expected.strip()
    actual_text = actual.strip() if isinstance(actual, str) and actual.strip() else None
    if selected_mode is AnswerCheckMode.EXACT:
        verdict = (
            AnswerCheckVerdict.PASS
            if actual_text is not None and _normalize(expected_text) == _normalize(actual_text)
            else AnswerCheckVerdict.FAIL_EXACT_MISMATCH
        )
        return AnswerCheckResult(
            mode=selected_mode,
            verdict=verdict,
            expected=expected_text,
            actual=actual_text,
            required_facts=(),
            semantic_similarity=semantic_similarity,
            semantic_threshold=semantic_threshold,
            semantic_used_for_verdict=False,
            summary="Exact normalized answer match." if verdict is AnswerCheckVerdict.PASS else "The actual answer is not an exact normalized match.",
        )

    facts = _required_facts(expected_text, actual_text)
    missing = tuple(fact for fact in facts if not fact.matched)
    if missing:
        contradiction = any(_has_conflicting_fact(fact, actual_text or "") for fact in missing)
        verdict = (
            AnswerCheckVerdict.FAIL_CONTRADICTORY
            if contradiction
            else AnswerCheckVerdict.FAIL_INCOMPLETE
        )
        values = ", ".join(fact.value for fact in missing)
        return AnswerCheckResult(
            mode=selected_mode,
            verdict=verdict,
            expected=expected_text,
            actual=actual_text,
            required_facts=facts,
            semantic_similarity=semantic_similarity,
            semantic_threshold=semantic_threshold,
            semantic_used_for_verdict=False,
            summary=(
                f"Required fact(s) missing or contradicted: {values}."
            ),
        )

    if selected_mode is AnswerCheckMode.SEMANTIC:
        if semantic_similarity is None:
            verdict = AnswerCheckVerdict.REVIEW_REQUIRED
            summary = "Semantic comparison was requested but no dense similarity was computed."
            semantic_used = False
        else:
            semantic_used = True
            verdict = (
                AnswerCheckVerdict.PASS
                if semantic_similarity >= semantic_threshold
                else AnswerCheckVerdict.REVIEW_REQUIRED
            )
            summary = (
                "Semantic similarity cleared the configured informational threshold."
                if verdict is AnswerCheckVerdict.PASS
                else "Semantic similarity did not clear the configured threshold."
            )
    else:
        semantic_used = False
        normalized_expected = _normalize(expected_text)
        normalized_actual = _normalize(actual_text or "")
        verdict = (
            AnswerCheckVerdict.PASS
            if normalized_expected in normalized_actual
            or normalized_actual in normalized_expected
            or _token_overlap(normalized_expected, normalized_actual) >= 0.7
            else AnswerCheckVerdict.REVIEW_REQUIRED
        )
        summary = (
            "All explicit facts matched and the remaining phrasing is compatible."
            if verdict is AnswerCheckVerdict.PASS
            else "Explicit facts matched, but free-text phrasing needs review."
        )
    return AnswerCheckResult(
        mode=selected_mode,
        verdict=verdict,
        expected=expected_text,
        actual=actual_text,
        required_facts=facts,
        semantic_similarity=semantic_similarity,
        semantic_threshold=semantic_threshold,
        semantic_used_for_verdict=semantic_used,
        summary=summary,
    )


def _required_facts(expected: str, actual: str | None) -> tuple[RequiredFact, ...]:
    """Extract dates/times/numbers/codes without treating date parts twice."""

    facts: list[tuple[str, str, tuple[int, int]]] = []
    occupied: list[tuple[int, int]] = []
    for pattern, fact_type in (
        (_TIME_RE, "time"),
        (_DATE_WORD_RE, "date"),
        (_NUMERIC_DATE_RE, "date"),
        (_PERCENT_RE, "percentage"),
        (_CURRENCY_RE, "currency"),
        (_CODE_RE, "code"),
        (_YEAR_RE, "year"),
    ):
        for match in pattern.finditer(expected):
            span = match.span()
            if any(_overlap(span, other) for other in occupied):
                continue
            value = match.group(0).strip()
            facts.append((fact_type, value, span))
            occupied.append(span)
    for match in _NUMBER_RE.finditer(expected):
        span = match.span()
        if any(_overlap(span, other) for other in occupied):
            continue
        facts.append(("number", match.group(0), span))
        occupied.append(span)
    for match in _QUOTE_RE.finditer(expected):
        value = match.group(1).strip()
        if value and not any(_normalize(value) == _normalize(item[1]) for item in facts):
            facts.append(("quoted", value, match.span()))
    facts.sort(key=lambda item: item[2][0])
    normalized_actual = _normalize(actual or "")
    return tuple(
        RequiredFact(fact_type, value, _fact_matches(fact_type, value, normalized_actual))
        for fact_type, value, _ in facts
    )


def _fact_matches(fact_type: str, value: str, normalized_actual: str) -> bool:
    normalized_value = _normalize(value)
    if fact_type in {"number", "year"}:
        return _number_key(value) in {_number_key(item) for item in _NUMBER_RE.findall(normalized_actual)}
    if fact_type == "percentage":
        return _number_key(value.rstrip("% ")) in {
            _number_key(item.rstrip("% ")) for item in _PERCENT_RE.findall(normalized_actual)
        }
    if fact_type == "time":
        return normalized_value in normalized_actual.replace(".", ":")
    return normalized_value in normalized_actual


def _has_conflicting_fact(fact: RequiredFact, actual: str) -> bool:
    normalized = _normalize(actual)
    if fact.fact_type == "time":
        return bool(_TIME_RE.search(normalized))
    if fact.fact_type == "date":
        return bool(_DATE_WORD_RE.search(normalized) or _NUMERIC_DATE_RE.search(normalized))
    if fact.fact_type == "percentage":
        return bool(_PERCENT_RE.search(normalized))
    return False


def _number_key(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("’", "'")
    return " ".join(value.split())


def _token_overlap(expected: str, actual: str) -> float:
    expected_tokens = set(re.findall(r"[\wÀ-ÿ]+", expected, flags=re.UNICODE))
    actual_tokens = set(re.findall(r"[\wÀ-ÿ]+", actual, flags=re.UNICODE))
    if not expected_tokens:
        return 1.0 if not actual_tokens else 0.0
    return len(expected_tokens & actual_tokens) / len(expected_tokens)


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
