"""Framework-independent answerability policy and decision signals."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re

from .entities import Decision, NoAnswerReason


_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_PERCENT_WORD_RE = re.compile(
    r"\by[uü]zde\s*([0-9]+(?:[.,][0-9]+)?)\b",
    flags=re.IGNORECASE,
)
_PERCENT_PREFIX_RE = re.compile(r"%\s*([0-9]+(?:[.,][0-9]+)?)\b")
_PERCENT_SUFFIX_RE = re.compile(r"\b([0-9]+(?:[.,][0-9]+)?)\s*%")
_QUOTED_RE = re.compile(
    r'"([^"\n]{2,80})"|“([^”\n]{2,80})”|«([^»\n]{2,80})»'
)
_QUALIFIER_CONTEXT_RADIUS = 48
_GENERIC_QUESTION_TERMS = {
    "bir",
    "bilgi",
    "bilgisi",
    "edinilir",
    "ediliyor",
    "hangi",
    "için",
    "kaç",
    "kapanış",
    "nedir",
    "olan",
    "program",
    "programı",
    "programının",
    "sıra",
    "sırası",
    "veriliyor",
}
_ATTRIBUTE_TERMS = {
    "başarı",
    "burs",
    "fiyat",
    "indirim",
    "kapanış",
    "kontenjan",
    "kontenjanı",
    "puan",
    "puanı",
    "sıra",
    "sırası",
    "ücret",
    "ücreti",
    "fiyatı",
}


@dataclass(frozen=True, slots=True)
class QualifierCoverage:
    """Coverage of high-confidence explicit question qualifiers.

    This deliberately covers only qualifiers whose presence can be checked
    deterministically without pretending to understand every question.  It
    is complementary to lexical retrieval overlap: a document can be about
    the right topic while still missing the requested year or exact term.
    """

    required: tuple[str, ...]
    matched: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def satisfied(self) -> bool:
        """Return whether every extracted explicit qualifier is supported."""

        return not self.missing


def qualifier_coverage(
    question: str,
    evidence_texts: Sequence[str],
) -> QualifierCoverage:
    """Check years, percentages and explicitly quoted terms in evidence.

    Years and percentages are normalized so equivalent spellings such as
    ``yüzde 50``, ``%50`` and ``50%`` compare equal.  Other numbers are not
    automatically treated as requirements; doing so would reject ordinary
    paraphrases and unrelated numeric context.  Quoted terms are included
    because quotation is a high-confidence request for that exact phrase.
    """

    required = _extract_qualifiers(question)
    if not required:
        return QualifierCoverage(required=(), matched=(), missing=())

    evidence = " ".join(evidence_texts)
    available = _extract_qualifier_occurrences(evidence)
    anchors = _question_anchor_tokens(question)
    attributes = _question_attribute_tokens(question)
    normalized_evidence = _normalize_phrase(evidence)
    matched: list[str] = []
    missing: list[str] = []
    for qualifier in required:
        if qualifier.startswith("quoted:"):
            value = qualifier.removeprefix("quoted:")
            is_present = value in normalized_evidence
        else:
            spans = available.get(qualifier, ())
            is_present = any(
                _qualifier_has_context(
                    qualifier,
                    evidence,
                    start=start,
                    end=end,
                    anchors=anchors,
                    attributes=attributes,
                )
                for start, end in spans
            )
        (matched if is_present else missing).append(qualifier)
    return QualifierCoverage(
        required=required,
        matched=tuple(matched),
        missing=tuple(missing),
    )


def _extract_qualifiers(text: str) -> tuple[str, ...]:
    """Extract deterministic qualifier keys in stable order."""

    values: set[str] = set()
    for match in _YEAR_RE.finditer(text):
        values.add(f"year:{match.group(0)}")
    for pattern in (
        _PERCENT_WORD_RE,
        _PERCENT_PREFIX_RE,
        _PERCENT_SUFFIX_RE,
    ):
        for match in pattern.finditer(text):
            values.add(f"percent:{_normalize_number(match.group(1))}")
    for match in _QUOTED_RE.finditer(text):
        quoted = next((group for group in match.groups() if group), "")
        normalized = _normalize_phrase(quoted)
        if normalized:
            values.add(f"quoted:{normalized}")
    return tuple(sorted(values))


def _extract_qualifier_occurrences(
    text: str,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return normalized qualifier keys and their source spans."""

    occurrences: dict[str, list[tuple[int, int]]] = {}

    def add(key: str, start: int, end: int) -> None:
        occurrences.setdefault(key, []).append((start, end))

    for match in _YEAR_RE.finditer(text):
        add(f"year:{match.group(0)}", match.start(), match.end())
    for pattern in (
        _PERCENT_WORD_RE,
        _PERCENT_PREFIX_RE,
        _PERCENT_SUFFIX_RE,
    ):
        for match in pattern.finditer(text):
            add(
                f"percent:{_normalize_number(match.group(1))}",
                match.start(),
                match.end(),
            )
    return {key: tuple(spans) for key, spans in occurrences.items()}


def _question_anchor_tokens(question: str) -> tuple[str, ...]:
    """Extract non-generic terms used to associate a qualifier with a topic."""

    quoted_ranges = [match.span() for match in _QUOTED_RE.finditer(question)]
    tokens: list[str] = []
    for match in re.finditer(r"\w+", question, flags=re.UNICODE):
        if any(start <= match.start() < end for start, end in quoted_ranges):
            continue
        token = match.group(0).casefold()
        if _YEAR_RE.fullmatch(token):
            continue
        if token == "yüzde":
            continue
        if token.isdigit():
            continue
        if len(token) < 3 or token in _GENERIC_QUESTION_TERMS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tuple(tokens)


def _question_attribute_tokens(question: str) -> tuple[str, ...]:
    """Extract a small vocabulary of explicit requested attributes."""

    return tuple(
        sorted(
            {
                token.casefold()
                for token in re.findall(r"\w+", question, flags=re.UNICODE)
                if token.casefold() in _ATTRIBUTE_TERMS
            }
        )
    )


def _qualifier_has_context(
    qualifier: str,
    evidence: str,
    *,
    start: int,
    end: int,
    anchors: Sequence[str],
    attributes: Sequence[str],
) -> bool:
    """Require explicit numeric qualifiers near the requested topic anchor."""

    if not anchors:
        if attributes and qualifier.startswith("year:"):
            return _year_has_attribute_context(
                evidence,
                start=start,
                end=end,
                attributes=attributes,
            )
        return True
    # In tables and list-like PDF text, the program/entity normally precedes
    # its year or percentage value.  Looking only behind the value prevents a
    # neighboring row that follows it from satisfying the wrong program's
    # qualifier.
    window = evidence[max(0, start - _QUALIFIER_CONTEXT_RADIUS) : start]
    context_tokens = {
        token.casefold()
        for token in re.findall(r"\w+", window, flags=re.UNICODE)
    }
    if anchors[0] not in context_tokens:
        return False
    hit_count = sum(anchor in context_tokens for anchor in anchors)
    # Two of three entity/attribute anchors are enough for ordinary
    # abbreviations (e.g. ``İng.`` vs ``İngilizce``), while a two-token
    # subject still requires both terms.  This is intentionally a local,
    # deterministic association rather than an NER or semantic parser.
    required_hits = max(1, (2 * len(anchors) + 2) // 3)
    if hit_count < required_hits:
        return False
    if attributes and qualifier.startswith("year:"):
        return _year_has_attribute_context(
            evidence,
            start=start,
            end=end,
            attributes=attributes,
        )
    return True


def _year_has_attribute_context(
    evidence: str,
    *,
    start: int,
    end: int,
    attributes: Sequence[str],
) -> bool:
    """Require a requested year to label the requested attribute.

    If a different year appears between the year qualifier and the nearby
    attribute, the pair is treated as ambiguous.  This catches common table
    patterns such as ``2025 closing rank; 2026 quota`` without interpreting
    every number in a document.
    """

    window_start = max(0, start - _QUALIFIER_CONTEXT_RADIUS)
    window_end = min(len(evidence), end + _QUALIFIER_CONTEXT_RADIUS)
    for match in re.finditer(r"\w+", evidence[window_start:window_end], flags=re.UNICODE):
        token = match.group(0).casefold()
        if token not in attributes:
            continue
        attribute_position = window_start + match.start()
        between_start = min(start, attribute_position)
        between_end = max(end, attribute_position)
        for year in _YEAR_RE.finditer(evidence[between_start:between_end]):
            absolute_position = between_start + year.start()
            if absolute_position != start:
                break
        else:
            return True
    return False


def _normalize_number(value: str) -> str:
    """Normalize decimal separators without inventing numeric precision."""

    normalized = value.replace(",", ".").strip()
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _normalize_phrase(value: str) -> str:
    """Normalize a phrase for safe exact-token containment checks."""

    return " ".join(
        token.casefold()
        for token in re.findall(r"\w+", value, flags=re.UNICODE)
    )


@dataclass(frozen=True, slots=True)
class AnswerabilitySignals:
    """Evidence signals evaluated before an LLM is allowed to run."""

    evidence_count: int
    top_score: float | None
    score_margin: float | None
    coverage_ratio: float
    filters_satisfied: bool = True
    required_qualifiers: tuple[str, ...] = ()
    matched_qualifiers: tuple[str, ...] = ()
    missing_qualifiers: tuple[str, ...] = ()
    qualifier_coverage_satisfied: bool = True

    def __post_init__(self) -> None:
        if self.evidence_count < 0:
            raise ValueError("evidence_count must not be negative")
        if self.coverage_ratio < 0 or self.coverage_ratio > 1:
            raise ValueError("coverage_ratio must be between zero and one")


@dataclass(frozen=True, slots=True)
class AnswerabilityDecision:
    """Stable result of the pre-generation answerability gate."""

    decision: Decision
    reason: NoAnswerReason | None
    top_score: float | None
    score_margin: float | None
    coverage_ratio: float
    policy_profile: str = "default"
    calibration_id: str | None = None
    score_threshold: float | None = None
    coverage_threshold: float = 0.0
    required_qualifiers: tuple[str, ...] = ()
    matched_qualifiers: tuple[str, ...] = ()
    missing_qualifiers: tuple[str, ...] = ()
    qualifier_coverage_satisfied: bool = True


@dataclass(frozen=True, slots=True)
class AnswerabilityPolicy:
    """Apply provisional, explicitly configurable evidence thresholds.

    The thresholds are calibration inputs, not universal truths. They must be
    re-estimated on a golden validation split before a production rollout.
    ``min_margin`` and ``min_coverage`` default to zero because this first
    vertical slice records those signals before enough labeled data exists to
    turn them into safe rejection gates.
    """

    min_dense_score: float = 0.338
    min_sparse_score: float = 0.1
    min_rerank_score: float = -5.0
    min_margin: float = 0.0
    min_coverage: float = 0.0
    profile_name: str = "default"
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        if self.min_coverage < 0 or self.min_coverage > 1:
            raise ValueError("min_coverage must be between zero and one")

    def decide(
        self,
        *,
        signals: AnswerabilitySignals,
        score_kind: str,
    ) -> AnswerabilityDecision:
        """Return answered/no-answer without consulting an LLM."""

        if signals.evidence_count == 0 or signals.top_score is None:
            return self._no_answer(
                signals,
                NoAnswerReason.NO_EVIDENCE,
                score_kind=score_kind,
            )
        if not signals.filters_satisfied:
            return self._no_answer(
                signals,
                NoAnswerReason.INSUFFICIENT_COVERAGE,
                score_kind=score_kind,
            )

        minimum = self._minimum_for(score_kind)
        if signals.top_score < minimum:
            return self._no_answer(
                signals,
                NoAnswerReason.LOW_RELEVANCE,
                score_kind=score_kind,
            )
        if not signals.qualifier_coverage_satisfied:
            return self._no_answer(
                signals,
                NoAnswerReason.INSUFFICIENT_COVERAGE,
                score_kind=score_kind,
            )
        if signals.coverage_ratio < self.min_coverage:
            return self._no_answer(
                signals,
                NoAnswerReason.INSUFFICIENT_COVERAGE,
                score_kind=score_kind,
            )
        if (
            signals.score_margin is not None
            and signals.score_margin < self.min_margin
        ):
            return self._no_answer(
                signals,
                NoAnswerReason.LOW_RELEVANCE,
                score_kind=score_kind,
            )
        return AnswerabilityDecision(
            decision=Decision.ANSWERED,
            reason=None,
            top_score=signals.top_score,
            score_margin=signals.score_margin,
            coverage_ratio=signals.coverage_ratio,
            policy_profile=self.profile_name,
            calibration_id=self.calibration_id,
            score_threshold=minimum,
            coverage_threshold=self.min_coverage,
            required_qualifiers=signals.required_qualifiers,
            matched_qualifiers=signals.matched_qualifiers,
            missing_qualifiers=signals.missing_qualifiers,
            qualifier_coverage_satisfied=signals.qualifier_coverage_satisfied,
        )

    def _minimum_for(self, score_kind: str) -> float:
        if score_kind == "rerank":
            return self.min_rerank_score
        if score_kind == "sparse":
            return self.min_sparse_score
        return self.min_dense_score

    def _no_answer(
        self,
        signals: AnswerabilitySignals,
        reason: NoAnswerReason,
        *,
        score_kind: str,
    ) -> AnswerabilityDecision:
        score_threshold = None
        if signals.top_score is not None:
            score_threshold = self._minimum_for(score_kind)
        return AnswerabilityDecision(
            decision=Decision.NO_ANSWER,
            reason=reason,
            top_score=signals.top_score,
            score_margin=signals.score_margin,
            coverage_ratio=signals.coverage_ratio,
            policy_profile=self.profile_name,
            calibration_id=self.calibration_id,
            score_threshold=score_threshold,
            coverage_threshold=self.min_coverage,
            required_qualifiers=signals.required_qualifiers,
            matched_qualifiers=signals.matched_qualifiers,
            missing_qualifiers=signals.missing_qualifiers,
            qualifier_coverage_satisfied=signals.qualifier_coverage_satisfied,
        )


@dataclass(frozen=True, slots=True)
class AnswerabilityPolicySet:
    """Select a frozen policy by resolved chunking profile.

    The default is intentionally conservative: when evidence has no profile
    metadata or mixes profiles, the caller keeps the existing mentor/default
    policy instead of guessing that a generic calibration applies.
    """

    default: AnswerabilityPolicy
    by_chunking_profile: Mapping[str, AnswerabilityPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_chunking_profile", dict(self.by_chunking_profile))

    def select(self, profiles: Sequence[str]) -> AnswerabilityPolicy:
        """Return a profile policy only for one known, uniform profile."""

        known_profiles = {profile for profile in profiles if profile}
        if len(known_profiles) != 1:
            return self.default
        profile = next(iter(known_profiles))
        return self.by_chunking_profile.get(profile, self.default)
