"""Framework-independent checks for direct prompt-injection attempts."""

from dataclasses import dataclass
from enum import StrEnum
import re
import unicodedata


class PromptSafetyReason(StrEnum):
    """Reason for declining a query before retrieval or generation."""

    DIRECT_INJECTION = "DIRECT_INJECTION"


@dataclass(frozen=True, slots=True)
class PromptSafetyDecision:
    """Deterministic safety result retained for policy and audit decisions."""

    allowed: bool
    reason: PromptSafetyReason | None
    matched_rules: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        """Return whether the query must not enter the RAG answer path."""

        return not self.allowed


@dataclass(frozen=True, slots=True)
class PromptSafetyPolicy:
    """Detect high-confidence direct injection patterns before the LLM.

    This is intentionally a narrow policy, not a general language-understanding
    classifier. A query that merely discusses system prompts is allowed; a
    query that asks to reveal them, override source rules or manufacture an
    unsupported fact is blocked. Unknown attacks still require structured
    prompting, provenance and output validation as defense-in-depth layers.
    """

    def evaluate(self, question: str) -> PromptSafetyDecision:
        """Return the matched high-confidence rules for one user question."""

        normalized = _normalize(question)
        matched = tuple(
            rule_name
            for rule_name, pattern in _RULES
            if pattern.search(normalized)
        )
        if matched:
            return PromptSafetyDecision(
                allowed=False,
                reason=PromptSafetyReason.DIRECT_INJECTION,
                matched_rules=matched,
            )
        return PromptSafetyDecision(
            allowed=True,
            reason=None,
            matched_rules=(),
        )


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\bonceki\s+(?:kural|talimat)\w*\s+"
            r"(?:yok say|gormezden gel)\b"
        ),
    ),
    (
        "source_override",
        re.compile(r"\bkaynak\w*\s+(?:yok say|gormezden gel)\b"),
    ),
    (
        "acl_filter_bypass",
        re.compile(
            r"\b(?:document|dokuman)\s+id\s+filtre\w*\s+"
            r"(?:yok say|gormezden gel|atlat|as)\w*\b"
        ),
    ),
    (
        "secret_extraction",
        re.compile(
            r"\b(?:system prompt\w*|gizli kural\w*)\b"
            r".{0,80}\b(?:goster|paylas|acikla|soyle|yaz|ortaya cikar)\b"
        ),
    ),
    (
        "unsupported_claim_injection",
        re.compile(
            r"\b(?:dokumanda|kaynakta)\s+olmayan\b"
            r".{0,100}\b(?:uydur|kesin|gercek|sun|yaz)\b"
        ),
    ),
    (
        "assumed_claim_injection",
        re.compile(
            r"\bvarsay\b.{0,100}\b(?:dokumanda|kaynakta)\s+olmayan\b"
        ),
    ),
)


def _normalize(text: str) -> str:
    """Normalize case, Turkish diacritics and punctuation for rule matching."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^\w\s]", " ", without_marks).split())
