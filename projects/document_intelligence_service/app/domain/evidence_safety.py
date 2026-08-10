"""Framework-independent filtering for high-confidence indirect injections."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
import unicodedata

from .retrieval import RetrievedChunk


@dataclass(frozen=True, slots=True)
class EvidenceSafetyResult:
    """Safe evidence plus source/rule metadata for a retrieval trace."""

    safe_evidence: tuple[RetrievedChunk, ...]
    blocked_source_ids: tuple[str, ...]
    matched_rules: tuple[str, ...]

    @property
    def blocked_count(self) -> int:
        """Return how many retrieved evidence chunks were removed."""

        return len(self.blocked_source_ids)


@dataclass(frozen=True, slots=True)
class EvidenceSafetyPolicy:
    """Remove only high-confidence instruction/exfiltration evidence.

    Documents are allowed to discuss prompts or security concepts. The policy
    targets instruction-shaped payloads that try to override the assistant,
    impersonate a chat role, or exfiltrate secrets. It is a containment layer,
    not a claim that arbitrary natural language can be classified perfectly.
    """

    def filter(self, evidence: Sequence[RetrievedChunk]) -> EvidenceSafetyResult:
        """Keep ordinary evidence and remove chunks matching attack rules."""

        safe: list[RetrievedChunk] = []
        blocked_sources: list[str] = []
        matched_rules: list[str] = []
        for item in evidence:
            item_rules = tuple(
                rule_name
                for rule_name, pattern in _RULES
                if pattern.search(_normalize(item.context_text))
            )
            if item_rules:
                blocked_sources.append(item.source_id)
                for rule in item_rules:
                    if rule not in matched_rules:
                        matched_rules.append(rule)
                continue
            safe.append(item)
        return EvidenceSafetyResult(
            safe_evidence=tuple(safe),
            blocked_source_ids=tuple(blocked_sources),
            matched_rules=tuple(matched_rules),
        )


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "evidence_instruction_override",
        re.compile(
            r"\b(?:onceki|previous)\s+(?:talimat|instruction|kural)\w*"
            r"\s+(?:yok say|ignore|gormezden gel)\b"
        ),
    ),
    (
        "chat_role_injection",
        re.compile(
            r"(?:<\|(?:system|assistant|user)\|>|###\s*(?:system|assistant))"
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(?:system prompt|gizli kural\w*|api key|secret)\b"
            r".{0,100}\b(?:gonder|send|paylas|email|http|url)\b"
        ),
    ),
)


def _normalize(text: str) -> str:
    """Normalize accents/case while preserving chat marker punctuation."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
