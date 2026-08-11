"""Bounded, observable prompt-packing value objects."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptEvidenceFragment:
    """The exact evidence fragment included in one grounded prompt."""

    source_id: str
    document_id: str
    page_start: int
    page_end: int
    included_text: str
    included_chars: int
    child_included: bool
    parent_context_chars: int
    truncated: bool
    excluded_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a bounded trace/UI projection."""

        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "included_text": self.included_text,
            "included_chars": self.included_chars,
            "child_included": self.child_included,
            "parent_context_chars": self.parent_context_chars,
            "truncated": self.truncated,
            "excluded_reason": self.excluded_reason,
        }


@dataclass(frozen=True, slots=True)
class PromptPackResult:
    """The authoritative result of packing selected evidence for generation."""

    prompt: str
    fragments: tuple[PromptEvidenceFragment, ...]
    selected_source_ids: tuple[str, ...]
    included_source_ids: tuple[str, ...]
    excluded_source_ids: tuple[str, ...]
    total_evidence_chars: int
    configured_budget_chars: int

    @property
    def membership_observed(self) -> bool:
        """Whether prompt membership is known from the real pack operation."""

        return True

    def as_dict(self, *, include_prompt: bool = False) -> dict[str, object]:
        """Return safe prompt-pack metadata without exposing the full prompt."""

        payload: dict[str, object] = {
            "selected_source_ids": list(self.selected_source_ids),
            "included_source_ids": list(self.included_source_ids),
            "excluded_source_ids": list(self.excluded_source_ids),
            "selected_count": len(self.selected_source_ids),
            "included_count": len(self.included_source_ids),
            "excluded_count": len(self.excluded_source_ids),
            "total_evidence_chars": self.total_evidence_chars,
            "configured_budget_chars": self.configured_budget_chars,
            "fragments": [fragment.as_dict() for fragment in self.fragments],
            "membership_observed": self.membership_observed,
        }
        if include_prompt:
            payload["prompt"] = self.prompt
        return payload
