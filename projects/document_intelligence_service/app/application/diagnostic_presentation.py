"""Bounded, user-facing projections for curated diagnostic runs.

The raw trace remains an engineering transport.  These small DTOs are the
stable presentation boundary used by the demo UI so a recursive trace
sanitizer can never turn a source, rank or required fact into ``[truncated]``.
"""

from collections.abc import Mapping
from dataclasses import dataclass


PRESENTATION_TEXT_LIMIT = 4000
PRESENTATION_EXCERPT_LIMIT = 520


def bounded_text(value: object, limit: int = PRESENTATION_TEXT_LIMIT) -> str:
    """Return a readable, bounded string without exposing an omission token."""

    if not isinstance(value, str):
        return ""
    if value.strip() == "[truncated]":
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 1, 1)].rstrip() + "…"


@dataclass(frozen=True, slots=True)
class DiagnosticChunkView:
    """One bounded chunk identity/content/rank projection."""

    source_id: str
    document_id: str
    title: str
    page_start: int | None
    page_end: int | None
    parent_id: str
    chunk_text: str
    parent_context: str
    excerpt: str
    dense_rank: int | None
    sparse_rank: int | None
    fusion_rank: int | None
    rerank_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    fused_score: float | None
    rerank_score: float | None
    selected_as_evidence: bool
    used_in_prompt: bool
    trusted: bool

    def as_dict(self) -> dict[str, object]:
        """Return only primitive, bounded UI fields."""

        return {
            "source_id": self.source_id,
            "document_id": self.document_id,
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "parent_id": self.parent_id,
            "chunk_text": self.chunk_text,
            "parent_context": self.parent_context,
            "excerpt": self.excerpt,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "fusion_rank": self.fusion_rank,
            "rerank_rank": self.rerank_rank,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "fused_score": self.fused_score,
            "rerank_score": self.rerank_score,
            "selected_as_evidence": self.selected_as_evidence,
            "used_in_prompt": self.used_in_prompt,
            "trusted": self.trusted,
        }


@dataclass(frozen=True, slots=True)
class FactBoundaryView:
    """Required-fact survival across the real pipeline boundaries."""

    fact_type: str
    value: str
    trusted: bool
    dense: bool | None
    bm25: bool | None
    rrf: bool | None
    reranker: bool | None
    evidence: bool
    prompt: bool
    final: bool

    def as_dict(self) -> dict[str, object]:
        """Return a compact fact-survival table row."""

        return {
            "type": self.fact_type,
            "value": self.value,
            "trusted": self.trusted,
            "dense": self.dense,
            "bm25": self.bm25,
            "rrf": self.rrf,
            "reranker": self.reranker,
            "evidence": self.evidence,
            "prompt": self.prompt,
            "final": self.final,
        }


def bounded_claims(value: Mapping[str, object] | None) -> dict[str, list[dict[str, object]]]:
    """Project structured claim comparisons into primitive UI fields.

    Claims are nested inside the diagnostic result, so sending the raw mapping
    through the generic trace sanitizer can omit leaf values at its depth
    boundary.  This projection keeps the reviewer-facing claim id, type,
    value and verdict bounded and readable.
    """

    result: dict[str, list[dict[str, object]]] = {}
    for bucket in ("expected", "forbidden"):
        raw_rows = value.get(bucket) if isinstance(value, Mapping) else None
        rows: list[dict[str, object]] = []
        if isinstance(raw_rows, (list, tuple)):
            for raw in raw_rows[:40]:
                if not isinstance(raw, Mapping):
                    continue
                claim_id = raw.get("claim_id")
                claim_type = raw.get("type")
                rows.append(
                    {
                        "claim_id": claim_id if isinstance(claim_id, str) else "claim",
                        "type": claim_type if isinstance(claim_type, str) else "claim",
                        "value": bounded_text(raw.get("value"), 300),
                        "matched": bool(raw.get("matched")),
                    }
                )
        result[bucket] = rows
    return result


@dataclass(frozen=True, slots=True)
class RerankerMovementView:
    """Before/after movement for one actual reranker candidate."""

    chunk: DiagnosticChunkView
    before_rank: int | None
    after_rank: int | None
    movement: int | None
    required_facts_carried: tuple[str, ...]
    status: str

    def as_dict(self) -> dict[str, object]:
        """Return a bounded movement row."""

        return {
            "chunk": self.chunk.as_dict(),
            "before_rank": self.before_rank,
            "after_rank": self.after_rank,
            "movement": self.movement,
            "required_facts_carried": list(self.required_facts_carried),
            "status": self.status,
        }


def chunk_view_from_mapping(
    item: Mapping[str, object],
    *,
    trusted: bool = False,
    selected_as_evidence: bool | None = None,
    used_in_prompt: bool | None = None,
) -> DiagnosticChunkView:
    """Build a chunk DTO from a trace/result mapping."""

    source_id = _string(item.get("source_id"))
    chunk_text = bounded_text(
        item.get("chunk_text") or item.get("text") or item.get("excerpt")
    )
    parent_context = bounded_text(
        item.get("parent_context") or item.get("context_text") or chunk_text
    )
    excerpt = bounded_text(
        item.get("excerpt") or chunk_text,
        PRESENTATION_EXCERPT_LIMIT,
    )
    return DiagnosticChunkView(
        source_id=source_id,
        document_id=_string(item.get("document_id")),
        title=_string(item.get("title")),
        page_start=_int_or_none(item.get("page_start")),
        page_end=_int_or_none(item.get("page_end")),
        parent_id=_string(item.get("parent_id")),
        chunk_text=chunk_text,
        parent_context=parent_context,
        excerpt=excerpt,
        dense_rank=_int_or_none(item.get("dense_rank")),
        sparse_rank=_int_or_none(item.get("sparse_rank")),
        fusion_rank=_int_or_none(item.get("fusion_rank") or item.get("rank")),
        rerank_rank=_int_or_none(item.get("rerank_rank")),
        dense_score=_float_or_none(item.get("dense_score")),
        sparse_score=_float_or_none(item.get("sparse_score")),
        fused_score=_float_or_none(item.get("fused_score")),
        rerank_score=_float_or_none(item.get("rerank_score")),
        selected_as_evidence=(
            bool(item.get("selected_as_evidence"))
            if selected_as_evidence is None
            else selected_as_evidence
        ),
        used_in_prompt=(
            bool(item.get("used_in_prompt"))
            if used_in_prompt is None
            else used_in_prompt
        ),
        trusted=trusted or bool(item.get("trusted")),
    )


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_or_none(value: object) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
