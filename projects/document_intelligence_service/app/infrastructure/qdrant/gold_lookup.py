"""Evaluation-only Qdrant lookup for trusted Gold Diagnostic locators."""

from typing import Any, cast

from qdrant_client import QdrantClient, models

from ...domain.gold_diagnostic import normalize_gold_text
from ...domain.retrieval import RetrievedChunk
from .schema import QdrantSchema, QdrantSchemaManager


class QdrantGoldEvidenceLookup:
    """Resolve a filename/page/text locator inside one active version only."""

    def __init__(self, client: QdrantClient, schema: QdrantSchema) -> None:
        self._client = client
        self._schema_manager = QdrantSchemaManager(client, schema)

    def find(
        self,
        *,
        document_id: str,
        version_id: str,
        page: int,
        must_contain: str,
        tenant_id: str = "default",
    ) -> tuple[RetrievedChunk, ...]:
        """Scroll a bounded active-version slice and match the locator locally."""

        self._schema_manager.ensure_collection()
        records, _ = self._client.scroll(
            collection_name=self._schema_manager.schema.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="active",
                        match=models.MatchValue(value=True),
                    ),
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchValue(value=version_id),
                    ),
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        locator = normalize_gold_text(must_contain)
        text_matches: list[RetrievedChunk] = []
        parent_matches: list[RetrievedChunk] = []
        for record in records:
            payload = cast(dict[str, Any], record.payload or {})
            page_start = _int(payload.get("page_start"))
            page_end = _int(payload.get("page_end"))
            text = _string(payload.get("text"))
            parent_text = _string(payload.get("parent_text"))
            if page_start is None or page_end is None or not text:
                continue
            if page < page_start or page > page_end:
                continue
            text_match = locator in normalize_gold_text(text)
            parent_match = locator in normalize_gold_text(parent_text)
            if not text_match and not parent_match:
                continue
            target = text_matches if text_match else parent_matches
            target.append(
                RetrievedChunk(
                    source_id=_required_string(payload, "chunk_id"),
                    document_id=_required_string(payload, "document_id"),
                    version_id=_required_string(payload, "version_id"),
                    parent_id=_required_string(payload, "parent_id"),
                    title=(
                        _string(payload.get("title"))
                        or _string(payload.get("filename"))
                    ),
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    score=0.0,
                    rank=len(target) + 1,
                    parent_text=parent_text or None,
                    tenant_id=_string(payload.get("tenant_id")) or tenant_id,
                    acl_tags=_tags(payload.get("acl_tags")),
                    chunking_profile=(
                        _string(payload.get("chunking_profile_resolved")) or None
                    ),
                )
            )
        matches = text_matches or parent_matches
        return tuple(sorted(matches, key=lambda item: (item.page_start, item.source_id)))


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = _string(payload.get(key))
    if not value:
        raise ValueError(f"gold lookup payload field {key!r} is missing")
    return value


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _tags(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ("public",)
    tags = tuple(item for item in value if isinstance(item, str) and item)
    return tags or ("public",)
