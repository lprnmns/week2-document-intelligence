"""Evaluation-only Qdrant lookup for trusted Gold Diagnostic locators."""

from collections.abc import Sequence
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

    def browse(
        self,
        *,
        document_ids: Sequence[str],
        page: int | None = None,
        text: str = "",
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        limit: int = 50,
    ) -> tuple[RetrievedChunk, ...]:
        """Browse active child chunks without using query retrieval ranking."""

        self._schema_manager.ensure_collection()
        normalized_ids = tuple(dict.fromkeys(item for item in document_ids if item))
        normalized_tags = tuple(dict.fromkeys(("public", *acl_tags)))
        conditions: list[models.Condition] = [
            models.FieldCondition(key="active", match=models.MatchValue(value=True)),
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=tenant_id)
            ),
            models.FieldCondition(
                key="acl_tags", match=models.MatchAny(any=list(normalized_tags))
            ),
        ]
        if normalized_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(normalized_ids)),
                )
            )
        records, _ = self._client.scroll(
            collection_name=self._schema_manager.schema.collection_name,
            scroll_filter=models.Filter(must=conditions),
            limit=min(max(limit * 4, limit), 1000),
            with_payload=True,
            with_vectors=False,
        )
        needle = normalize_gold_text(text) if text.strip() else ""
        found: list[RetrievedChunk] = []
        direct_matches: list[RetrievedChunk] = []
        parent_matches: list[RetrievedChunk] = []
        for record in records:
            payload = cast(dict[str, Any], record.payload or {})
            item = _payload_chunk(payload, tenant_id)
            if item is None:
                continue
            if page is not None and not (item.page_start <= page <= item.page_end):
                continue
            if not needle:
                found.append(item)
            elif needle in normalize_gold_text(item.text):
                direct_matches.append(item)
            elif needle in normalize_gold_text(item.parent_text or ""):
                parent_matches.append(item)
        if needle:
            # Prefer the exact child chunks. Parent context is intentionally
            # only a fallback when no child text contains the search term;
            # otherwise one matching parent makes every sibling look like a
            # direct result in the trusted-evidence picker.
            found = direct_matches or parent_matches
        return tuple(sorted(found, key=lambda item: (item.page_start, item.source_id))[:limit])

    def find_source_ids(
        self,
        *,
        source_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> tuple[RetrievedChunk, ...]:
        """Resolve child IDs only inside active tenant/ACL scope."""

        normalized_ids = tuple(dict.fromkeys(item for item in source_ids if item))
        if not normalized_ids:
            return ()
        self._schema_manager.ensure_collection()
        records, _ = self._client.scroll(
            collection_name=self._schema_manager.schema.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="active", match=models.MatchValue(value=True)
                    ),
                    models.FieldCondition(
                        key="tenant_id", match=models.MatchValue(value=tenant_id)
                    ),
                    models.FieldCondition(
                        key="acl_tags",
                        match=models.MatchAny(any=list(dict.fromkeys(("public", *acl_tags)))),
                    ),
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchAny(any=list(normalized_ids)),
                    ),
                ]
            ),
            limit=min(max(len(normalized_ids) * 2, 50), 1000),
            with_payload=True,
            with_vectors=False,
        )
        found = []
        for record in records:
            item = _payload_chunk(cast(dict[str, Any], record.payload or {}), tenant_id)
            if item is not None:
                found.append(item)
        by_id = {item.source_id: item for item in found}
        return tuple(by_id[item] for item in normalized_ids if item in by_id)


def _payload_chunk(payload: dict[str, Any], tenant_id: str) -> RetrievedChunk | None:
    """Map one active payload to a safe browse result."""

    page_start = _int(payload.get("page_start"))
    page_end = _int(payload.get("page_end"))
    text = _string(payload.get("text"))
    source_id = _string(payload.get("chunk_id"))
    if page_start is None or page_end is None or not text or not source_id:
        return None
    return RetrievedChunk(
        source_id=source_id,
        document_id=_required_string(payload, "document_id"),
        version_id=_required_string(payload, "version_id"),
        parent_id=_required_string(payload, "parent_id"),
        title=_string(payload.get("title")) or _string(payload.get("filename")),
        text=text,
        page_start=page_start,
        page_end=page_end,
        score=0.0,
        rank=0,
        parent_text=_string(payload.get("parent_text")) or None,
        tenant_id=_string(payload.get("tenant_id")) or tenant_id,
        acl_tags=_tags(payload.get("acl_tags")),
        chunking_profile=_string(payload.get("chunking_profile_resolved")) or None,
    )


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
