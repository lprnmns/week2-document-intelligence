"""Qdrant dense and sparse retrieval adapter."""

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from qdrant_client import QdrantClient, models

from ...domain.retrieval import RetrievedChunk
from ...domain.vectors import SparseVector
from .schema import QdrantSchema, QdrantSchemaManager


class QdrantRetriever:
    """Search only active Qdrant points and restore source metadata."""

    def __init__(
        self,
        client: QdrantClient,
        schema: QdrantSchema,
        pipeline_fingerprint: str | None = None,
        corpus_point_ids: Sequence[str] = (),
        active_version_ids_provider: Callable[[Sequence[str], str], Sequence[str]]
        | None = None,
    ) -> None:
        self._client = client
        self._schema_manager = QdrantSchemaManager(client, schema)
        self._pipeline_fingerprint = pipeline_fingerprint
        self._corpus_point_ids = tuple(
            dict.fromkeys(point_id for point_id in corpus_point_ids if point_id)
        )
        self._active_version_ids_provider = active_version_ids_provider

    def snapshot_active_version_ids(
        self,
        document_ids: Sequence[str],
        tenant_id: str,
    ) -> tuple[str, ...] | None:
        """Capture the authoritative product-version scope once per query."""

        if self._active_version_ids_provider is None:
            return None
        return tuple(
            dict.fromkeys(
                version_id
                for version_id in self._active_version_ids_provider(
                    document_ids,
                    tenant_id,
                )
                if version_id
            )
        )

    def search_dense(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        active_version_ids: Sequence[str] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """Run cosine search on the named dense vector."""

        self._validate_limit(limit)
        resolved_active_versions = self._resolve_active_version_ids(
            active_version_ids,
            document_ids,
            tenant_id,
        )
        if resolved_active_versions is not None and not resolved_active_versions:
            return ()
        self._schema_manager.ensure_collection()
        response = self._client.query_points(
            collection_name=self._schema_manager.schema.collection_name,
            query=list(query_vector),
            using=self._schema_manager.schema.dense_name,
            query_filter=self._active_filter(
                document_ids,
                tenant_id,
                acl_tags,
                pipeline_fingerprint=self._pipeline_fingerprint,
                corpus_point_ids=self._corpus_point_ids,
                active_version_ids=resolved_active_versions,
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return tuple(
            self._map_point(point, rank=index, score_kind="dense")
            for index, point in enumerate(response.points, start=1)
        )

    def search_sparse(
        self,
        *,
        query_vector: SparseVector,
        limit: int,
        document_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        active_version_ids: Sequence[str] | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """Run lexical search on the named IDF sparse vector."""

        self._validate_limit(limit)
        resolved_active_versions = self._resolve_active_version_ids(
            active_version_ids,
            document_ids,
            tenant_id,
        )
        if resolved_active_versions is not None and not resolved_active_versions:
            return ()
        self._schema_manager.ensure_collection()
        response = self._client.query_points(
            collection_name=self._schema_manager.schema.collection_name,
            query=models.SparseVector(
                indices=list(query_vector.indices),
                values=list(query_vector.values),
            ),
            using=self._schema_manager.schema.sparse_name,
            query_filter=self._active_filter(
                document_ids,
                tenant_id,
                acl_tags,
                pipeline_fingerprint=self._pipeline_fingerprint,
                corpus_point_ids=self._corpus_point_ids,
                active_version_ids=resolved_active_versions,
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return tuple(
            self._map_point(point, rank=index, score_kind="sparse")
            for index, point in enumerate(response.points, start=1)
        )

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit <= 0 or limit > 50:
            raise ValueError("retrieval limit must be between 1 and 50")

    @staticmethod
    def _active_filter(
        document_ids: Sequence[str],
        tenant_id: str,
        acl_tags: Sequence[str],
        pipeline_fingerprint: str | None = None,
        corpus_point_ids: Sequence[str] = (),
        active_version_ids: Sequence[str] | None = None,
    ) -> models.Filter:
        active_condition = models.FieldCondition(
            key="active",
            match=models.MatchValue(value=True),
        )
        normalized_ids = tuple(dict.fromkeys(document_id for document_id in document_ids if document_id))
        normalized_tags = tuple(
            dict.fromkeys(("public", *(tag for tag in acl_tags if tag)))
        )
        must: list[models.Condition] = [
            active_condition,
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="acl_tags",
                match=models.MatchAny(any=list(normalized_tags or ("public",))),
            ),
        ]
        if pipeline_fingerprint:
            must.append(
                models.FieldCondition(
                    key="pipeline_fingerprint",
                    match=models.MatchValue(value=pipeline_fingerprint),
                )
            )
        if normalized_ids:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(normalized_ids)),
                )
            )
        normalized_point_ids = tuple(
            dict.fromkeys(point_id for point_id in corpus_point_ids if point_id)
        )
        if normalized_point_ids:
            must.append(models.HasIdCondition(has_id=list(normalized_point_ids)))
        if active_version_ids is not None:
            normalized_version_ids = tuple(
                dict.fromkeys(version_id for version_id in active_version_ids if version_id)
            )
            if normalized_version_ids:
                must.append(
                    models.FieldCondition(
                        key="version_id",
                        match=models.MatchAny(any=list(normalized_version_ids)),
                    )
                )
        return models.Filter(must=must)

    def _resolve_active_version_ids(
        self,
        active_version_ids: Sequence[str] | None,
        document_ids: Sequence[str],
        tenant_id: str,
    ) -> tuple[str, ...] | None:
        """Use a caller snapshot or capture the provider's current scope."""

        if active_version_ids is not None:
            return tuple(dict.fromkeys(item for item in active_version_ids if item))
        return self.snapshot_active_version_ids(document_ids, tenant_id)

    @classmethod
    def _map_point(
        cls,
        point: models.ScoredPoint,
        *,
        rank: int,
        score_kind: Literal["dense", "sparse"],
    ) -> RetrievedChunk:
        payload = cast(dict[str, Any], point.payload or {})
        score = float(point.score)
        return RetrievedChunk(
            source_id=cls._required_string(payload, "chunk_id"),
            document_id=cls._required_string(payload, "document_id"),
            version_id=cls._required_string(payload, "version_id"),
            parent_id=cls._required_string(payload, "parent_id"),
            title=cls._optional_string(payload, "title"),
            text=cls._required_string(payload, "text"),
            page_start=cls._required_int(payload, "page_start"),
            page_end=cls._required_int(payload, "page_end"),
            score=score,
            rank=rank,
            dense_score=score if score_kind == "dense" else None,
            sparse_score=score if score_kind == "sparse" else None,
            parent_text=cls._optional_string(payload, "parent_text") or None,
            tenant_id=cls._optional_string(payload, "tenant_id") or "default",
            acl_tags=cls._optional_tags(payload, "acl_tags"),
            chunking_profile=cls._optional_string(
                payload,
                "chunking_profile_resolved",
            ) or None,
        )

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Qdrant payload field {key!r} must be a non-empty string")
        return value

    @staticmethod
    def _optional_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        return value if isinstance(value, str) else ""

    @staticmethod
    def _optional_tags(payload: dict[str, Any], key: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list):
            return ("public",)
        return tuple(item for item in value if isinstance(item, str)) or ("public",)

    @staticmethod
    def _required_int(payload: dict[str, Any], key: str) -> int:
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Qdrant payload field {key!r} must be an integer")
        return value
