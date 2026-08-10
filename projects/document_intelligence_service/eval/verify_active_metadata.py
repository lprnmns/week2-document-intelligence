"""Verify required metadata on every active Qdrant point."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


REQUIRED_FIELDS = frozenset(
    {
        "chunk_id",
        "parent_id",
        "document_id",
        # ``version_id`` is the service's canonical ingestion_version.  It is
        # content+pipeline derived and is intentionally stronger than a
        # mutable integer version label.
        "version_id",
        "source",
        "filename",
        "title_path",
        "text",
        "text_hash",
        "parent_text",
        "chunk_index",
        "page_start",
        "page_end",
        "pipeline_fingerprint",
        "content_hash",
        "embedding_model",
        "sparse_encoder",
        "parser_version",
        "chunker_version",
        "tenant_id",
        "acl_tags",
        "active",
        "is_active",
    }
)


def verify_active_metadata(
    *,
    client: QdrantClient,
    collection: str,
    page_size: int = 256,
) -> dict[str, object]:
    """Return a deterministic completeness report for active points."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    active_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True),
            )
        ]
    )
    active_points = client.count(
        collection_name=collection,
        count_filter=active_filter,
        exact=True,
    ).count

    missing_by_field = {field: 0 for field in sorted(REQUIRED_FIELDS)}
    pipeline_fingerprints: set[str] = set()
    embedding_models: set[str] = set()
    sparse_encoders: set[str] = set()
    ingestion_versions: set[str] = set()
    documents: set[str] = set()
    inspected = 0
    points_with_missing_metadata = 0
    offset: Any = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=active_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            inspected += 1
            payload = point.payload or {}
            point_missing = False
            for field in REQUIRED_FIELDS:
                value = payload.get(field)
                if value is None or value == "" or value == []:
                    missing_by_field[field] += 1
                    point_missing = True
            if point_missing:
                points_with_missing_metadata += 1

            _add_non_empty(payload.get("pipeline_fingerprint"), pipeline_fingerprints)
            _add_non_empty(payload.get("embedding_model"), embedding_models)
            _add_non_empty(payload.get("sparse_encoder"), sparse_encoders)
            _add_non_empty(payload.get("version_id"), ingestion_versions)
            _add_non_empty(payload.get("document_id"), documents)

        if next_offset is None or not points:
            break
        offset = next_offset

    return {
        "collection": collection,
        "active_points": active_points,
        "inspected_active_points": inspected,
        "missing_required_metadata": points_with_missing_metadata,
        "missing_by_field": {
            field: count
            for field, count in sorted(missing_by_field.items())
            if count
        },
        "pipeline_fingerprints": sorted(pipeline_fingerprints),
        "embedding_models": sorted(embedding_models),
        "sparse_encoders": sorted(sparse_encoders),
        "ingestion_versions": sorted(ingestion_versions),
        "documents": sorted(documents),
        "version_identity": "version_id (canonical ingestion_version)",
    }


def _add_non_empty(value: object, target: set[str]) -> None:
    """Collect only safe string identity values from payloads."""

    if isinstance(value, str) and value:
        target.add(value)


def main() -> None:
    """Run the verifier and optionally persist its JSON report."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("DIS_QDRANT_URL", "http://127.0.0.1:6335"),
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get(
            "DIS_QDRANT_COLLECTION",
            "document_chunks_week2_final_v1",
        ),
    )
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_active_metadata(
        client=QdrantClient(url=args.qdrant_url),
        collection=args.collection,
        page_size=args.page_size,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
