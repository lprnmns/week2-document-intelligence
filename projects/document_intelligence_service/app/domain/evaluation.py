"""Framework-independent evaluation run state and result contracts."""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .entities import EvaluationRunStatus, RetrievalMode

MetricValue = int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class EvaluationCorpusSnapshot:
    """Immutable membership manifest for one evaluation corpus.

    A pipeline fingerprint describes how points were produced.  It is not a
    corpus boundary, so the frozen benchmark also records the exact point IDs
    (and their document/version identity) that belong to the snapshot.
    """

    snapshot_id: str
    collection: str
    point_count: int
    point_ids: tuple[str, ...]
    document_versions: tuple[tuple[str, str], ...]
    pipeline_fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationCorpusSnapshot":
        """Validate and restore a committed snapshot manifest."""

        def required_string(key: str) -> str:
            item = value.get(key)
            if not isinstance(item, str) or not item:
                raise ValueError(f"corpus snapshot field {key!r} is required")
            return item

        point_count = value.get("point_count")
        if not isinstance(point_count, int) or isinstance(point_count, bool):
            raise ValueError("corpus snapshot point_count must be an integer")
        raw_point_ids = value.get("point_ids")
        if not isinstance(raw_point_ids, list) or not all(
            isinstance(item, str) and item for item in raw_point_ids
        ):
            raise ValueError("corpus snapshot point_ids must be a string list")
        point_ids = tuple(raw_point_ids)
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("corpus snapshot point_ids must be unique")
        if len(point_ids) != point_count:
            raise ValueError("corpus snapshot point_count does not match point_ids")

        raw_membership = value.get("document_versions")
        if not isinstance(raw_membership, list):
            raise ValueError("corpus snapshot document_versions must be a list")
        document_versions: list[tuple[str, str]] = []
        for item in raw_membership:
            if not isinstance(item, Mapping):
                raise ValueError("corpus snapshot membership entry must be an object")
            document_id = item.get("document_id")
            version_id = item.get("version_id")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("corpus snapshot document_id is required")
            if not isinstance(version_id, str) or not version_id:
                raise ValueError("corpus snapshot version_id is required")
            document_versions.append((document_id, version_id))

        return cls(
            snapshot_id=required_string("corpus_snapshot_id"),
            collection=required_string("collection"),
            point_count=point_count,
            point_ids=point_ids,
            document_versions=tuple(document_versions),
            pipeline_fingerprint=required_string("pipeline_fingerprint"),
        )

    def as_dict(self, *, include_point_ids: bool = True) -> dict[str, object]:
        """Return safe manifest metadata for run artifacts and API config."""

        payload: dict[str, object] = {
            "corpus_snapshot_id": self.snapshot_id,
            "collection": self.collection,
            "point_count": self.point_count,
            "document_versions": [
                {"document_id": document_id, "version_id": version_id}
                for document_id, version_id in self.document_versions
            ],
            "pipeline_fingerprint": self.pipeline_fingerprint,
        }
        if include_point_ids:
            payload["point_ids"] = list(self.point_ids)
        return payload


def load_corpus_snapshot(path: str | Path) -> EvaluationCorpusSnapshot:
    """Load one immutable JSON snapshot manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("corpus snapshot manifest must contain an object")
    return EvaluationCorpusSnapshot.from_mapping(payload)


def compute_corpus_snapshot_id(
    *,
    dataset_sha256: str | None,
    qdrant_collection: str,
    point_count: int | None,
    pipeline_fingerprint: str | None,
) -> str:
    """Return one stable identity for the evaluated corpus/configuration.

    The ID deliberately includes the dataset, active collection cardinality
    and vector-producing pipeline identity. It is a reproducibility handle,
    not a substitute for the metadata verifier or a Qdrant backup.
    """

    material = json.dumps(
        {
            "dataset_sha256": dataset_sha256,
            "qdrant_collection": qdrant_collection,
            "active_point_count": point_count,
            "pipeline_fingerprint": pipeline_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationRunSnapshot:
    """Public state of one reproducible offline benchmark invocation."""

    run_id: str
    status: EvaluationRunStatus
    evaluation_type: str
    dataset: str
    split: str
    mode: RetrievalMode
    top_k: int
    reranker_enabled: bool
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    case_count: int | None = None
    metrics: dict[str, MetricValue] | None = None
    artifact_path: str | None = None
    git_sha: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    configuration: dict[str, object] | None = None
