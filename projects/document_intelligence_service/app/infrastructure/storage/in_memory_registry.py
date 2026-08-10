"""Development registry for idempotent ingestion acceptance.

This adapter is intentionally replaceable. It is not restart-safe and will be
replaced by durable document/job persistence before production deployment.
"""

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from ...domain.entities import DocumentStatus, JobStatus, StageStatus
from ...domain.errors import ErrorCode, ServiceError
from ...domain.ingestion import (
    DocumentPage,
    DocumentSnapshot,
    IngestionReceipt,
    JobSnapshot,
    PreparedIngestion,
    StageEvent,
    create_ingestion_receipt,
    normalize_idempotency_key,
    normalize_tenant_id,
)


@dataclass(slots=True)
class _StoredIngestion:
    receipt: IngestionReceipt
    identity: tuple[str, str, str]
    prepared: PreparedIngestion
    accepted_at: datetime


class InMemoryIngestionRegistry:
    """Bounded development state with atomic duplicate checks."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._by_identity: dict[tuple[str, str, str], _StoredIngestion] = {}
        self._by_idempotency: dict[str, _StoredIngestion] = {}
        self._idempotency_identity: dict[str, tuple[str, str, str]] = {}
        self._jobs: dict[str, JobSnapshot] = {}
        self._content_by_job: dict[str, bytes] = {}
        self._stage_events: dict[str, list[StageEvent]] = {}

    async def accept(
        self,
        prepared: PreparedIngestion,
        idempotency_key: str | None,
    ) -> IngestionReceipt:
        """Return an existing receipt or atomically create one."""

        identity = (
            prepared.upload.tenant_id,
            prepared.upload.content_hash,
            prepared.pipeline_fingerprint,
        )
        normalized_key = normalize_idempotency_key(idempotency_key)
        async with self._lock:
            if normalized_key is not None:
                previous_identity = self._idempotency_identity.get(normalized_key)
                if previous_identity is not None and previous_identity != identity:
                    raise ServiceError(
                        code=ErrorCode.INGESTION_CONFLICT,
                        message="Idempotency-Key was already used for another upload",
                    )
                existing_by_key = self._by_idempotency.get(normalized_key)
                if existing_by_key is not None:
                    return replace(existing_by_key.receipt, idempotent_hit=True)

            existing = self._by_identity.get(identity)
            if existing is not None:
                if normalized_key is not None:
                    self._by_idempotency[normalized_key] = existing
                    self._idempotency_identity[normalized_key] = identity
                return replace(existing.receipt, idempotent_hit=True)

            receipt = create_ingestion_receipt(identity)
            stored = _StoredIngestion(
                receipt=receipt,
                identity=identity,
                prepared=prepared,
                accepted_at=datetime.now(timezone.utc),
            )
            self._by_identity[identity] = stored
            if normalized_key is not None:
                self._by_idempotency[normalized_key] = stored
                self._idempotency_identity[normalized_key] = identity
            self._jobs[receipt.job_id] = JobSnapshot(
                job_id=receipt.job_id,
                document_id=receipt.document_id,
                status=JobStatus.QUEUED,
                progress_percent=0,
                error_code=None,
                page_count=prepared.pdf.page_count,
            )
            self._content_by_job[receipt.job_id] = prepared.content
            self._stage_events[receipt.job_id] = []
            return receipt

    async def get_job(self, job_id: str) -> JobSnapshot | None:
        """Return a queued job snapshot, if known."""

        async with self._lock:
            return self._jobs.get(job_id)

    async def claim_job(
        self,
        job_id: str,
        stale_after_seconds: float = 300.0,
    ) -> JobSnapshot | None:
        """Atomically claim a queued, retryable or stale-running job."""

        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        now = datetime.now(timezone.utc)
        async with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                return None
            if current.status is JobStatus.SUCCEEDED:
                return current
            if current.attempt_count >= current.max_attempts:
                return current
            if current.next_attempt_at is not None and current.next_attempt_at > now:
                return None
            if current.status is JobStatus.RUNNING and (
                current.last_attempt_at is not None
                and now - current.last_attempt_at
                < timedelta(seconds=stale_after_seconds)
            ):
                return None
            claimed = replace(
                current,
                status=JobStatus.RUNNING,
                progress_percent=max(current.progress_percent, 1),
                error_code=None,
                error_message=None,
                next_attempt_at=None,
                last_attempt_at=now,
                attempt_count=current.attempt_count + 1,
            )
            self._jobs[job_id] = claimed
            return claimed

    async def list_recoverable_jobs(
        self,
        limit: int = 10,
        stale_after_seconds: float = 300.0,
    ) -> tuple[str, ...]:
        """Return jobs eligible for the next worker polling pass."""

        if limit <= 0 or stale_after_seconds <= 0:
            raise ValueError("worker polling limits must be positive")
        now = datetime.now(timezone.utc)
        async with self._lock:
            candidates = [
                job
                for job in self._jobs.values()
                if _job_recoverable(job, now, stale_after_seconds)
            ]
            candidates.sort(key=lambda job: (job.next_attempt_at or now, job.job_id))
            return tuple(job.job_id for job in candidates[:limit])

    async def get_staged_content(self, job_id: str) -> bytes | None:
        """Return staged bytes for a future worker in this process."""

        async with self._lock:
            return self._content_by_job.get(job_id)

    async def get_staged_ingestion(self, job_id: str) -> PreparedIngestion | None:
        """Return the complete staged identity for the ingestion worker."""

        async with self._lock:
            for stored in self._by_identity.values():
                if stored.receipt.job_id == job_id:
                    return stored.prepared
        return None

    async def list_documents(
        self,
        limit: int,
        cursor: str | None,
        tenant_id: str = "default",
    ) -> DocumentPage:
        """Return stable cursor pagination over logical documents."""

        if limit <= 0 or limit > 100:
            raise ValueError("document limit must be between 1 and 100")
        offset = _parse_cursor(cursor)
        normalized_tenant = normalize_tenant_id(tenant_id)
        async with self._lock:
            snapshots = _snapshots(
                stored
                for stored in self._by_identity.values()
                if stored.prepared.upload.tenant_id == normalized_tenant
            )
        page = snapshots[offset : offset + limit]
        next_cursor = str(offset + limit) if offset + limit < len(snapshots) else None
        return DocumentPage(items=tuple(page), next_cursor=next_cursor)

    async def get_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> DocumentSnapshot | None:
        """Return one logical document and all known versions."""

        normalized_tenant = normalize_tenant_id(tenant_id)
        async with self._lock:
            records = tuple(
                stored
                for stored in self._by_identity.values()
                if stored.receipt.document_id == document_id
                and stored.prepared.upload.tenant_id == normalized_tenant
            )
            if not records:
                return None
            return _snapshot(records)

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> None:
        """Mark all versions deleted after rejecting active ingestion jobs."""

        normalized_tenant = normalize_tenant_id(tenant_id)
        async with self._lock:
            records = [
                stored
                for stored in self._by_identity.values()
                if stored.receipt.document_id == document_id
                and stored.prepared.upload.tenant_id == normalized_tenant
            ]
            if not records:
                raise ServiceError(
                    code=ErrorCode.DOCUMENT_NOT_FOUND,
                    message="Document was not found",
                )
            if any(
                self._jobs[stored.receipt.job_id].status
                in (JobStatus.QUEUED, JobStatus.RUNNING)
                for stored in records
            ):
                raise ServiceError(
                    code=ErrorCode.DOCUMENT_BUSY,
                    message="Document has an ingestion job in progress",
                )
            for stored in records:
                stored.receipt = IngestionReceipt(
                    document_id=stored.receipt.document_id,
                    version_id=stored.receipt.version_id,
                    job_id=stored.receipt.job_id,
                    status=DocumentStatus.DELETED,
                    idempotent_hit=stored.receipt.idempotent_hit,
                )

    async def update_job(self, snapshot: JobSnapshot) -> None:
        """Replace one job snapshot under the same registry lock."""

        async with self._lock:
            if snapshot.job_id not in self._jobs:
                raise KeyError(f"unknown job: {snapshot.job_id}")
            self._jobs[snapshot.job_id] = snapshot

    async def record_stage_event(self, job_id: str, event: StageEvent) -> None:
        """Append a transition while exposing the latest stage snapshot."""

        async with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job: {job_id}")
            history = self._stage_events.setdefault(job_id, [])
            history.append(event)
            latest: dict[str, StageEvent] = {item.name: item for item in history}
            current = self._jobs[job_id]
            outputs = event.outputs or {}
            point_count = outputs.get("points")
            self._jobs[job_id] = replace(
                current,
                current_stage=event.name,
                stages=tuple(latest.values()),
                point_count=(
                    int(point_count)
                    if isinstance(point_count, (int, float))
                    else current.point_count
                ),
                error_code=event.error_code or current.error_code,
                error_message=event.error_message or current.error_message,
                failed_stage=(
                    event.name
                    if event.status is StageStatus.FAILED
                    else current.failed_stage
                ),
            )

    async def set_document_status(
        self,
        *,
        document_id: str,
        version_id: str,
        status: DocumentStatus,
    ) -> None:
        """Update the receipt status without changing job progress."""

        async with self._lock:
            for stored in self._by_identity.values():
                if (
                    stored.receipt.document_id == document_id
                    and stored.receipt.version_id == version_id
                ):
                    stored.receipt = IngestionReceipt(
                        document_id=document_id,
                        version_id=version_id,
                        job_id=stored.receipt.job_id,
                        status=status,
                    )
                    return
        raise KeyError(f"unknown document version: {document_id}/{version_id}")


def _parse_cursor(cursor: str | None) -> int:
    """Parse the intentionally opaque offset cursor used by this adapter."""

    if cursor is None:
        return 0
    if not cursor.isdigit():
        raise ServiceError(
            code=ErrorCode.INVALID_REQUEST,
            message="Document cursor is invalid",
        )
    return int(cursor)


def _job_recoverable(
    job: JobSnapshot,
    now: datetime,
    stale_after_seconds: float,
) -> bool:
    """Return whether a job should be offered to a worker poller."""

    if job.status is JobStatus.SUCCEEDED or job.attempt_count >= job.max_attempts:
        return False
    if job.next_attempt_at is not None and job.next_attempt_at > now:
        return False
    if job.status is JobStatus.RUNNING:
        return job.last_attempt_at is None or (
            now - job.last_attempt_at
            >= timedelta(seconds=stale_after_seconds)
        )
    return job.status in (JobStatus.QUEUED, JobStatus.FAILED)


def _snapshots(
    records: Iterable[_StoredIngestion],
) -> list[DocumentSnapshot]:
    """Group stored versions into stable logical-document read models."""

    grouped: dict[str, list[_StoredIngestion]] = {}
    for record in records:
        grouped.setdefault(record.receipt.document_id, []).append(record)
    snapshots = [_snapshot(items) for items in grouped.values()]
    return sorted(
        snapshots,
        key=lambda item: (item.created_at, item.document_id),
        reverse=True,
    )


def _snapshot(
    records: list[_StoredIngestion] | tuple[_StoredIngestion, ...],
) -> DocumentSnapshot:
    """Build one public document read model from its stored versions."""

    ordered = sorted(records, key=lambda item: item.accepted_at)
    statuses = {item.receipt.status for item in ordered}
    if DocumentStatus.ACTIVE in statuses:
        status = DocumentStatus.ACTIVE
    elif DocumentStatus.INDEXING in statuses:
        status = DocumentStatus.INDEXING
    elif DocumentStatus.FAILED in statuses:
        status = DocumentStatus.FAILED
    else:
        status = DocumentStatus.DELETED
    active_versions = [
        item for item in ordered if item.receipt.status is DocumentStatus.ACTIVE
    ]
    latest = ordered[-1]
    return DocumentSnapshot(
        document_id=latest.receipt.document_id,
        title=latest.prepared.upload.filename,
        content_hash=latest.prepared.upload.content_hash,
        active_version_id=(
            active_versions[-1].receipt.version_id if active_versions else None
        ),
        status=status,
        created_at=ordered[0].accepted_at,
        available_version_ids=tuple(item.receipt.version_id for item in ordered),
        tenant_id=latest.prepared.upload.tenant_id,
    )
