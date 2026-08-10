"""Restart-safe SQLite adapter for ingestion identities, jobs and staged PDFs."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from ...domain.entities import DocumentStatus, JobStatus, StageStatus
from ...domain.errors import ErrorCode, ServiceError
from ...domain.ingestion import (
    DocumentPage,
    DocumentSnapshot,
    IngestionReceipt,
    JobSnapshot,
    PdfInspection,
    PreparedIngestion,
    StageEvent,
    UploadMetadata,
    ChunkingResolution,
    create_ingestion_receipt,
    restore_pipeline_config,
    serialize_pipeline_metadata,
    normalize_idempotency_key,
    normalize_tenant_id,
)

class SqliteIngestionRegistry:
    """Persist accepted jobs and PDF bytes so another process can resume them."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        if str(self._database_path) != ":memory:":
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def database_path(self) -> Path:
        """Return the configured SQLite file path."""

        return self._database_path

    async def accept(
        self,
        prepared: PreparedIngestion,
        idempotency_key: str | None,
    ) -> IngestionReceipt:
        """Atomically accept or reuse one content/pipeline identity."""

        return await asyncio.to_thread(
            self._accept_sync,
            prepared,
            normalize_idempotency_key(idempotency_key),
        )

    async def get_job(self, job_id: str) -> JobSnapshot | None:
        """Read one durable job snapshot."""

        return await asyncio.to_thread(self._get_job_sync, job_id)

    async def claim_job(
        self,
        job_id: str,
        stale_after_seconds: float = 300.0,
    ) -> JobSnapshot | None:
        """Atomically claim a queued, retryable or stale-running job."""

        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        return await asyncio.to_thread(
            self._claim_job_sync,
            job_id,
            stale_after_seconds,
        )

    async def list_recoverable_jobs(
        self,
        limit: int = 10,
        stale_after_seconds: float = 300.0,
    ) -> tuple[str, ...]:
        """Return bounded jobs that the separate worker should poll."""

        if limit <= 0 or stale_after_seconds <= 0:
            raise ValueError("worker polling limits must be positive")
        return await asyncio.to_thread(
            self._list_recoverable_jobs_sync,
            limit,
            stale_after_seconds,
        )

    async def get_staged_content(self, job_id: str) -> bytes | None:
        """Read the staged PDF bytes for a worker."""

        prepared = await self.get_staged_ingestion(job_id)
        return prepared.content if prepared is not None else None

    async def get_staged_ingestion(self, job_id: str) -> PreparedIngestion | None:
        """Read the complete staged identity required by the worker."""

        return await asyncio.to_thread(self._get_staged_sync, job_id)

    async def list_documents(
        self,
        limit: int,
        cursor: str | None,
        tenant_id: str = "default",
    ) -> DocumentPage:
        """Return stable cursor pagination over logical documents."""

        if limit <= 0 or limit > 100:
            raise ValueError("document limit must be between 1 and 100")
        return await asyncio.to_thread(
            self._list_documents_sync,
            limit,
            cursor,
            normalize_tenant_id(tenant_id),
        )

    async def get_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> DocumentSnapshot | None:
        """Return one logical document and all known versions."""

        return await asyncio.to_thread(
            self._get_document_sync,
            document_id,
            normalize_tenant_id(tenant_id),
        )

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> None:
        """Mark all versions deleted unless an ingestion is still running."""

        await asyncio.to_thread(
            self._delete_document_sync,
            document_id,
            normalize_tenant_id(tenant_id),
        )

    async def update_job(self, snapshot: JobSnapshot) -> None:
        """Persist one worker progress transition."""

        await asyncio.to_thread(self._update_job_sync, snapshot)

    async def record_stage_event(self, job_id: str, event: StageEvent) -> None:
        """Append one stage transition and update the job summary."""

        await asyncio.to_thread(self._record_stage_event_sync, job_id, event)

    async def set_document_status(
        self,
        *,
        document_id: str,
        version_id: str,
        status: DocumentStatus,
    ) -> None:
        """Persist a version lifecycle transition independently from the job."""

        await asyncio.to_thread(
            self._set_document_status_sync,
            document_id,
            version_id,
            status,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS ingestions (
                    content_hash TEXT NOT NULL,
                    pipeline_fingerprint TEXT NOT NULL,
                    pipeline_config_json TEXT NOT NULL DEFAULT '{}',
                    document_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    job_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    acl_tags_json TEXT NOT NULL DEFAULT '["public"]',
                    created_at TEXT NOT NULL,
                    content BLOB NOT NULL,
                    document_status TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_percent INTEGER NOT NULL,
                    error_code TEXT,
                    current_stage TEXT,
                    point_count INTEGER,
                    error_message TEXT,
                    failed_stage TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    next_attempt_at TEXT,
                    last_attempt_at TEXT,
                    PRIMARY KEY (tenant_id, content_hash, pipeline_fingerprint)
                );

                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    content_hash TEXT NOT NULL,
                    pipeline_fingerprint TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_ingestions_job_id
                    ON ingestions(job_id);

                CREATE TABLE IF NOT EXISTS ingestion_stage_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_ms REAL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    decision TEXT,
                    warnings_json TEXT NOT NULL,
                    error_code TEXT,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_stage_events_job_id
                    ON ingestion_stage_events(job_id, event_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(ingestions)")
            }
            if "document_status" not in columns:
                connection.execute(
                    """
                    ALTER TABLE ingestions
                    ADD COLUMN document_status TEXT NOT NULL DEFAULT 'indexing'
                    """
                )
            for column, definition in (
                ("pipeline_config_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
                ("acl_tags_json", "TEXT NOT NULL DEFAULT '[\"public\"]'"),
                (
                    "created_at",
                    "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00+00:00'",
                ),
                ("current_stage", "TEXT"),
                ("point_count", "INTEGER"),
                ("error_message", "TEXT"),
                ("failed_stage", "TEXT"),
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("max_attempts", "INTEGER NOT NULL DEFAULT 3"),
                ("next_attempt_at", "TEXT"),
                ("last_attempt_at", "TEXT"),
            ):
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE ingestions ADD COLUMN {column} {definition}"
                    )
            idempotency_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(idempotency_keys)")
            }
            if "tenant_id" not in idempotency_columns:
                connection.execute(
                    "ALTER TABLE idempotency_keys ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )
            self._migrate_tenant_scoped_primary_key(connection)

    @staticmethod
    def _migrate_tenant_scoped_primary_key(connection: sqlite3.Connection) -> None:
        """Rebuild pre-ACL databases whose identity key was tenant-blind."""

        primary_key_columns = [
            row["name"]
            for row in sorted(
                connection.execute("PRAGMA table_info(ingestions)").fetchall(),
                key=lambda row: row["pk"],
            )
            if row["pk"]
        ]
        expected = ["tenant_id", "content_hash", "pipeline_fingerprint"]
        if primary_key_columns == expected:
            return
        if primary_key_columns != ["content_hash", "pipeline_fingerprint"]:
            raise RuntimeError("unsupported ingestions primary key schema")

        connection.execute(
            """
            CREATE TABLE ingestions_v2 (
                content_hash TEXT NOT NULL,
                pipeline_fingerprint TEXT NOT NULL,
                pipeline_config_json TEXT NOT NULL DEFAULT '{}',
                document_id TEXT NOT NULL,
                version_id TEXT NOT NULL,
                job_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                acl_tags_json TEXT NOT NULL DEFAULT '["public"]',
                created_at TEXT NOT NULL,
                content BLOB NOT NULL,
                document_status TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_percent INTEGER NOT NULL,
                error_code TEXT,
                current_stage TEXT,
                point_count INTEGER,
                error_message TEXT,
                failed_stage TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                next_attempt_at TEXT,
                last_attempt_at TEXT,
                PRIMARY KEY (tenant_id, content_hash, pipeline_fingerprint)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO ingestions_v2 (
                content_hash, pipeline_fingerprint, pipeline_config_json,
                document_id, version_id,
                job_id, filename, content_type, size_bytes, page_count,
                tenant_id, acl_tags_json, created_at, content, document_status,
                status, progress_percent, error_code, current_stage, point_count,
                error_message, failed_stage, attempt_count, max_attempts,
                next_attempt_at, last_attempt_at
            )
            SELECT content_hash, pipeline_fingerprint, pipeline_config_json,
                   document_id, version_id,
                   job_id, filename, content_type, size_bytes, page_count,
                   tenant_id, acl_tags_json, created_at, content, document_status,
                   status, progress_percent, error_code, current_stage, point_count,
                   error_message, failed_stage, attempt_count, max_attempts,
                   next_attempt_at, last_attempt_at
            FROM ingestions
            """
        )
        connection.execute("DROP TABLE ingestions")
        connection.execute("ALTER TABLE ingestions_v2 RENAME TO ingestions")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestions_job_id ON ingestions(job_id)"
        )

    def _accept_sync(
        self,
        prepared: PreparedIngestion,
        idempotency_key: str | None,
    ) -> IngestionReceipt:
        identity = (
            prepared.upload.tenant_id,
            prepared.upload.content_hash,
            prepared.pipeline_fingerprint,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key is not None:
                key_row = connection.execute(
                    """
                    SELECT tenant_id, content_hash, pipeline_fingerprint
                    FROM idempotency_keys
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if key_row is not None:
                    previous_identity = (
                        key_row["tenant_id"],
                        key_row["content_hash"],
                        key_row["pipeline_fingerprint"],
                    )
                    if previous_identity != identity:
                        raise ServiceError(
                            code=ErrorCode.INGESTION_CONFLICT,
                            message="Idempotency-Key was already used for another upload",
                        )
                    return self._receipt_for_identity(connection, identity)

            existing = connection.execute(
                """
                SELECT * FROM ingestions
                WHERE tenant_id = ? AND content_hash = ? AND pipeline_fingerprint = ?
                """,
                identity,
            ).fetchone()
            if existing is not None:
                if idempotency_key is not None:
                    connection.execute(
                        """
                        INSERT INTO idempotency_keys
                            (idempotency_key, tenant_id, content_hash, pipeline_fingerprint)
                        VALUES (?, ?, ?, ?)
                        """,
                        (idempotency_key, *identity),
                    )
                return self._receipt_from_row(existing, idempotent_hit=True)

            receipt = create_ingestion_receipt(identity)
            connection.execute(
                """
                INSERT INTO ingestions (
                    content_hash, pipeline_fingerprint, pipeline_config_json,
                    document_id, version_id,
                    job_id, filename, content_type, size_bytes, page_count,
                    tenant_id, acl_tags_json, created_at, content, document_status, status, progress_percent, error_code,
                    current_stage, point_count, error_message, failed_stage,
                    attempt_count, max_attempts, next_attempt_at, last_attempt_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prepared.upload.content_hash,
                    prepared.pipeline_fingerprint,
                    _serialized_pipeline_metadata(prepared),
                    receipt.document_id,
                    receipt.version_id,
                    receipt.job_id,
                    prepared.upload.filename,
                    prepared.upload.content_type,
                    prepared.upload.size_bytes,
                    prepared.pdf.page_count,
                    prepared.upload.tenant_id,
                    json.dumps(list(prepared.upload.acl_tags), ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    prepared.content,
                    receipt.status.value,
                    JobStatus.QUEUED.value,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    3,
                    None,
                    None,
                ),
            )
            if idempotency_key is not None:
                connection.execute(
                    """
                    INSERT INTO idempotency_keys
                        (idempotency_key, tenant_id, content_hash, pipeline_fingerprint)
                    VALUES (?, ?, ?, ?)
                    """,
                    (idempotency_key, *identity),
                )
            return receipt

    def _get_job_sync(self, job_id: str) -> JobSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return self._job_from_row(
                row,
                self._stage_events_sync(connection, job_id),
            )

    def _claim_job_sync(
        self,
        job_id: str,
        stale_after_seconds: float,
    ) -> JobSnapshot | None:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ingestions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            current = self._job_from_row(
                row,
                self._stage_events_sync(connection, job_id),
            )
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
            connection.execute(
                """
                UPDATE ingestions
                SET status = ?, progress_percent = MAX(progress_percent, 1),
                    error_code = NULL, error_message = NULL,
                    next_attempt_at = NULL, last_attempt_at = ?,
                    attempt_count = attempt_count + 1
                WHERE job_id = ?
                """,
                (JobStatus.RUNNING.value, now.isoformat(), job_id),
            )
            updated = connection.execute(
                "SELECT * FROM ingestions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if updated is None:
                raise RuntimeError("claimed ingestion disappeared")
            return self._job_from_row(
                updated,
                self._stage_events_sync(connection, job_id),
            )

    def _list_recoverable_jobs_sync(
        self,
        limit: int,
        stale_after_seconds: float,
    ) -> tuple[str, ...]:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, status, attempt_count, max_attempts,
                       next_attempt_at, last_attempt_at
                FROM ingestions
                WHERE status IN (?, ?, ?)
                  AND attempt_count < max_attempts
                ORDER BY COALESCE(next_attempt_at, created_at), job_id
                LIMIT ?
                """,
                (
                    JobStatus.QUEUED.value,
                    JobStatus.FAILED.value,
                    JobStatus.RUNNING.value,
                    limit * 2,
                ),
            ).fetchall()
        recoverable: list[str] = []
        for row in rows:
            next_attempt_at = (
                datetime.fromisoformat(row["next_attempt_at"])
                if row["next_attempt_at"]
                else None
            )
            last_attempt_at = (
                datetime.fromisoformat(row["last_attempt_at"])
                if row["last_attempt_at"]
                else None
            )
            if next_attempt_at is not None and next_attempt_at > now:
                continue
            if row["status"] == JobStatus.RUNNING.value and (
                last_attempt_at is not None
                and now - last_attempt_at
                < timedelta(seconds=stale_after_seconds)
            ):
                continue
            recoverable.append(row["job_id"])
            if len(recoverable) >= limit:
                break
        return tuple(recoverable)

    def _get_staged_sync(self, job_id: str) -> PreparedIngestion | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        metadata = _pipeline_metadata_from_json(row["pipeline_config_json"])
        pipeline_config = restore_pipeline_config(metadata)
        raw_chunking = metadata.get("chunking")
        chunking = (
            ChunkingResolution.from_mapping(raw_chunking)
            if isinstance(raw_chunking, dict)
            else None
        )
        return PreparedIngestion(
            content=bytes(row["content"]),
            upload=UploadMetadata(
                filename=row["filename"],
                content_type=row["content_type"],
                size_bytes=row["size_bytes"],
                content_hash=row["content_hash"],
                tenant_id=row["tenant_id"],
                acl_tags=tuple(json.loads(row["acl_tags_json"])),
            ),
            pdf=PdfInspection(page_count=row["page_count"]),
            pipeline_fingerprint=row["pipeline_fingerprint"],
            pipeline_config=pipeline_config,
            chunking=chunking,
        )

    def _list_documents_sync(
        self,
        limit: int,
        cursor: str | None,
        tenant_id: str,
    ) -> DocumentPage:
        offset = _parse_cursor(cursor)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ingestions
                WHERE tenant_id = ?
                ORDER BY created_at ASC, document_id ASC, version_id ASC
                """,
                (tenant_id,),
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["document_id"], []).append(row)
        snapshots = sorted(
            (_snapshot_from_rows(items) for items in grouped.values()),
            key=lambda item: (item.created_at, item.document_id),
            reverse=True,
        )
        page = snapshots[offset : offset + limit]
        next_cursor = str(offset + limit) if offset + limit < len(snapshots) else None
        return DocumentPage(items=tuple(page), next_cursor=next_cursor)

    def _get_document_sync(
        self,
        document_id: str,
        tenant_id: str,
    ) -> DocumentSnapshot | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ingestions
                WHERE document_id = ? AND tenant_id = ?
                ORDER BY created_at ASC, version_id ASC
                """,
                (document_id, tenant_id),
            ).fetchall()
        return _snapshot_from_rows(rows) if rows else None

    def _delete_document_sync(self, document_id: str, tenant_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT status FROM ingestions WHERE document_id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            ).fetchall()
            if not rows:
                raise ServiceError(
                    code=ErrorCode.DOCUMENT_NOT_FOUND,
                    message="Document was not found",
                )
            if any(
                JobStatus(row["status"]) in (JobStatus.QUEUED, JobStatus.RUNNING)
                for row in rows
            ):
                raise ServiceError(
                    code=ErrorCode.DOCUMENT_BUSY,
                    message="Document has an ingestion job in progress",
                )
            connection.execute(
                "UPDATE ingestions SET document_status = ? WHERE document_id = ? AND tenant_id = ?",
                (DocumentStatus.DELETED.value, document_id, tenant_id),
            )

    def _update_job_sync(self, snapshot: JobSnapshot) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingestions
                SET status = ?, progress_percent = ?, error_code = ?,
                    current_stage = ?, point_count = ?, error_message = ?,
                    failed_stage = ?, attempt_count = ?, max_attempts = ?,
                    next_attempt_at = ?, last_attempt_at = ?
                WHERE job_id = ?
                """,
                (
                    snapshot.status.value,
                    snapshot.progress_percent,
                    snapshot.error_code,
                    snapshot.current_stage,
                    snapshot.point_count,
                    snapshot.error_message,
                    snapshot.failed_stage,
                    snapshot.attempt_count,
                    snapshot.max_attempts,
                    snapshot.next_attempt_at.isoformat()
                    if snapshot.next_attempt_at
                    else None,
                    snapshot.last_attempt_at.isoformat()
                    if snapshot.last_attempt_at
                    else None,
                    snapshot.job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown job: {snapshot.job_id}")

    def _record_stage_event_sync(self, job_id: str, event: StageEvent) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO ingestion_stage_events (
                    job_id, stage_name, status, started_at, finished_at,
                    duration_ms, inputs_json, outputs_json, decision,
                    warnings_json, error_code, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    event.name,
                    event.status.value,
                    event.started_at.isoformat(),
                    event.finished_at.isoformat() if event.finished_at else None,
                    event.duration_ms,
                    json.dumps(event.inputs or {}, sort_keys=True),
                    json.dumps(event.outputs or {}, sort_keys=True),
                    event.decision,
                    json.dumps(list(event.warnings), ensure_ascii=False),
                    event.error_code,
                    event.error_message,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stage event was not persisted")
            output_points = (event.outputs or {}).get("points")
            connection.execute(
                """
                UPDATE ingestions
                SET current_stage = ?,
                    point_count = COALESCE(?, point_count),
                    error_code = COALESCE(?, error_code),
                    error_message = COALESCE(?, error_message),
                    failed_stage = CASE WHEN ? = 'failed' THEN ? ELSE failed_stage END
                WHERE job_id = ?
                """,
                (
                    event.name,
                    int(output_points)
                    if isinstance(output_points, (int, float))
                    else None,
                    event.error_code,
                    event.error_message,
                    event.status.value,
                    event.name,
                    job_id,
                ),
            )

    @staticmethod
    def _stage_events_sync(
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[StageEvent, ...]:
        rows = connection.execute(
            """
            SELECT stage_name, status, started_at, finished_at, duration_ms,
                   inputs_json, outputs_json, decision, warnings_json,
                   error_code, error_message
            FROM ingestion_stage_events
            WHERE job_id = ?
            ORDER BY event_id
            """,
            (job_id,),
        ).fetchall()
        latest: dict[str, StageEvent] = {}
        for row in rows:
            latest[row["stage_name"]] = StageEvent(
                name=row["stage_name"],
                status=StageStatus(row["status"]),
                started_at=datetime.fromisoformat(row["started_at"]),
                finished_at=(
                    datetime.fromisoformat(row["finished_at"])
                    if row["finished_at"]
                    else None
                ),
                duration_ms=row["duration_ms"],
                inputs=json.loads(row["inputs_json"]),
                outputs=json.loads(row["outputs_json"]),
                decision=row["decision"],
                warnings=tuple(json.loads(row["warnings_json"])),
                error_code=row["error_code"],
                error_message=row["error_message"],
            )
        return tuple(latest.values())

    def _set_document_status_sync(
        self,
        document_id: str,
        version_id: str,
        status: DocumentStatus,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE ingestions
                SET document_status = ?
                WHERE document_id = ? AND version_id = ?
                """,
                (status.value, document_id, version_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown document version: {document_id}/{version_id}")

    def _receipt_for_identity(
        self,
        connection: sqlite3.Connection,
        identity: tuple[str, str, str],
    ) -> IngestionReceipt:
        row = connection.execute(
            """
            SELECT * FROM ingestions
            WHERE tenant_id = ? AND content_hash = ? AND pipeline_fingerprint = ?
            """,
            identity,
        ).fetchone()
        if row is None:
            raise RuntimeError("idempotency key points to a missing ingestion")
        return self._receipt_from_row(row, idempotent_hit=True)

    @staticmethod
    def _receipt_from_row(
        row: sqlite3.Row,
        *,
        idempotent_hit: bool = False,
    ) -> IngestionReceipt:
        return IngestionReceipt(
            document_id=row["document_id"],
            version_id=row["version_id"],
            job_id=row["job_id"],
            status=DocumentStatus(row["document_status"]),
            idempotent_hit=idempotent_hit,
        )

    @staticmethod
    def _job_from_row(
        row: sqlite3.Row,
        stages: tuple[StageEvent, ...] = (),
    ) -> JobSnapshot:
        return JobSnapshot(
            job_id=row["job_id"],
            document_id=row["document_id"],
            status=JobStatus(row["status"]),
            progress_percent=row["progress_percent"],
            error_code=row["error_code"],
            current_stage=row["current_stage"],
            stages=stages,
            page_count=row["page_count"],
            point_count=row["point_count"],
            error_message=row["error_message"],
            failed_stage=row["failed_stage"],
            attempt_count=row["attempt_count"],
            max_attempts=row["max_attempts"],
            next_attempt_at=(
                datetime.fromisoformat(row["next_attempt_at"])
                if row["next_attempt_at"]
                else None
            ),
            last_attempt_at=(
                datetime.fromisoformat(row["last_attempt_at"])
                if row["last_attempt_at"]
                else None
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=30,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection


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


def _serialized_pipeline_metadata(prepared: PreparedIngestion) -> str:
    """Serialize effective profile/configuration for the separate worker."""

    if prepared.pipeline_config is None or prepared.chunking is None:
        return "{}"
    return json.dumps(
        serialize_pipeline_metadata(prepared.pipeline_config, prepared.chunking),
        ensure_ascii=False,
        sort_keys=True,
    )


def _pipeline_metadata_from_json(value: object) -> dict[str, object]:
    """Read metadata from legacy-compatible SQLite rows without raising."""

    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _snapshot_from_rows(rows: list[sqlite3.Row]) -> DocumentSnapshot:
    """Build one public document read model from its stored versions."""

    ordered = sorted(
        rows,
        key=lambda row: (row["created_at"], row["version_id"]),
    )
    statuses = {DocumentStatus(row["document_status"]) for row in ordered}
    if DocumentStatus.ACTIVE in statuses:
        status = DocumentStatus.ACTIVE
    elif DocumentStatus.INDEXING in statuses:
        status = DocumentStatus.INDEXING
    elif DocumentStatus.FAILED in statuses:
        status = DocumentStatus.FAILED
    else:
        status = DocumentStatus.DELETED
    active_versions = [
        row for row in ordered
        if DocumentStatus(row["document_status"]) is DocumentStatus.ACTIVE
    ]
    latest = ordered[-1]
    return DocumentSnapshot(
        document_id=latest["document_id"],
        title=latest["filename"],
        content_hash=latest["content_hash"],
        active_version_id=(
            active_versions[-1]["version_id"] if active_versions else None
        ),
        status=status,
        created_at=datetime.fromisoformat(ordered[0]["created_at"]),
        available_version_ids=tuple(row["version_id"] for row in ordered),
        tenant_id=latest["tenant_id"],
    )
