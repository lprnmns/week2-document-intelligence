"""Unit tests for Day 2 ingestion identity and validation."""

import asyncio
from datetime import datetime, timezone
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from projects.document_intelligence_service.app.domain.entities import (
    DocumentStatus,
    JobStatus,
    StageStatus,
)
from projects.document_intelligence_service.app.application.ingestion_service import (
    IngestionPreparationService,
)
from projects.document_intelligence_service.app.domain.errors import (
    ErrorCode,
    ServiceError,
)
from projects.document_intelligence_service.app.domain.ingestion import (
    IngestionLimits,
    JobSnapshot,
    PipelineConfig,
    StageEvent,
    compute_content_hash,
    compute_pipeline_fingerprint,
    validate_upload_metadata,
)
from projects.document_intelligence_service.app.infrastructure.parsing.pdf_inspector import (
    PypdfInspector,
)
from projects.document_intelligence_service.app.infrastructure.storage.in_memory_registry import (
    InMemoryIngestionRegistry,
)
from projects.document_intelligence_service.app.infrastructure.storage.sqlite_registry import (
    SqliteIngestionRegistry,
)


def make_pdf(page_count: int = 1) -> bytes:
    """Create a small valid PDF fixture in memory."""

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_content_hash_is_stable_and_changes_with_bytes() -> None:
    assert compute_content_hash(b"same") == compute_content_hash(b"same")
    assert compute_content_hash(b"same") != compute_content_hash(b"changed")


def test_pipeline_fingerprint_changes_when_chunker_changes() -> None:
    base = PipelineConfig()
    changed = PipelineConfig(chunker="fixed_token_v2")

    assert compute_pipeline_fingerprint(base) == compute_pipeline_fingerprint(base)
    assert compute_pipeline_fingerprint(base) != compute_pipeline_fingerprint(changed)


def test_pipeline_fingerprint_changes_when_section_profile_changes() -> None:
    base = PipelineConfig(section_marker_profile="none")
    mentor = PipelineConfig(section_marker_profile="mentor_program_v1")

    assert compute_pipeline_fingerprint(base) != compute_pipeline_fingerprint(mentor)


def test_pipeline_fingerprint_does_not_reindex_for_query_reranker_change() -> None:
    base = PipelineConfig(reranker_model="reranker-v1")
    changed = PipelineConfig(reranker_model="reranker-v2")

    assert compute_pipeline_fingerprint(base) == compute_pipeline_fingerprint(changed)


def test_upload_metadata_validates_pdf_and_sanitizes_filename() -> None:
    metadata = validate_upload_metadata(
        content=make_pdf(),
        filename="/tmp\\private\\report.pdf",
        content_type="application/pdf; charset=binary",
        limits=IngestionLimits(),
    )

    assert metadata.filename == "report.pdf"
    assert metadata.content_type == "application/pdf"
    assert metadata.size_bytes > 0
    assert len(metadata.content_hash) == 64


@pytest.mark.parametrize(
    ("content", "filename", "content_type", "expected_code"),
    [
        (b"%PDF-1.7", "report.pdf", "text/plain", ErrorCode.UNSUPPORTED_MEDIA_TYPE),
        (b"not a pdf", "report.pdf", "application/pdf", ErrorCode.DOCUMENT_PARSE_FAILED),
        (b"%PDF-1.7", "report.txt", "application/pdf", ErrorCode.UNSUPPORTED_MEDIA_TYPE),
    ],
)
def test_upload_metadata_rejects_unsafe_input(
    content: bytes,
    filename: str,
    content_type: str,
    expected_code: ErrorCode,
) -> None:
    with pytest.raises(ServiceError) as raised:
        validate_upload_metadata(
            content=content,
            filename=filename,
            content_type=content_type,
            limits=IngestionLimits(),
        )

    assert raised.value.code is expected_code


def test_upload_size_limit_is_checked_before_pdf_parsing() -> None:
    with pytest.raises(ServiceError) as raised:
        validate_upload_metadata(
            content=b"%PDF-1.7" + b"x" * 20,
            filename="report.pdf",
            content_type="application/pdf",
            limits=IngestionLimits(max_upload_bytes=10),
        )

    assert raised.value.code is ErrorCode.UPLOAD_TOO_LARGE


def test_pypdf_inspector_returns_page_count_and_enforces_limit() -> None:
    inspector = PypdfInspector()

    assert inspector.inspect(make_pdf(page_count=2), max_pages=2).page_count == 2
    with pytest.raises(ServiceError) as raised:
        inspector.inspect(make_pdf(page_count=2), max_pages=1)

    assert raised.value.code is ErrorCode.DOCUMENT_PARSE_FAILED


def test_preparation_service_combines_upload_pdf_and_pipeline_identity() -> None:
    service = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )

    prepared = service.prepare(
        content=make_pdf(page_count=2),
        filename="guide.pdf",
        content_type="application/pdf",
    )

    assert prepared.pdf.page_count == 2
    assert len(prepared.upload.content_hash) == 64
    assert len(prepared.pipeline_fingerprint) == 64


def test_development_registry_keeps_staged_bytes_for_future_worker() -> None:
    content = make_pdf()
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=content,
        filename="guide.pdf",
        content_type="application/pdf",
    )
    registry = InMemoryIngestionRegistry()

    receipt = asyncio.run(registry.accept(prepared, "stage-1"))
    staged = asyncio.run(registry.get_staged_content(receipt.job_id))

    assert staged == content


def test_in_memory_registry_claims_once_and_recovers_stale_job() -> None:
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=make_pdf(),
        filename="recoverable.pdf",
        content_type="application/pdf",
    )
    registry = InMemoryIngestionRegistry()
    receipt = asyncio.run(registry.accept(prepared, "recover-1"))

    claimed = asyncio.run(registry.claim_job(receipt.job_id))
    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING
    assert claimed.attempt_count == 1
    assert asyncio.run(registry.claim_job(receipt.job_id)) is None

    stale = replace(
        claimed,
        last_attempt_at=datetime.fromtimestamp(0, tz=timezone.utc),
    )
    asyncio.run(registry.update_job(stale))
    assert asyncio.run(registry.list_recoverable_jobs(stale_after_seconds=1)) == (
        receipt.job_id,
    )
    recovered = asyncio.run(registry.claim_job(receipt.job_id, stale_after_seconds=1))
    assert recovered is not None
    assert recovered.attempt_count == 2


def test_same_pdf_can_be_indexed_by_two_tenants_without_identity_collision() -> None:
    content = make_pdf()
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    tenant_a = preparation.prepare(
        content=content,
        filename="shared.pdf",
        content_type="application/pdf",
        tenant_id="tenant_a",
        acl_tags=("private",),
    )
    tenant_b = preparation.prepare(
        content=content,
        filename="shared.pdf",
        content_type="application/pdf",
        tenant_id="tenant_b",
        acl_tags=("private",),
    )
    registry = InMemoryIngestionRegistry()

    receipt_a = asyncio.run(registry.accept(tenant_a, None))
    receipt_b = asyncio.run(registry.accept(tenant_b, None))

    assert receipt_a.document_id != receipt_b.document_id
    assert receipt_a.version_id != receipt_b.version_id
    assert receipt_a.job_id != receipt_b.job_id


def test_duplicate_content_and_pipeline_returns_an_idempotent_hit() -> None:
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=make_pdf(),
        filename="duplicate.pdf",
        content_type="application/pdf",
    )
    registry = InMemoryIngestionRegistry()

    first = asyncio.run(registry.accept(prepared, None))
    second = asyncio.run(registry.accept(prepared, None))

    assert first.idempotent_hit is False
    assert second.idempotent_hit is True
    assert (first.document_id, first.version_id, first.job_id) == (
        second.document_id,
        second.version_id,
        second.job_id,
    )


def test_sqlite_registry_survives_a_new_registry_instance(tmp_path: Path) -> None:
    content = make_pdf()
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=content,
        filename="guide.pdf",
        content_type="application/pdf",
    )
    database_path = tmp_path / "state" / "ingestions.sqlite3"

    first_registry = SqliteIngestionRegistry(database_path)
    first_receipt = asyncio.run(first_registry.accept(prepared, "durable-1"))
    queued = asyncio.run(first_registry.get_job(first_receipt.job_id))
    asyncio.run(
        first_registry.update_job(
            JobSnapshot(
                job_id=first_receipt.job_id,
                document_id=first_receipt.document_id,
                status=JobStatus.RUNNING,
                progress_percent=35,
                error_code=None,
            )
        )
    )
    started_at = datetime.now(timezone.utc)
    asyncio.run(
        first_registry.record_stage_event(
            first_receipt.job_id,
            StageEvent(
                name="inspect",
                status=StageStatus.SUCCEEDED,
                started_at=started_at,
                finished_at=started_at,
                duration_ms=1.5,
                inputs={"bytes": prepared.upload.size_bytes},
                outputs={"pages": prepared.pdf.page_count},
                decision="accepted",
            ),
        )
    )

    restarted_registry = SqliteIngestionRegistry(database_path)
    restored = asyncio.run(restarted_registry.get_job(first_receipt.job_id))
    restored_content = asyncio.run(
        restarted_registry.get_staged_content(first_receipt.job_id)
    )
    restored_prepared = asyncio.run(
        restarted_registry.get_staged_ingestion(first_receipt.job_id)
    )
    retried = asyncio.run(restarted_registry.accept(prepared, "durable-1"))

    assert queued is not None
    assert queued.status is JobStatus.QUEUED
    assert restored is not None
    assert restored.status is JobStatus.RUNNING
    assert restored.progress_percent == 35
    assert restored.current_stage == "inspect"
    assert restored.stages == (
        StageEvent(
            name="inspect",
            status=StageStatus.SUCCEEDED,
            started_at=started_at,
            finished_at=started_at,
            duration_ms=1.5,
            inputs={"bytes": prepared.upload.size_bytes},
            outputs={"pages": prepared.pdf.page_count},
            decision="accepted",
        ),
    )
    assert restored_content == content
    assert restored_prepared is not None
    assert restored_prepared.pipeline_config is not None
    assert restored_prepared.pipeline_config.section_marker_profile == "generic_v1"
    assert restored_prepared.chunking is not None
    assert restored_prepared.chunking.resolved_profile == "generic_v1"
    assert retried.document_id == first_receipt.document_id
    assert retried.version_id == first_receipt.version_id
    assert retried.job_id == first_receipt.job_id
    assert first_receipt.idempotent_hit is False
    assert retried.idempotent_hit is True


def test_sqlite_registry_lists_details_and_deletes_completed_document(
    tmp_path: Path,
) -> None:
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=make_pdf(),
        filename="catalog.pdf",
        content_type="application/pdf",
    )
    registry = SqliteIngestionRegistry(tmp_path / "catalog.sqlite3")
    receipt = asyncio.run(registry.accept(prepared, "catalog-1"))
    queued = asyncio.run(registry.get_job(receipt.job_id))
    assert queued is not None
    asyncio.run(
        registry.update_job(
            replace(queued, status=JobStatus.SUCCEEDED, progress_percent=100)
        )
    )
    asyncio.run(
        registry.set_document_status(
            document_id=receipt.document_id,
            version_id=receipt.version_id,
            status=DocumentStatus.ACTIVE,
        )
    )

    page = asyncio.run(registry.list_documents(limit=10, cursor=None))
    assert len(page.items) == 1
    assert page.items[0].status is DocumentStatus.ACTIVE
    assert page.items[0].active_version_id == receipt.version_id
    assert page.items[0].available_version_ids == (receipt.version_id,)
    detail = asyncio.run(registry.get_document(receipt.document_id))
    assert detail == page.items[0]

    asyncio.run(registry.delete_document(receipt.document_id))
    deleted = asyncio.run(registry.get_document(receipt.document_id))
    assert deleted is not None
    assert deleted.status is DocumentStatus.DELETED
    assert deleted.active_version_id is None
