"""Tests for the stage -> verify -> activate ingestion worker."""

import asyncio
from io import BytesIO
import time
from typing import Sequence

import pytest
from pypdf import PdfWriter

from projects.document_intelligence_service.app.domain.entities import JobStatus
from projects.document_intelligence_service.app.application.chunking_service import (
    DocumentChunkingService,
)
from projects.document_intelligence_service.app.application.ingestion_service import (
    IngestionPreparationService,
)
from projects.document_intelligence_service.app.application.ingestion_worker import (
    IngestionWorker,
)
from projects.document_intelligence_service.app.domain.chunks import PageText
from projects.document_intelligence_service.app.domain.ingestion import (
    IngestionLimits,
    PipelineConfig,
)
from projects.document_intelligence_service.app.domain.vectors import SparseVector
from projects.document_intelligence_service.app.infrastructure.parsing.pdf_inspector import (
    PypdfInspector,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.chunk_store import (
    QdrantChunkStore,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.schema import (
    QdrantSchema,
)
from projects.document_intelligence_service.app.infrastructure.storage.in_memory_registry import (
    InMemoryIngestionRegistry,
)
from qdrant_client import QdrantClient


def make_pdf() -> bytes:
    """Create a structurally valid PDF for the preparation step."""

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class FakeExtractor:
    """Return stable text while avoiding a model or OCR dependency in tests."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        assert content
        return (
            PageText(1, "RAG sistemi kanıt arar. Qdrant point saklar. Model cevap yazar."),
        )


class EmptyExtractor:
    """Return no selectable text so the domain parse guard is exercised."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        assert content
        return ()


class FakeDenseEmbedder:
    """Small deterministic dense encoder for the worker integration test."""

    dimension = 2

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


class FakeSparseEmbedder:
    """Small deterministic sparse encoder for the worker integration test."""

    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        return tuple(
            SparseVector(indices=(1, 2), values=(1.0, 0.5)) for _ in texts
        )


class SlowDenseEmbedder(FakeDenseEmbedder):
    """Make event-loop blocking visible without loading a real model."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        time.sleep(0.08)
        return super().embed_documents(texts)


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
def test_worker_stages_verifies_and_activates_a_version() -> None:
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(
            chunk_size_sentences=2,
            chunk_overlap_sentences=1,
        ),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=make_pdf(),
        filename="guide.pdf",
        content_type="application/pdf",
        tenant_id="tenant-a",
    )
    registry = InMemoryIngestionRegistry()
    receipt = asyncio.run(registry.accept(prepared, "worker-1"))
    store = QdrantChunkStore(
        QdrantClient(":memory:"),
        QdrantSchema(collection_name="worker_test", dense_size=2),
    )
    worker = IngestionWorker(
        registry=registry,
        chunker=DocumentChunkingService(
            extractor=FakeExtractor(),
            pipeline_config=PipelineConfig(
                chunk_size_sentences=2,
                chunk_overlap_sentences=1,
            ),
        ),
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    snapshot = asyncio.run(worker.run_job(receipt.job_id))

    assert snapshot.status.value == "succeeded"
    assert snapshot.progress_percent == 100
    assert snapshot.error_code is None
    assert snapshot.current_stage == "complete"
    assert [stage.name for stage in snapshot.stages] == [
        "validate",
        "inspect",
        "parse",
        "normalize",
        "chunk",
        "embed_dense",
        "embed_sparse",
        "stage_qdrant",
        "verify",
        "activate",
        "complete",
    ]
    validate_stage = snapshot.stages[0]
    assert validate_stage.status.value == "succeeded"
    assert validate_stage.inputs == {"bytes": prepared.upload.size_bytes, "pages": 1}
    assert validate_stage.outputs == {"bytes": prepared.upload.size_bytes, "pages": 1}
    assert validate_stage.duration_ms is not None
    chunk_stage = next(stage for stage in snapshot.stages if stage.name == "chunk")
    assert chunk_stage.decision == "generic_fallback"
    assert chunk_stage.outputs is not None
    assert chunk_stage.outputs["requested_profile"] == "auto"
    assert chunk_stage.outputs["resolved_profile"] == "generic_v1"
    assert chunk_stage.outputs["strategy"] == "generic_fallback"
    assert snapshot.point_count == 2
    completed_receipt = asyncio.run(registry.accept(prepared, "worker-1"))
    assert completed_receipt.status.value == "active"
    assert registry.active_version_ids(
        (completed_receipt.document_id,),
        tenant_id="tenant-a",
    ) == (completed_receipt.version_id,)
    assert store.client.count(store.collection_name, exact=True).count == 2


def test_worker_marks_empty_pdf_text_as_failed_without_indexing() -> None:
    preparation = IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(),
        pdf_inspector=PypdfInspector(),
    )
    prepared = preparation.prepare(
        content=make_pdf(),
        filename="empty.pdf",
        content_type="application/pdf",
    )
    registry = InMemoryIngestionRegistry()
    receipt = asyncio.run(registry.accept(prepared, "worker-empty"))
    store = QdrantChunkStore(
        QdrantClient(":memory:"),
        QdrantSchema(collection_name="worker_empty_test", dense_size=2),
    )
    worker = IngestionWorker(
        registry=registry,
        chunker=DocumentChunkingService(
            extractor=EmptyExtractor(),
            pipeline_config=PipelineConfig(),
        ),
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        vector_store=store,
    )

    snapshot = asyncio.run(worker.run_job(receipt.job_id))

    assert snapshot.status.value == "failed"
    assert snapshot.error_code == "DOCUMENT_PARSE_FAILED"
    assert snapshot.current_stage == "normalize"
    assert snapshot.failed_stage == "normalize"
    assert snapshot.stages[-1].status.value == "failed"
    assert snapshot.stages[-1].inputs == {"pages": 0}
    assert snapshot.stages[-1].error_message == "PDF contains no selectable text"
    failed_receipt = asyncio.run(registry.accept(prepared, "worker-empty"))
    assert failed_receipt.status.value == "failed"
    assert not store.client.collection_exists("worker_empty_test")


def test_worker_offloads_sync_stages_so_event_loop_can_progress() -> None:
    """A slow synchronous embedder must not starve async job polling."""

    async def scenario() -> int:
        preparation = IngestionPreparationService(
            limits=IngestionLimits(),
            pipeline_config=PipelineConfig(),
            pdf_inspector=PypdfInspector(),
        )
        prepared = preparation.prepare(
            content=make_pdf(),
            filename="responsive.pdf",
            content_type="application/pdf",
        )
        registry = InMemoryIngestionRegistry()
        receipt = await registry.accept(prepared, "responsive-1")
        store = QdrantChunkStore(
            QdrantClient(":memory:"),
            QdrantSchema(collection_name="worker_responsive_test", dense_size=2),
        )
        worker = IngestionWorker(
            registry=registry,
            chunker=DocumentChunkingService(
                extractor=FakeExtractor(),
                pipeline_config=PipelineConfig(),
            ),
            dense_embedder=SlowDenseEmbedder(),
            sparse_embedder=FakeSparseEmbedder(),
            vector_store=store,
        )

        task = asyncio.create_task(worker.run_job(receipt.job_id))
        polling_ticks = 0
        while not task.done():
            polling_ticks += 1
            await asyncio.sleep(0.01)
        snapshot = await task

        assert snapshot.status is JobStatus.SUCCEEDED
        return polling_ticks

    assert asyncio.run(scenario()) >= 4
