"""Regression tests for automatic and explicit ingestion chunking profiles."""

import asyncio
from io import BytesIO

import pytest
from pypdf import PdfWriter

from projects.document_intelligence_service.app.application.chunking_service import (
    DocumentChunkingService,
)
from projects.document_intelligence_service.app.application.ingestion_service import (
    IngestionPreparationService,
)
from projects.document_intelligence_service.app.application.ports import (
    PageTextExtractor,
)
from projects.document_intelligence_service.app.domain.chunks import PageText
from projects.document_intelligence_service.app.domain.errors import (
    ErrorCode,
    ServiceError,
)
from projects.document_intelligence_service.app.domain.ingestion import (
    IngestionLimits,
    PipelineConfig,
    compute_pipeline_fingerprint,
)
from projects.document_intelligence_service.app.infrastructure.parsing.pdf_inspector import (
    PypdfInspector,
)
from projects.document_intelligence_service.app.infrastructure.parsing.section_markers import (
    MENTOR_PROGRAM_V1_MARKERS,
    KnownSectionMarkerProfileResolver,
)
from projects.document_intelligence_service.app.infrastructure.storage.in_memory_registry import (
    InMemoryIngestionRegistry,
)


def make_pdf() -> bytes:
    """Create a structurally valid PDF for preparation tests."""

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class ArbitraryTextExtractor:
    """Return ordinary PDF-like text without mentor-specific headings."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        assert content
        return (
            PageText(
                1,
                "35K SAY tıp tercih raporu. Üniversite ve bölüm seçenekleri karşılaştırılır.",
            ),
        )


class MentorTextExtractor:
    """Return the complete Week-1 marker contract in stable order."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        assert content
        return (
            PageText(
                1,
                " ".join(
                    f"{marker.marker}. İçerik devam eder."
                    for marker in MENTOR_PROGRAM_V1_MARKERS
                ),
            ),
        )


def preparation(
    *,
    profile: str,
    extractor: PageTextExtractor | None = None,
) -> IngestionPreparationService:
    """Build the preparation service with an optional deterministic detector."""

    profile_resolver = (
        KnownSectionMarkerProfileResolver(extractor)
        if extractor is not None
        else None
    )
    return IngestionPreparationService(
        limits=IngestionLimits(),
        pipeline_config=PipelineConfig(section_marker_profile=profile),
        pdf_inspector=PypdfInspector(),
        profile_resolver=profile_resolver,
    )


def test_auto_falls_back_to_generic_for_an_arbitrary_valid_pdf() -> None:
    prepared = preparation(
        profile="auto",
        extractor=ArbitraryTextExtractor(),
    ).prepare(
        content=make_pdf(),
        filename="medical.pdf",
        content_type="application/pdf",
    )

    assert prepared.chunking is not None
    assert prepared.chunking.requested_profile == "auto"
    assert prepared.chunking.resolved_profile == "generic_v1"
    assert prepared.chunking.confidence == "low"
    assert prepared.pipeline_config is not None
    assert prepared.pipeline_config.section_marker_profile == "generic_v1"
    assert prepared.pipeline_fingerprint == compute_pipeline_fingerprint(
        prepared.pipeline_config
    )


def test_auto_detects_a_complete_known_structure_without_guessing_partial_structure() -> None:
    prepared = preparation(
        profile="auto",
        extractor=MentorTextExtractor(),
    ).prepare(
        content=make_pdf(),
        filename="mentor.pdf",
        content_type="application/pdf",
    )

    assert prepared.chunking is not None
    assert prepared.chunking.resolved_profile == "mentor_program_v1"
    assert prepared.chunking.confidence == "high"
    assert prepared.chunking.fallback_reason is None


def test_missing_explicit_mentor_marker_is_chunking_error_not_parse_error() -> None:
    chunker = DocumentChunkingService(
        extractor=ArbitraryTextExtractor(),
        pipeline_config=PipelineConfig(
            section_marker_profile="mentor_program_v1",
        ),
    )

    with pytest.raises(ServiceError) as raised:
        chunker.build_from_pages(
            pages=(PageText(1, "ordinary text without the mentor contract."),),
            document_id="doc-1",
            version_id="ver-1",
            source="ordinary.pdf",
            markers=MENTOR_PROGRAM_V1_MARKERS,
        )

    assert raised.value.code is ErrorCode.DOCUMENT_CHUNKING_FAILED


def test_auto_and_explicit_generic_share_effective_identity() -> None:
    content = make_pdf()
    auto = preparation(
        profile="auto",
        extractor=ArbitraryTextExtractor(),
    ).prepare(content=content, filename="same.pdf", content_type="application/pdf")
    generic = preparation(profile="generic_v1").prepare(
        content=content,
        filename="same.pdf",
        content_type="application/pdf",
    )
    registry = InMemoryIngestionRegistry()

    first = asyncio.run(registry.accept(auto, None))
    second = asyncio.run(registry.accept(generic, None))

    assert auto.pipeline_fingerprint == generic.pipeline_fingerprint
    assert second.idempotent_hit is True
    assert second.version_id == first.version_id


def test_same_pdf_with_different_effective_profile_creates_a_new_version() -> None:
    content = make_pdf()
    generic = preparation(profile="generic_v1").prepare(
        content=content,
        filename="same.pdf",
        content_type="application/pdf",
    )
    mentor = preparation(profile="mentor_program_v1").prepare(
        content=content,
        filename="same.pdf",
        content_type="application/pdf",
    )
    registry = InMemoryIngestionRegistry()

    first = asyncio.run(registry.accept(generic, None))
    second = asyncio.run(registry.accept(mentor, None))

    assert generic.pipeline_fingerprint != mentor.pipeline_fingerprint
    assert second.idempotent_hit is False
    assert second.version_id != first.version_id


def test_frozen_mentor_pipeline_fingerprint_is_unchanged() -> None:
    assert compute_pipeline_fingerprint(
        PipelineConfig(section_marker_profile="mentor_program_v1")
    ) == "132e52a3e8358e66906a7dd9bcfd0c8b57aa228dd3102e9b3d8f39ccfb4c41a4"
