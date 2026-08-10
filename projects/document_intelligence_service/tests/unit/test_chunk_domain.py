"""Unit tests for page-aware parent/child chunking."""

import pytest

from projects.document_intelligence_service.app.application.chunking_service import (
    DocumentChunkingService,
)
from projects.document_intelligence_service.app.domain.chunks import (
    PageText,
    ParentSection,
    SectionMarker,
    chunk_parent_section,
    normalize_page_text,
    sectionize_pages,
)
from projects.document_intelligence_service.app.domain.ingestion import PipelineConfig


def test_normalize_page_text_collapses_pdf_whitespace() -> None:
    assert normalize_page_text("  Python\n\n sanal   ortam ") == "Python sanal ortam"


def test_sectionize_pages_preserves_parent_titles_and_page_ranges() -> None:
    pages = (
        PageText(1, "01 Temel Bilgiler İlk cümle. İkinci cümle."),
        PageText(2, "02 RAG Akışı Üçüncü cümle. Dördüncü cümle."),
    )
    parents = sectionize_pages(
        pages=pages,
        document_id="doc_1",
        version_id="ver_1",
        source="guide.pdf",
        markers=(
            SectionMarker("Temel Bilgiler", "01 Temel Bilgiler"),
            SectionMarker("RAG Akışı", "02 RAG Akışı"),
        ),
    )

    assert [parent.title for parent in parents] == ["Temel Bilgiler", "RAG Akışı"]
    assert parents[0].page_start == 1
    assert parents[0].page_end == 1
    assert parents[1].page_start == 2
    assert parents[1].page_end == 2
    assert parents[0].parent_id.endswith("parent:000")


def test_sectionize_without_markers_keeps_one_document_parent() -> None:
    parents = sectionize_pages(
        pages=(PageText(1, "İlk sayfa."), PageText(2, "İkinci sayfa.")),
        document_id="doc_1",
        version_id="ver_1",
        source="guide.pdf",
    )

    assert len(parents) == 1
    assert parents[0].page_start == 1
    assert parents[0].page_end == 2
    assert "İkinci sayfa." in parents[0].text


def test_generic_parent_windows_are_bounded_without_heading_knowledge() -> None:
    pages = tuple(
        PageText(page_number, " ".join(f"sayfa{page_number}_kelime{i}" for i in range(12)))
        for page_number in range(1, 4)
    )

    parents = sectionize_pages(
        pages=pages,
        document_id="doc_1",
        version_id="ver_1",
        source="ordinary.pdf",
        max_parent_chars=80,
    )

    assert len(parents) > 1
    assert all(len(parent.text) <= 80 for parent in parents)
    assert all(parent.page_start <= parent.page_end for parent in parents)
    assert parents[0].parent_id.endswith("parent:000")
    assert parents[-1].parent_id.endswith(f"parent:{len(parents) - 1:03d}")


def test_chunk_parent_section_keeps_overlap_and_parent_metadata() -> None:
    parent = ParentSection(
        parent_id="doc_1:ver_1:parent:000",
        document_id="doc_1",
        version_id="ver_1",
        source="guide.pdf",
        title="RAG",
        text="Birinci bilgi. İkinci bilgi. Üçüncü bilgi. Dördüncü bilgi.",
        page_start=2,
        page_end=3,
    )

    chunks = chunk_parent_section(parent, max_sentences=2, overlap_sentences=1)

    assert len(chunks) == 3
    assert chunks[0].text == "Birinci bilgi. İkinci bilgi."
    assert chunks[1].text == "İkinci bilgi. Üçüncü bilgi."
    assert chunks[0].parent_id == parent.parent_id
    assert chunks[0].parent_text == parent.text
    assert chunks[0].page_start == 2
    assert chunks[0].page_end == 3
    assert len(chunks[0].text_hash) == 64


def test_chunks_report_their_own_page_ranges() -> None:
    parents = sectionize_pages(
        pages=(
            PageText(1, "İlk sayfadaki bilgi."),
            PageText(2, "İkinci sayfadaki bilgi."),
        ),
        document_id="doc_1",
        version_id="ver_1",
        source="guide.pdf",
    )

    chunks = chunk_parent_section(
        parents[0],
        max_sentences=1,
        overlap_sentences=0,
    )

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [
        (1, 1),
        (2, 2),
    ]


def test_chunk_crossing_a_page_break_reports_both_pages() -> None:
    parents = sectionize_pages(
        pages=(
            PageText(1, "Bu cümle sayfa"),
            PageText(2, " sınırında devam eder."),
        ),
        document_id="doc_1",
        version_id="ver_1",
        source="guide.pdf",
    )

    chunks = chunk_parent_section(
        parents[0],
        max_sentences=1,
        overlap_sentences=0,
    )

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_chunk_settings_are_validated() -> None:
    parent = ParentSection("p", "d", "v", "s", "t", "One sentence.", 1, 1)

    with pytest.raises(ValueError):
        chunk_parent_section(parent, max_sentences=0)
    with pytest.raises(ValueError):
        chunk_parent_section(parent, max_sentences=2, overlap_sentences=2)


class FakeExtractor:
    """Deterministic page extractor for application service tests."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        assert content == b"pdf"
        return (
            PageText(1, "01 Başlık Birinci. İkinci."),
            PageText(2, "Üçüncü. Dördüncü."),
        )


def test_chunking_service_connects_extractor_and_domain_policy() -> None:
    service = DocumentChunkingService(
        extractor=FakeExtractor(),
        pipeline_config=PipelineConfig(
            chunk_size_sentences=2,
            chunk_overlap_sentences=1,
        ),
    )

    parents, children = service.build_chunks(
        content=b"pdf",
        document_id="doc_1",
        version_id="ver_1",
        source="guide.pdf",
        markers=(SectionMarker("Başlık", "01 Başlık"),),
    )

    assert len(parents) == 1
    assert len(children) == 3
    assert all(child.title == "Başlık" for child in children)
