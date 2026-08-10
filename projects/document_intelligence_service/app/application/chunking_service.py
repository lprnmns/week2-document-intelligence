"""Page-aware parsing and chunking use case."""

from ..domain.chunks import (
    ChildChunk,
    PageText,
    ParentSection,
    SectionMarker,
    chunk_parent_section,
    normalize_page_text,
    sectionize_pages,
)
from ..domain.errors import ErrorCode, ServiceError
from ..domain.ingestion import PipelineConfig
from .ports import PageTextExtractor


class DocumentChunkingService:
    """Orchestrate extraction, sectioning and child chunk creation."""

    def __init__(
        self,
        *,
        extractor: PageTextExtractor,
        pipeline_config: PipelineConfig,
    ) -> None:
        self._extractor = extractor
        self._pipeline_config = pipeline_config

    @property
    def chunk_size_sentences(self) -> int:
        """Return the bounded child-chunk size for observability."""

        return self._pipeline_config.chunk_size_sentences

    @property
    def chunk_overlap_sentences(self) -> int:
        """Return the deterministic overlap used by child chunking."""

        return self._pipeline_config.chunk_overlap_sentences

    @property
    def generic_parent_max_chars(self) -> int:
        """Return the configured generic parent-context character bound."""

        return self._pipeline_config.generic_parent_max_chars

    def parse_pages(self, content: bytes) -> tuple[PageText, ...]:
        """Parse the PDF while retaining page boundaries."""

        return self._extractor.extract(content)

    @staticmethod
    def normalize_pages(pages: tuple[PageText, ...]) -> tuple[PageText, ...]:
        """Normalize page whitespace and reject a text-free PDF explicitly."""

        normalized = tuple(
            PageText(page.page_number, normalize_page_text(page.text))
            for page in pages
            if normalize_page_text(page.text)
        )
        if not normalized:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_PARSE_FAILED,
                message="PDF contains no selectable text",
            )
        return normalized

    def build_from_pages(
        self,
        *,
        pages: tuple[PageText, ...],
        document_id: str,
        version_id: str,
        source: str,
        markers: tuple[SectionMarker, ...] = (),
    ) -> tuple[tuple[ParentSection, ...], tuple[ChildChunk, ...]]:
        """Build parent context and child chunks from normalized pages."""

        parents = sectionize_pages(
            pages=pages,
            document_id=document_id,
            version_id=version_id,
            source=source,
            markers=markers,
            max_parent_chars=self._pipeline_config.generic_parent_max_chars,
        )
        children = tuple(
            child
            for parent in parents
            for child in chunk_parent_section(
                parent,
                max_sentences=self._pipeline_config.chunk_size_sentences,
                overlap_sentences=self._pipeline_config.chunk_overlap_sentences,
            )
        )
        if not children:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_CHUNKING_FAILED,
                message="PDF contains no indexable text",
            )
        return parents, children

    def build_chunks(
        self,
        *,
        content: bytes,
        document_id: str,
        version_id: str,
        source: str,
        markers: tuple[SectionMarker, ...] = (),
    ) -> tuple[tuple[ParentSection, ...], tuple[ChildChunk, ...]]:
        """Return parent context and retrieval children with page metadata."""

        pages = self.parse_pages(content)
        normalized_pages = self.normalize_pages(pages)
        return self.build_from_pages(
            pages=normalized_pages,
            document_id=document_id,
            version_id=version_id,
            source=source,
            markers=markers,
        )
