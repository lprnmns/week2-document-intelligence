"""Page-aware parent and child chunk domain objects."""

from dataclasses import dataclass
import hashlib
import re

from .errors import ErrorCode, ServiceError

_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class PageText:
    """Normalized selectable text belonging to one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class SectionMarker:
    """Explicit heading rule for a known document family."""

    title: str
    marker: str


@dataclass(frozen=True, slots=True)
class ParentSection:
    """Larger source context retained around child chunks."""

    parent_id: str
    document_id: str
    version_id: str
    source: str
    title: str
    text: str
    page_start: int
    page_end: int
    page_boundaries: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ChildChunk:
    """Retrieval unit with enough metadata to restore parent evidence."""

    chunk_id: str
    parent_id: str
    document_id: str
    version_id: str
    source: str
    title: str
    text: str
    chunk_index: int
    page_start: int
    page_end: int
    token_count_estimate: int
    text_hash: str
    parent_text: str = ""


def normalize_page_text(text: str) -> str:
    """Collapse PDF whitespace while preserving readable text order."""

    return " ".join(text.split())


def sectionize_pages(
    *,
    pages: tuple[PageText, ...],
    document_id: str,
    version_id: str,
    source: str,
    markers: tuple[SectionMarker, ...] = (),
    max_parent_chars: int = 4000,
) -> tuple[ParentSection, ...]:
    """Build ordered parent sections from pages and explicit markers.

    When no markers are configured, pages are grouped into deterministic,
    bounded parent windows. We do not guess headings from arbitrary typography
    at this stage. Explicit marker profiles retain their original boundaries.
    """

    if max_parent_chars <= 0:
        raise ValueError("max_parent_chars must be greater than zero")

    non_empty_pages = tuple(
        PageText(page.page_number, normalize_page_text(page.text))
        for page in pages
        if normalize_page_text(page.text)
    )
    if not non_empty_pages:
        raise ServiceError(
            code=ErrorCode.DOCUMENT_PARSE_FAILED,
            message="PDF contains no selectable text",
        )

    joined_text = "\n".join(page.text for page in non_empty_pages)
    boundaries = _page_boundaries(non_empty_pages)
    if not markers:
        return tuple(
            _make_parent(
                document_id=document_id,
                version_id=version_id,
                source=source,
                title=source,
                text=joined_text[start:end],
                start=start,
                end=end,
                boundaries=boundaries,
                index=index,
            )
            for index, (start, end) in enumerate(
                _bounded_parent_spans(non_empty_pages, max_parent_chars)
            )
        )

    positions: list[tuple[int, SectionMarker]] = []
    for marker in markers:
        position = joined_text.find(marker.marker)
        if position < 0:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_CHUNKING_FAILED,
                message="Configured section marker was not found",
            )
        positions.append((position, marker))

    if positions != sorted(positions, key=lambda item: item[0]):
        raise ServiceError(
            code=ErrorCode.DOCUMENT_CHUNKING_FAILED,
            message="Configured section markers are out of order",
        )

    parents: list[ParentSection] = []
    for index, (start, marker) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(joined_text)
        parents.append(
            _make_parent(
                document_id=document_id,
                version_id=version_id,
                source=source,
                title=marker.title,
                text=joined_text[start:end],
                start=start,
                end=end,
                boundaries=boundaries,
                index=index,
            )
        )
    return tuple(parents)


def chunk_parent_section(
    parent: ParentSection,
    *,
    max_sentences: int = 3,
    overlap_sentences: int = 1,
) -> tuple[ChildChunk, ...]:
    """Create deterministic overlapping child chunks from one parent."""

    if max_sentences <= 0:
        raise ValueError("max_sentences must be greater than zero")
    if overlap_sentences < 0 or overlap_sentences >= max_sentences:
        raise ValueError("overlap_sentences must be smaller than max_sentences")

    sentences = _split_sentence_spans(parent.text)
    if not sentences:
        return ()

    step = max_sentences - overlap_sentences
    chunks: list[ChildChunk] = []
    start = 0
    index = 1
    while start < len(sentences):
        selected = sentences[start : start + max_sentences]
        text = " ".join(sentence for sentence, _, _ in selected)
        child_page_start, child_page_end = _page_range_for_span(
            selected[0][1],
            selected[-1][2],
            parent.page_boundaries,
            fallback=(parent.page_start, parent.page_end),
        )
        chunks.append(
            ChildChunk(
                chunk_id=f"{parent.parent_id}:child:{index:03d}",
                parent_id=parent.parent_id,
                document_id=parent.document_id,
                version_id=parent.version_id,
                source=parent.source,
                title=parent.title,
                text=text,
                chunk_index=index,
                page_start=child_page_start,
                page_end=child_page_end,
                token_count_estimate=len(text.split()),
                text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                parent_text=parent.text,
            )
        )
        if start + max_sentences >= len(sentences):
            break
        start += step
        index += 1
    return tuple(chunks)


def _split_sentences(text: str) -> list[str]:
    return [sentence for sentence, _, _ in _split_sentence_spans(text)]


def _split_sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Split normalized text while retaining offsets for page lookup."""

    cleaned = normalize_page_text(text)
    if not cleaned:
        return []

    sentences: list[tuple[str, int, int]] = []
    segment_start = 0
    for boundary in _SENTENCE_PATTERN.finditer(cleaned):
        sentence_start = segment_start
        sentence_end = boundary.start()
        sentence = cleaned[sentence_start:sentence_end].strip()
        if sentence:
            left_trim = len(cleaned[sentence_start:sentence_end]) - len(
                cleaned[sentence_start:sentence_end].lstrip()
            )
            right_trim = len(cleaned[sentence_start:sentence_end]) - len(
                cleaned[sentence_start:sentence_end].rstrip()
            )
            sentences.append(
                (
                    sentence,
                    sentence_start + left_trim,
                    sentence_end - right_trim,
                )
            )
        segment_start = boundary.end()

    sentence = cleaned[segment_start:].strip()
    if sentence:
        left_trim = len(cleaned[segment_start:]) - len(
            cleaned[segment_start:].lstrip()
        )
        right_trim = len(cleaned[segment_start:]) - len(
            cleaned[segment_start:].rstrip()
        )
        sentences.append(
            (
                sentence,
                segment_start + left_trim,
                len(cleaned) - right_trim,
            )
        )
    return sentences


def _page_boundaries(pages: tuple[PageText, ...]) -> tuple[tuple[int, int, int], ...]:
    boundaries: list[tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        end = cursor + len(page.text)
        boundaries.append((cursor, end, page.page_number))
        cursor = end + 1
    return tuple(boundaries)


def _bounded_parent_spans(
    pages: tuple[PageText, ...],
    max_parent_chars: int,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic global text spans for generic parent windows."""

    spans: list[tuple[int, int]] = []
    cursor = 0
    current_start: int | None = None
    current_end: int | None = None

    def flush() -> None:
        nonlocal current_start, current_end
        if current_start is not None and current_end is not None:
            spans.append((current_start, current_end))
        current_start = None
        current_end = None

    for page in pages:
        page_start = cursor
        page_end = page_start + len(page.text)
        cursor = page_end + 1

        if len(page.text) <= max_parent_chars:
            proposed_length = (
                len(page.text)
                if current_start is None or current_end is None
                else page_end - current_start
            )
            if (
                current_start is None
                or current_end is None
                or proposed_length > max_parent_chars
            ):
                flush()
                current_start = page_start
                current_end = page_end
            else:
                current_end = page_end
            continue

        # A single page may exceed the window. Split only at whitespace when
        # possible; this keeps generic chunking deterministic without OCR or
        # layout inference.
        flush()
        for local_start, local_end in _bounded_text_spans(
            page.text,
            max_parent_chars,
        ):
            spans.append((page_start + local_start, page_start + local_end))

    flush()
    return tuple(spans)


def _bounded_text_spans(text: str, max_chars: int) -> tuple[tuple[int, int], ...]:
    """Split one normalized text string into whitespace-bounded windows."""

    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        target = min(start + max_chars, length)
        if target < length:
            boundary = text.rfind(" ", start + 1, target + 1)
            end = boundary if boundary > start else target
        else:
            end = target
        spans.append((start, end))
        start = end
        while start < length and text[start] == " ":
            start += 1
    return tuple(spans)


def _page_for_offset(offset: int, boundaries: tuple[tuple[int, int, int], ...]) -> int:
    for start, end, page_number in boundaries:
        if start <= offset <= end:
            return page_number
    return boundaries[-1][2]


def _page_range_for_span(
    start: int,
    end: int,
    boundaries: tuple[tuple[int, int, int], ...],
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """Resolve the pages touched by a child span, with legacy fallback."""

    if not boundaries:
        return fallback
    pages = [
        page_number
        for boundary_start, boundary_end, page_number in boundaries
        if start < boundary_end and end > boundary_start
    ]
    if not pages:
        page = _page_for_offset(start, boundaries)
        return page, page
    return min(pages), max(pages)


def _normalized_with_offsets(text: str) -> tuple[str, tuple[int, ...]]:
    """Collapse whitespace and retain each output character's source offset."""

    characters: list[str] = []
    offsets: list[int] = []
    pending_space: int | None = None
    for index, character in enumerate(text):
        if character.isspace():
            if characters and pending_space is None:
                pending_space = index
            continue
        if pending_space is not None:
            characters.append(" ")
            offsets.append(pending_space)
            pending_space = None
        characters.append(character)
        offsets.append(index)
    return "".join(characters), tuple(offsets)


def _relative_page_boundaries(
    *,
    source_start: int,
    source_length: int,
    output_offsets: tuple[int, ...],
    boundaries: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Map global page spans into the normalized parent text."""

    relative: list[tuple[int, int, int]] = []
    source_end = source_start + source_length
    for boundary_start, boundary_end, page_number in boundaries:
        output_indexes = [
            output_index
            for output_index, local_offset in enumerate(output_offsets)
            if boundary_start <= source_start + local_offset < boundary_end
        ]
        if output_indexes:
            relative.append(
                (min(output_indexes), max(output_indexes) + 1, page_number)
            )
        elif boundary_start < source_end and boundary_end > source_start:
            # A section boundary can leave only whitespace from a page. Keep a
            # zero-width marker so parent ranges still retain that page.
            insertion = min(
                (index for index, offset in enumerate(output_offsets)
                 if source_start + offset >= boundary_start),
                default=len(output_offsets),
            )
            relative.append((insertion, insertion, page_number))
    return tuple(relative)


def _make_parent(
    *,
    document_id: str,
    version_id: str,
    source: str,
    title: str,
    text: str,
    start: int,
    end: int,
    boundaries: tuple[tuple[int, int, int], ...],
    index: int,
) -> ParentSection:
    parent_text, output_offsets = _normalized_with_offsets(text)
    relative_boundaries = _relative_page_boundaries(
        source_start=start,
        source_length=len(text),
        output_offsets=output_offsets,
        boundaries=boundaries,
    )
    page_start, page_end = _page_range_for_span(
        0,
        len(parent_text),
        relative_boundaries,
        fallback=(
            _page_for_offset(start, boundaries),
            _page_for_offset(max(start, end - 1), boundaries),
        ),
    )
    return ParentSection(
        parent_id=f"{document_id}:{version_id}:parent:{index:03d}",
        document_id=document_id,
        version_id=version_id,
        source=source,
        title=title,
        text=parent_text,
        page_start=page_start,
        page_end=page_end,
        page_boundaries=relative_boundaries,
    )
