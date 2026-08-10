"""Verify the 28-page Week 2 PDF structure and page-aware chunk metadata."""

from argparse import ArgumentParser
import hashlib
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..app.domain.chunks import chunk_parent_section, sectionize_pages
    from ..app.infrastructure.parsing.pdf_text import PypdfTextExtractor
    from ..app.infrastructure.parsing.section_markers import get_section_markers
else:  # pragma: no cover - package selection depends on the image layout
    _app_prefix = (
        "projects.document_intelligence_service.app"
        if __package__ and __package__.startswith("projects.")
        else "app"
    )
    _chunks = importlib.import_module(f"{_app_prefix}.domain.chunks")
    _pdf_text = importlib.import_module(
        f"{_app_prefix}.infrastructure.parsing.pdf_text"
    )
    _section_markers = importlib.import_module(
        f"{_app_prefix}.infrastructure.parsing.section_markers"
    )
    chunk_parent_section = _chunks.chunk_parent_section
    sectionize_pages = _chunks.sectionize_pages
    PypdfTextExtractor = _pdf_text.PypdfTextExtractor
    get_section_markers = _section_markers.get_section_markers


def inspect_week2_pdf(pdf_path: Path) -> dict[str, object]:
    """Extract and validate all 28 Week 2 sections without modifying the PDF."""

    content = pdf_path.read_bytes()
    pages = PypdfTextExtractor().extract(content)
    if len(pages) != 28:
        raise ValueError(f"expected 28 pages, found {len(pages)}")
    parents = sectionize_pages(
        pages=pages,
        document_id="week2_pdf_inspection",
        version_id=hashlib.sha256(content).hexdigest(),
        source=pdf_path.name,
        markers=get_section_markers("mentor_program_week2_v1"),
    )
    children = tuple(
        child
        for parent in parents
        for child in chunk_parent_section(
            parent,
            max_sentences=3,
            overlap_sentences=1,
        )
    )
    invalid_children = [
        child.chunk_id
        for child in children
        if child.page_start < 1
        or child.page_end > 28
        or child.page_start > child.page_end
    ]
    if invalid_children:
        raise ValueError(f"invalid child page ranges: {invalid_children[:3]}")
    return {
        "pdf": str(pdf_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "page_count": len(pages),
        "non_empty_page_count": sum(bool(page.text.strip()) for page in pages),
        "parent_count": len(parents),
        "child_count": len(children),
        "parent_page_ranges": [
            {
                "title": parent.title,
                "page_start": parent.page_start,
                "page_end": parent.page_end,
            }
            for parent in parents
        ],
        "child_page_ranges_valid": not invalid_children,
        "child_pages_covered": sorted(
            {
                page
                for child in children
                for page in range(child.page_start, child.page_end + 1)
            }
        ),
    }


def main() -> None:
    """Run the inspection and optionally persist its JSON report."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_week2_pdf(args.pdf)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
