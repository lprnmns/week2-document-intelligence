"""pypdf-backed page-preserving text extraction."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ...domain.chunks import PageText, normalize_page_text
from ...domain.errors import ErrorCode, ServiceError


class PypdfTextExtractor:
    """Extract selectable text one page at a time."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        """Return non-empty normalized pages without losing page numbers."""

        try:
            reader = PdfReader(BytesIO(content), strict=False)
            pages = tuple(
                PageText(page_number=index, text=normalize_page_text(page.extract_text() or ""))
                for index, page in enumerate(reader.pages, start=1)
            )
        except (PdfReadError, OSError, ValueError) as error:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_PARSE_FAILED,
                message="PDF text could not be extracted",
            ) from error

        return tuple(page for page in pages if page.text)
