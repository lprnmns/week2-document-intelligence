"""pypdf-backed structural PDF inspection."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ...domain.errors import ErrorCode, ServiceError
from ...domain.ingestion import PdfInspection


class PypdfInspector:
    """Inspect page count without extracting or embedding document text yet."""

    def inspect(self, content: bytes, max_pages: int) -> PdfInspection:
        """Read page metadata and enforce the configured page limit."""

        try:
            page_count = len(PdfReader(BytesIO(content), strict=False).pages)
        except (PdfReadError, OSError, ValueError) as error:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_PARSE_FAILED,
                message="PDF structure could not be parsed",
            ) from error

        if page_count == 0 or page_count > max_pages:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_PARSE_FAILED,
                message="PDF page count is outside the configured limit",
            )
        return PdfInspection(page_count=page_count)
