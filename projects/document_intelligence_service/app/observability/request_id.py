"""Request ID middleware for cross-layer correlation."""

from contextvars import ContextVar
from contextlib import contextmanager
import re
from uuid import uuid4
from collections.abc import Iterator

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID = ContextVar("request_id", default="")
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
REQUEST_ID_HEADER = "X-Request-ID"


def new_request_id() -> str:
    """Create a short, non-sensitive request correlation ID."""

    return f"req_{uuid4().hex}"


def get_request_id() -> str:
    """Return the current request ID for logs and error responses."""

    return _REQUEST_ID.get()


@contextmanager
def request_id_context(request_id: str) -> Iterator[None]:
    """Temporarily bind a known correlation ID to a background task."""

    token = _REQUEST_ID.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID.reset(token)


def _select_request_id(raw_value: str | None) -> str:
    if raw_value is not None and _VALID_REQUEST_ID.fullmatch(raw_value):
        return raw_value
    return new_request_id()


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a bounded request ID to context and every HTTP response."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process one request and guarantee the response header."""

        request_id = _select_request_id(request.headers.get(REQUEST_ID_HEADER))
        token = _REQUEST_ID.set(request_id)
        try:
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            _REQUEST_ID.reset(token)
