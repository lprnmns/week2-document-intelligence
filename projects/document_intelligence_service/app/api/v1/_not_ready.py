"""Explicit scaffold behavior until application use cases are wired."""

from typing import NoReturn

from ...domain.errors import ErrorCode, ServiceError


def feature_not_ready(feature: str) -> NoReturn:
    """Fail transparently instead of returning fabricated business data."""

    raise ServiceError(
        code=ErrorCode.FEATURE_NOT_READY,
        message=f"{feature} workflow is not wired yet",
    )
