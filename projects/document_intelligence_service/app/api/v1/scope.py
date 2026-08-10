"""Request-scope resolution shared by retrieval-facing API routes."""

from collections.abc import Sequence
from dataclasses import dataclass

from ...domain.errors import ErrorCode, ServiceError
from ...domain.ingestion import normalize_acl_tags, normalize_tenant_id


@dataclass(frozen=True, slots=True)
class RequestScope:
    """Canonical tenant and ACL scope passed to application services."""

    tenant_id: str
    acl_tags: tuple[str, ...]


def resolve_request_scope(
    *,
    body_tenant_id: str | None,
    header_tenant_id: str | None,
    body_acl_tags: Sequence[str],
    header_acl_tags: str | None,
) -> RequestScope:
    """Resolve body/header scope and reject ambiguous authorization inputs.

    Headers are the transport-level canonical scope. Body fields remain
    supported for backwards-compatible clients, but supplying two different
    values is rejected instead of silently choosing one.
    """

    normalized_body_tenant = (
        normalize_tenant_id(body_tenant_id)
        if body_tenant_id is not None
        else None
    )
    normalized_header_tenant = (
        normalize_tenant_id(header_tenant_id)
        if header_tenant_id is not None
        else None
    )
    if (
        normalized_body_tenant is not None
        and normalized_header_tenant is not None
        and normalized_body_tenant != normalized_header_tenant
    ):
        raise ServiceError(
            code=ErrorCode.INVALID_REQUEST,
            message="Tenant ID in header and body do not match",
        )

    body_acl = (
        normalize_acl_tags(tuple(body_acl_tags))
        if body_acl_tags
        else None
    )
    header_acl = (
        normalize_acl_tags(
            tuple(
                tag.strip()
                for tag in header_acl_tags.split(",")
                if tag.strip()
            )
        )
        if header_acl_tags is not None
        else None
    )
    if (
        body_acl is not None
        and header_acl is not None
        and frozenset(body_acl) != frozenset(header_acl)
    ):
        raise ServiceError(
            code=ErrorCode.INVALID_REQUEST,
            message="ACL tags in header and body do not match",
        )

    return RequestScope(
        tenant_id=normalized_header_tenant
        or normalized_body_tenant
        or "default",
        acl_tags=header_acl or body_acl or ("public",),
    )
