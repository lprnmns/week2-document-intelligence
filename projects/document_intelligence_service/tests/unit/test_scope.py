"""Unit tests for retrieval request tenant/ACL scope resolution."""

import pytest

from projects.document_intelligence_service.app.api.v1.scope import (
    resolve_request_scope,
)
from projects.document_intelligence_service.app.domain.errors import ErrorCode, ServiceError


def test_scope_prefers_matching_transport_headers() -> None:
    scope = resolve_request_scope(
        body_tenant_id=" tenant-a ",
        header_tenant_id="tenant-a",
        body_acl_tags=("public", "finance"),
        header_acl_tags="finance, public",
    )

    assert scope.tenant_id == "tenant-a"
    assert scope.acl_tags == ("finance", "public")


def test_scope_defaults_to_public_default_tenant() -> None:
    scope = resolve_request_scope(
        body_tenant_id=None,
        header_tenant_id=None,
        body_acl_tags=(),
        header_acl_tags=None,
    )

    assert scope.tenant_id == "default"
    assert scope.acl_tags == ("public",)


@pytest.mark.parametrize(
    ("body_tenant_id", "header_tenant_id", "body_acl_tags", "header_acl_tags"),
    (
        ("tenant-a", "tenant-b", (), None),
        (None, None, ("finance",), "public"),
    ),
)
def test_scope_rejects_conflicting_body_and_header_values(
    body_tenant_id: str | None,
    header_tenant_id: str | None,
    body_acl_tags: tuple[str, ...],
    header_acl_tags: str | None,
) -> None:
    with pytest.raises(ServiceError) as raised:
        resolve_request_scope(
            body_tenant_id=body_tenant_id,
            header_tenant_id=header_tenant_id,
            body_acl_tags=body_acl_tags,
            header_acl_tags=header_acl_tags,
        )

    assert raised.value.code is ErrorCode.INVALID_REQUEST
