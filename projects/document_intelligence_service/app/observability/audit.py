"""Privacy-safe document lifecycle audit events."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping

AuditValue = str | int | float | bool | None


def emit_audit(
    *,
    action: str,
    result: str,
    document_id: str,
    version_id: str | None = None,
    tenant_id: str | None = None,
    job_id: str | None = None,
    metadata: Mapping[str, AuditValue] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit one bounded audit event without PDF, question or chunk text."""

    payload: dict[str, object] = {
        "event": "document.audit",
        "action": action,
        "result": result,
        "document_id": document_id,
        "version_id": version_id,
        "tenant_id": tenant_id,
        "job_id": job_id,
        "metadata": dict(metadata or {}),
    }
    (logger or logging.getLogger("document_intelligence_service.audit")).info(
        "%s",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
