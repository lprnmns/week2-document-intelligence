"""Tests for bounded, privacy-safe audit events."""

import json
import logging
from typing import Any

from projects.document_intelligence_service.app.observability.audit import emit_audit


def test_audit_event_contains_lifecycle_identity_but_not_document_content(
    caplog: Any,
) -> None:
    with caplog.at_level(logging.INFO, logger="document_intelligence_service.audit"):
        emit_audit(
            action="ingestion.version_activated",
            result="success",
            document_id="doc_1",
            version_id="ver_1",
            tenant_id="tenant_a",
            metadata={"points": 4},
        )

    record = caplog.records[0]
    payload = json.loads(record.message)
    assert payload["action"] == "ingestion.version_activated"
    assert payload["document_id"] == "doc_1"
    assert payload["metadata"] == {"points": 4}
    assert "question" not in record.message
    assert "evidence" not in record.message
    assert "secret PDF text" not in record.message
