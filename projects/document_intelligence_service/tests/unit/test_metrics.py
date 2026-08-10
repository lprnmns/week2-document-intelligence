"""Tests for bounded privacy-safe operational metrics."""

from projects.document_intelligence_service.app.observability.metrics import (
    MetricsRegistry,
)


def test_metrics_snapshot_keeps_labels_and_percentiles_without_query_text() -> None:
    registry = MetricsRegistry()
    registry.increment("rag_query_total", {"decision": "no_answer"})
    registry.observe("rag_query_duration_ms", 10.0, {"stage": "search"})
    registry.observe("rag_query_duration_ms", 30.0, {"stage": "search"})

    snapshot = registry.snapshot()

    assert snapshot["counters"] == [
        {
            "name": "rag_query_total",
            "labels": {"decision": "no_answer"},
            "value": 1,
        }
    ]
    assert snapshot["histograms"] == [
        {
            "name": "rag_query_duration_ms",
            "labels": {"stage": "search"},
            "count": 2,
            "p50_ms": 20.0,
            "p95_ms": 29.0,
            "last_ms": 30.0,
        }
    ]
