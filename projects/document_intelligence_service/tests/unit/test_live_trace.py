"""Tests for the bounded demo trace transport."""

from typing import cast

from projects.document_intelligence_service.app.observability.query_trace import (
    LiveQueryTraceStore,
)


def test_live_trace_store_keeps_sequence_and_sanitizes_bounded_details() -> None:
    store = LiveQueryTraceStore(ttl_seconds=60, max_runs=2)
    run_id = store.create(request_id="req_demo")
    recorder = store.recorder(run_id)
    recorder.emit(
        "dense_retrieval",
        "passed",
        "Dense candidates ready",
        {"count": 2, "excerpt": "x" * 1000},
        12.5,
    )
    recorder.emit("reranker", "skipped", "Stage skipped", {"reason": "configuration"}, 0)

    snapshot = store.snapshot(run_id)
    assert snapshot["request_id"] == "req_demo"
    events = cast(list[dict[str, object]], snapshot["events"])
    assert [event["sequence"] for event in events] == [1, 2]
    details = cast(dict[str, object], events[0]["details"])
    assert len(cast(str, details["excerpt"])) == 500
    assert events[1]["status"] == "skipped"


def test_live_trace_store_publishes_result_without_full_prompt() -> None:
    store = LiveQueryTraceStore()
    run_id = store.create(request_id="req_result")
    store.finish(run_id, {"answer": "grounded", "sources": ["chunk-1"]})

    snapshot = store.snapshot(run_id)
    assert snapshot["status"] == "completed"
    assert snapshot["result"] == {"answer": "grounded", "sources": ["chunk-1"]}


def test_live_trace_store_preserves_bounded_diagnostic_presentation() -> None:
    store = LiveQueryTraceStore()
    run_id = store.create(request_id="req_presentation")
    store.finish(
        run_id,
        {
            "expected_check": {
                "presentation": {
                    "trusted_chunks": [
                        {
                            "source_id": "doc:parent:1:child:1",
                            "chunk_text": "10 August 2026 23:59",
                        }
                    ],
                    "fact_survival": [{"value": "23:59", "prompt": True}],
                }
            }
        },
    )

    snapshot = store.snapshot(run_id)
    result = snapshot["result"]
    assert isinstance(result, dict)
    expected_check = result["expected_check"]
    assert isinstance(expected_check, dict)
    presentation = expected_check["presentation"]
    assert isinstance(presentation, dict)
    assert presentation["trusted_chunks"][0]["chunk_text"] == "10 August 2026 23:59"
    assert "[truncated]" not in str(presentation)


def test_live_trace_store_preserves_expected_check_leaf_values() -> None:
    store = LiveQueryTraceStore()
    run_id = store.create(request_id="req_expected_check")
    store.finish(
        run_id,
        {
            "expected_check": {
                "root_cause": "PROMPT_CONSTRUCTION_LOSS",
                "first_divergence": "prompt",
                "answer_check": {
                    "required_facts": [{"type": "time", "value": "23:59", "matched": False}],
                },
            }
        },
    )
    result = cast(dict[str, object], store.snapshot(run_id)["result"])
    expected_check = cast(dict[str, object], result["expected_check"])
    assert expected_check["root_cause"] == "PROMPT_CONSTRUCTION_LOSS"
    answer_check = cast(dict[str, object], expected_check["answer_check"])
    assert answer_check["required_facts"] == [{"type": "time", "value": "23:59", "matched": False}]
    assert "bounded details omitted" not in str(expected_check)
