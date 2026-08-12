"""Tests for asynchronous evaluation lifecycle and raw artifacts."""

import asyncio
import json
from pathlib import Path

import pytest

from projects.document_intelligence_service.app.application.evaluation_service import (
    EvaluationExecution,
    EvaluationService,
    EvaluationSpec,
    _git_sha,
)
from projects.document_intelligence_service.app.domain.entities import (
    EvaluationRunStatus,
    RetrievalMode,
)
from projects.document_intelligence_service.app.domain.errors import (
    ErrorCode,
    ServiceError,
)
from projects.document_intelligence_service.app.infrastructure.storage.in_memory_evaluation_registry import (
    InMemoryEvaluationRegistry,
)


class FakeEvaluationExecutor:
    """Return deterministic metrics without loading Qdrant or a model."""

    def execute(self, spec: EvaluationSpec) -> EvaluationExecution:
        return EvaluationExecution(
            case_count=3,
            metrics={"recall_at_5": 1.0, "query_count": 3},
            raw={
                "strategy": spec.mode.value,
                "total_latency": {
                    "p50_ms": 12.0,
                    "p95_ms": 18.0,
                },
                "embedding_latency": {
                    "p50_ms": 2.0,
                    "p95_ms": 3.0,
                },
                "search_latency": {
                    "p50_ms": 4.0,
                    "p95_ms": 5.0,
                },
                "rerank_latency": None,
                "observations": [
                    {"case_id": "a", "category": "direct_fact", "status": "ok"},
                    {"case_id": "b", "category": "near_miss", "status": "error"},
                ],
            },
        )


def spec() -> EvaluationSpec:
    """Create one bounded test configuration."""

    return EvaluationSpec(
        evaluation_type="retrieval",
        dataset="mentor_program_pdf_rag_golden_v1",
        split="test",
        mode=RetrievalMode.HYBRID,
        top_k=5,
        reranker_enabled=False,
    )


def test_evaluation_run_persists_metrics_and_raw_artifact(tmp_path: Path) -> None:
    service = EvaluationService(
        registry=InMemoryEvaluationRegistry(),
        executor=FakeEvaluationExecutor(),
        artifact_dir=tmp_path / "artifacts",
        repo_root=tmp_path,
    )

    queued = asyncio.run(service.create_run(spec()))
    assert queued.status is EvaluationRunStatus.QUEUED
    completed = asyncio.run(service.execute_run(queued.run_id))

    assert completed.status is EvaluationRunStatus.SUCCEEDED
    assert completed.case_count == 3
    assert completed.metrics is not None
    assert completed.metrics["recall_at_5"] == 1.0
    assert completed.metrics["retrieval_p50_ms"] == 12.0
    assert completed.artifact_path is not None
    artifact = json.loads(Path(completed.artifact_path).read_text(encoding="utf-8"))
    assert artifact["run_id"] == queued.run_id
    assert artifact["metrics"]["recall_at_5"] == 1.0
    assert artifact["metrics"]["retrieval_p50_ms"] == 12.0
    assert artifact["metrics"]["retrieval_p95_ms"] == 18.0
    assert artifact["metrics"]["failure_rate"] == 0.5
    assert artifact["metrics"]["slice_direct_fact_count"] == 1
    assert len(artifact["run"]["observations"]) == 2


def test_missing_evaluation_executor_fails_explicitly() -> None:
    service = EvaluationService(
        registry=InMemoryEvaluationRegistry(),
        executor=None,
        artifact_dir="/tmp/document-intelligence-eval-test",
        repo_root="/tmp",
    )

    queued = asyncio.run(service.create_run(spec()))
    failed = asyncio.run(service.execute_run(queued.run_id))

    assert failed.status is EvaluationRunStatus.FAILED
    assert failed.error_code == ErrorCode.DEPENDENCY_UNAVAILABLE.value
    with pytest.raises(ServiceError) as raised:
        asyncio.run(service.get_run("eval_missing"))
    assert raised.value.code is ErrorCode.EVALUATION_NOT_FOUND


def test_explicit_source_revision_is_used_when_image_has_no_git_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DIS_SOURCE_REVISION", "source-revision-for-test")

    assert _git_sha(tmp_path) == "source-revision-for-test"


def test_clean_delivery_source_uses_parent_delivery_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("DIS_SOURCE_REVISION", raising=False)
    source_root = tmp_path / "source"
    source_root.mkdir()
    delivery_sha = "0123456789abcdef0123456789abcdef01234567"
    (tmp_path / "DELIVERY_SHA.txt").write_text(
        f"Delivery commit: {delivery_sha}\n",
        encoding="utf-8",
    )

    assert _git_sha(source_root) == delivery_sha
