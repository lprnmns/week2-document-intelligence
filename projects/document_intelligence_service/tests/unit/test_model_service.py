"""Tests for system profile and local model compatibility decisions."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from projects.document_intelligence_service.app.application.model_service import (
    ModelCompatibilityEstimator,
    ModelService,
)
from projects.document_intelligence_service.app.domain.model_profile import (
    ModelMetadata,
    RuntimeStatus,
)
from projects.document_intelligence_service.app.domain.system_profile import SystemProfile


class FakeHost:
    def detect(self) -> SystemProfile:
        return SystemProfile(
            operating_system="Linux",
            architecture="x86_64",
            cpu_model="Test CPU",
            logical_cores=16,
            physical_cores=8,
            total_ram_gb=32.0,
            available_ram_gb=20.0,
            gpu_available=False,
            gpu_name=None,
            gpu_vram_gb=None,
            acceleration=(),
            containerized=True,
        )


class FakeRuntime:
    def __init__(self, available: bool = True) -> None:
        self.available = available

    async def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus(
            name="ollama",
            available=self.available,
            detail="runtime available" if self.available else "runtime unavailable",
            installed_count=1 if self.available else 0,
        )

    async def list_installed_models(self) -> tuple[ModelMetadata, ...]:
        return (
            ModelMetadata(
                model_id="qwen3:4b",
                size_bytes=2_500_000_000,
                parameter_count_b=4.0,
                quantization="Q4_K_M",
            ),
        )

    async def pull_model(
        self,
        model_id: str,
        on_progress: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    ) -> None:
        del model_id, on_progress


def test_compatibility_estimator_is_deterministic_and_heuristic() -> None:
    system = FakeHost().detect()
    metadata = ModelMetadata(model_id="qwen3:4b", size_bytes=2_500_000_000)
    estimator = ModelCompatibilityEstimator()
    first = estimator.estimate(system=system, metadata=metadata)
    second = estimator.estimate(system=system, metadata=metadata)
    assert first == second
    assert first.confidence == "heuristic"
    assert first.estimated_memory_gb is not None


def test_unknown_metadata_does_not_invent_precision() -> None:
    result = ModelCompatibilityEstimator().estimate(
        system=FakeHost().detect(),
        metadata=None,
    )
    assert result.classification.value == "unknown"
    assert result.estimated_memory_gb is None
    assert result.label == "Resource estimate unavailable"


def test_model_snapshot_distinguishes_installed_model_and_runtime() -> None:
    service = ModelService(
        host_profile=FakeHost(),
        runtime=FakeRuntime(),
        generation_model="qwen3:4b",
        embedding_model="dense-model",
        sparse_model="bm25_qdrant_idf_v2",
        reranker_model="reranker-model",
        ollama_catalog=("qwen3:4b", "missing-model"),
    )
    snapshot = asyncio.run(service.snapshot())
    models = cast(list[dict[str, object]], snapshot["models"])
    runtime = cast(dict[str, object], snapshot["runtime"])
    assert snapshot["generation_readiness"] == {"model": "qwen3:4b", "status": "ready"}
    assert runtime["available"] is True
    assert any(item["model_id"] == "qwen3:4b" and item["installed"] for item in models)
    assert any(item["model_id"] == "missing-model" and not item["installed"] for item in models)


def test_missing_selected_model_is_not_runtime_unavailable() -> None:
    service = ModelService(
        host_profile=FakeHost(),
        runtime=FakeRuntime(),
        generation_model="missing-model",
        embedding_model="dense-model",
        sparse_model="bm25",
        reranker_model="reranker",
    )
    snapshot = asyncio.run(service.snapshot())
    runtime = cast(dict[str, object], snapshot["runtime"])
    readiness = cast(dict[str, object], snapshot["generation_readiness"])
    assert runtime["available"] is True
    assert readiness["status"] == "model_missing"


def test_generation_probe_separates_installation_from_runtime_readiness() -> None:
    service = ModelService(
        host_profile=FakeHost(),
        runtime=FakeRuntime(),
        generation_model="gemma3:4b",
        embedding_model="dense-model",
        sparse_model="bm25",
        reranker_model="reranker",
        ollama_catalog=("qwen3:4b",),
    )

    initial = asyncio.run(service.snapshot())
    initial_models = cast(list[dict[str, object]], initial["models"])
    initial_qwen = next(item for item in initial_models if item["model_id"] == "qwen3:4b")
    assert initial_qwen["readiness"] == "installed_unverified"

    service.record_generation_probe(
        "qwen3:4b",
        status="last_probe_failed",
        reason="EMPTY_RESPONSE",
    )
    failed = asyncio.run(service.snapshot())
    failed_models = cast(list[dict[str, object]], failed["models"])
    failed_qwen = next(item for item in failed_models if item["model_id"] == "qwen3:4b")
    assert failed_qwen["installed"] is True
    assert failed_qwen["readiness"] == "last_probe_failed"
    assert failed_qwen["readiness_reason"] == "EMPTY_RESPONSE"


def test_unconfigured_installed_runtime_model_is_not_offered_as_generation_role() -> None:
    class RuntimeWithExtraModel(FakeRuntime):
        async def list_installed_models(self) -> tuple[ModelMetadata, ...]:
            return (
                ModelMetadata(model_id="qwen3:4b", size_bytes=2_500_000_000),
                ModelMetadata(model_id="embedding-only:latest", size_bytes=500_000_000),
            )

    service = ModelService(
        host_profile=FakeHost(),
        runtime=RuntimeWithExtraModel(),
        generation_model="qwen3:4b",
        embedding_model="dense-model",
        sparse_model="bm25",
        reranker_model="reranker",
        ollama_catalog=("qwen3:4b",),
    )
    snapshot = asyncio.run(service.snapshot())
    installed_models = cast(list[dict[str, object]], snapshot["installed_models"])
    models = cast(list[dict[str, object]], snapshot["models"])

    assert any(
        item["model_id"] == "embedding-only:latest"
        for item in installed_models
    )
    assert not any(
        item["model_id"] == "embedding-only:latest" for item in models
    )


def test_model_pull_validation_happens_before_runtime_operation() -> None:
    service = ModelService(
        host_profile=FakeHost(),
        runtime=FakeRuntime(),
        generation_model="qwen3:4b",
        embedding_model="dense-model",
        sparse_model="bm25",
        reranker_model="reranker",
        ollama_catalog=("qwen3:4b",),
    )

    assert asyncio.run(service.validate_model_pull("unknown:latest")) == (
        "not_allowlisted"
    )


def test_model_pull_validation_distinguishes_unavailable_runtime() -> None:
    service = ModelService(
        host_profile=FakeHost(),
        runtime=FakeRuntime(available=False),
        generation_model="qwen3:4b",
        embedding_model="dense-model",
        sparse_model="bm25",
        reranker_model="reranker",
        ollama_catalog=("qwen3:4b",),
    )

    assert asyncio.run(service.validate_model_pull("qwen3:4b")) == (
        "runtime_unavailable"
    )
