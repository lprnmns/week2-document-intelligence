"""Contract tests for the local Demo trace and system/model endpoints."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx
from fastapi import FastAPI

from projects.document_intelligence_service.app.application.health_service import (
    HealthService,
)
from projects.document_intelligence_service.app.application.query_service import (
    QueryService,
)
from projects.document_intelligence_service.app.application.evaluation_service import (
    EvaluationService,
)
from projects.document_intelligence_service.app.domain.answerability import (
    AnswerabilityPolicy,
)
from projects.document_intelligence_service.app.domain.entities import RetrievalMode
from projects.document_intelligence_service.app.main import create_app
from projects.document_intelligence_service.app.settings import Settings
from projects.document_intelligence_service.app.infrastructure.storage.in_memory_evaluation_registry import (
    InMemoryEvaluationRegistry,
)
from projects.document_intelligence_service.tests.unit.test_evaluation_service import (
    FakeEvaluationExecutor,
)
from projects.document_intelligence_service.tests.unit.test_query_service import (
    FakeAnswerGenerator,
)
from projects.document_intelligence_service.tests.unit.test_retrieval_service import (
    make_service,
)


async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Keep the lifespan alive while a background demo run is polled."""

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-Request-ID": "demo-system-contract"},
        ) as client:
            yield client


def test_demo_trace_publishes_real_query_stages_and_canonical_sources() -> None:
    async def scenario() -> None:
        generator = FakeAnswerGenerator()
        app = create_app(
            health_service=HealthService(()),
            query_service=QueryService(
                retrieval_service=make_service(),
                answerability=AnswerabilityPolicy(min_dense_score=0.45),
                answer_generator=generator,
            ),
        )
        async for client in _client(app):
            start = await client.post(
                "/v1/demo/query-runs",
                json={
                    "question": "Qdrant ne işe yarar?",
                    "retrieval_mode": RetrievalMode.HYBRID.value,
                    "top_k": 2,
                    "reranker_enabled": False,
                },
            )
            assert start.status_code == 202
            start_body = cast(dict[str, object], start.json())
            assert cast(str, start_body["request_id"]).startswith("req_")

            snapshot: dict[str, object] = {}
            for _ in range(20):
                response = await client.get(
                    f"/v1/demo/query-runs/{start_body['run_id']}"
                )
                assert response.status_code == 200
                snapshot = response.json()
                if snapshot["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.01)

            assert snapshot["status"] == "completed"
            events = cast(list[dict[str, object]], snapshot["events"])
            stages = {event["stage"] for event in events}
            assert {
                "scope_normalization",
                "query_representation",
                "dense_retrieval",
                "sparse_retrieval",
                "rrf_fusion",
                "reranker",
                "answerability",
                "llm",
                "response",
            } <= stages
            reranker_events = [event for event in events if event["stage"] == "reranker"]
            assert reranker_events[-1]["status"] == "skipped"
            reranker_details = cast(
                dict[str, object], reranker_events[-1]["details"]
            )
            assert reranker_details["reason"] == "configuration"
            result = cast(dict[str, object], snapshot["result"])
            assert result["decision"] == "answered"
            sources = cast(list[dict[str, object]], result["sources"])
            assert sources[0]["source_id"] == "shared"
            assert generator.call_count == 1

    asyncio.run(scenario())


class FakeModelService:
    """Small endpoint double for readiness-state contract tests."""

    def __init__(self, validation_status: str = "ready") -> None:
        self.validation_status = validation_status

    async def snapshot(self) -> dict[str, object]:
        return {
            "system": {
                "operating_system": "Linux",
                "architecture": "x86_64",
                "cpu": {"model": "Test CPU", "cores": 8, "threads": 16},
                "memory": {"total_gb": 32.0, "available_gb": 20.0},
                "gpu": {"available": False, "name": None, "vram_gb": None},
                "acceleration": [],
                "containerized": True,
            },
            "runtime": {
                "name": "ollama",
                "available": self.validation_status != "runtime_unavailable",
                "detail": "test",
                "installed_count": 1,
            },
            "configured": {
                "generation": "qwen3:4b",
                "embedding": "dense",
                "sparse": "bm25",
                "reranker": "reranker",
            },
            "generation_readiness": {"model": "qwen3:4b", "status": "ready"},
            "models": [],
        }

    async def validate_generation_model(self, model_id: str) -> str:
        del model_id
        return self.validation_status


def test_system_profile_returns_sanitized_shape_without_host_paths() -> None:
    async def scenario() -> None:
        app = create_app(
            health_service=HealthService(()),
            model_service=FakeModelService(),  # type: ignore[arg-type]
        )
        async for client in _client(app):
            response = await client.get("/v1/system/profile")
            assert response.status_code == 200
            body = response.json()
            assert body["system"]["cpu"]["model"] == "Test CPU"
            assert "username" not in str(body).lower()
            assert "home/" not in str(body)
            assert "environment_variables" not in body

    asyncio.run(scenario())


def test_demo_model_validation_distinguishes_missing_runtime_and_allowlist() -> None:
    async def scenario() -> None:
        for validation_status, expected_code, expected_reason in (
            ("model_missing", "DEPENDENCY_UNAVAILABLE", "model_missing"),
            ("runtime_unavailable", "DEPENDENCY_UNAVAILABLE", "runtime_unavailable"),
            ("not_allowlisted", "INVALID_REQUEST", "not_allowlisted"),
        ):
            app = create_app(
                health_service=HealthService(()),
                query_service=QueryService(
                    retrieval_service=make_service(),
                    answerability=AnswerabilityPolicy(min_dense_score=0.45),
                    answer_generator=FakeAnswerGenerator(),
                ),
                model_service=FakeModelService(validation_status),  # type: ignore[arg-type]
            )
            async for client in _client(app):
                response = await client.post(
                    "/v1/demo/query-runs",
                    json={
                        "question": "Qdrant ne işe yarar?",
                        "generation_model": "selected-model",
                    },
                )
                assert response.status_code == (
                    503 if expected_code == "DEPENDENCY_UNAVAILABLE" else 400
                )
                error = response.json()["error"]
                assert error["code"] == expected_code
                assert error["reason"] == expected_reason

    asyncio.run(scenario())


def test_model_pull_rejects_unknown_id_before_creating_pull_state() -> None:
    async def scenario() -> None:
        from projects.document_intelligence_service.tests.unit.test_model_service import (
            FakeHost,
            FakeRuntime,
        )
        from projects.document_intelligence_service.app.application.model_service import (
            ModelService,
        )

        app = create_app(
            settings=Settings(
                local_model_management_enabled=True,
                llm_model="qwen3:4b",
                ollama_model_catalog="qwen3:4b",
            ),
            health_service=HealthService(()),
            model_service=ModelService(
                host_profile=FakeHost(),
                runtime=FakeRuntime(),
                generation_model="qwen3:4b",
                embedding_model="dense-model",
                sparse_model="bm25",
                reranker_model="reranker",
                ollama_catalog=("qwen3:4b",),
            ),
        )
        async for client in _client(app):
            response = await client.post(
                "/v1/system/models/pulls",
                json={"model_id": "not-allowlisted:latest"},
            )
            assert response.status_code == 400
            assert response.json()["error"]["reason"] == "not_allowlisted"
            assert app.state.model_pull_store == {}

    asyncio.run(scenario())


def test_evaluation_api_exposes_reproducibility_configuration_and_metrics(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service = EvaluationService(
            registry=InMemoryEvaluationRegistry(),
            executor=FakeEvaluationExecutor(),
            artifact_dir=tmp_path / "artifacts",
            repo_root=tmp_path,
            default_configuration={
                "dataset_version": "golden-v1",
                "corpus_snapshot_id": "snapshot-1",
                "models": {
                    "dense": "dense-v1",
                    "sparse": "bm25-v1",
                    "reranker": "reranker-v1",
                    "llm": "llm-v1",
                },
                "retrieval": {"candidate_k": 30, "fusion_k": 20, "rerank_k": 5},
                "machine": {"cpu": {"threads": 8}, "memory": {"total_gb": 32}},
            },
        )
        app = create_app(
            health_service=HealthService(()),
            evaluation_service=service,
        )
        async for client in _client(app):
            configuration_response = await client.get("/v1/evaluations/config")
            assert configuration_response.status_code == 200
            assert configuration_response.json()["corpus_snapshot_id"] == "snapshot-1"
            response = await client.post(
                "/v1/evaluations/runs",
                json={
                    "evaluation_type": "retrieval",
                    "dataset": "mentor_program_pdf_rag_golden_v1",
                    "split": "test",
                    "mode": "hybrid",
                    "top_k": 5,
                    "reranker_enabled": True,
                },
            )
            assert response.status_code == 202
            queued = cast(dict[str, object], response.json())
            queued_configuration = cast(
                dict[str, object], queued["configuration"]
            )
            queued_models = cast(
                dict[str, object], queued_configuration["models"]
            )
            assert queued_configuration["dataset_version"] == "golden-v1"
            assert queued_configuration["corpus_snapshot_id"] == "snapshot-1"
            assert queued_models["dense"] == "dense-v1"

            completed: dict[str, object] = {}
            for _ in range(30):
                completed = cast(dict[str, object], (await client.get(
                    f"/v1/evaluations/runs/{queued['run_id']}"
                )).json())
                if completed["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.01)

            assert completed["status"] == "succeeded"
            configuration = cast(dict[str, object], completed["configuration"])
            retrieval = cast(dict[str, object], configuration["retrieval"])
            metrics = cast(dict[str, object], completed["metrics"])
            assert retrieval["fusion_k"] == 20
            assert metrics["retrieval_p50_ms"] == 12.0
            assert metrics["failure_rate"] == 0.5

    asyncio.run(scenario())
