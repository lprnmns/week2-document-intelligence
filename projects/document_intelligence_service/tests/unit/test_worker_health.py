"""Tests for the shared worker heartbeat health boundary."""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from projects.document_intelligence_service.app.domain.health import DependencyState
from projects.document_intelligence_service.app.domain.model_profile import (
    ModelMetadata,
    RuntimeStatus,
)
from projects.document_intelligence_service.app.infrastructure.health_checks import (
    HeartbeatFileProbe,
    LocalModelHealthProbe,
)
from projects.document_intelligence_service.app.infrastructure.worker_heartbeat import (
    WorkerHeartbeat,
)


def test_heartbeat_probe_reports_a_fresh_worker(tmp_path: Path) -> None:
    path = tmp_path / "worker_heartbeat.json"
    assert WorkerHeartbeat(path).write(state="idle") is True

    result = asyncio.run(
        HeartbeatFileProbe(
            name="worker",
            path=path,
            stale_after_seconds=10,
        ).check()
    )

    assert result.state is DependencyState.UP
    assert "state=idle" in (result.detail or "")


def test_heartbeat_probe_reports_missing_worker(tmp_path: Path) -> None:
    result = asyncio.run(
        HeartbeatFileProbe(
            name="worker",
            path=tmp_path / "missing.json",
            stale_after_seconds=10,
        ).check()
    )

    assert result.state is DependencyState.DOWN
    assert result.detail == "heartbeat missing or invalid"


def test_heartbeat_probe_reports_stale_worker(tmp_path: Path) -> None:
    path = tmp_path / "worker_heartbeat.json"
    path.write_text(
        json.dumps(
            {
                "updated_at": (
                    datetime.now(timezone.utc) - timedelta(seconds=30)
                ).isoformat(),
                "state": "idle",
            }
        ),
        encoding="utf-8",
    )

    result = asyncio.run(
        HeartbeatFileProbe(
            name="worker",
            path=path,
            stale_after_seconds=10,
        ).check()
    )

    assert result.state is DependencyState.DOWN
    assert "heartbeat stale" in (result.detail or "")


class FakeRuntime:
    """Runtime double for selected-model readiness states."""

    def __init__(self, available: bool, models: tuple[str, ...] = ()) -> None:
        self.available = available
        self.models = models

    async def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus("ollama", self.available)

    async def list_installed_models(self) -> tuple[ModelMetadata, ...]:
        return tuple(ModelMetadata(model_id=model) for model in self.models)


def test_model_health_distinguishes_runtime_unavailable() -> None:
    result = asyncio.run(
        LocalModelHealthProbe(
            name="llm",
            runtime=FakeRuntime(False),
            model_id="gemma3:4b",
        ).check()
    )

    assert result.state is DependencyState.DOWN
    assert result.detail == "runtime unavailable"


def test_model_health_distinguishes_selected_model_missing() -> None:
    result = asyncio.run(
        LocalModelHealthProbe(
            name="llm",
            runtime=FakeRuntime(True, ("qwen3:4b",)),
            model_id="gemma3:4b",
        ).check()
    )

    assert result.state is DependencyState.DOWN
    assert result.detail == "selected generation model is not installed"


def test_model_health_reports_selected_model_ready() -> None:
    result = asyncio.run(
        LocalModelHealthProbe(
            name="llm",
            runtime=FakeRuntime(True, ("gemma3:4b",)),
            model_id="gemma3:4b",
        ).check()
    )

    assert result.state is DependencyState.UP
    assert "selected model ready" in (result.detail or "")
