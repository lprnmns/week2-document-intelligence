"""Unit tests for health orchestration."""

import asyncio

from projects.document_intelligence_service.app.application.health_service import (
    HealthService,
)
from projects.document_intelligence_service.app.domain.health import (
    DependencyHealth,
    DependencyState,
)


class FakeProbe:
    """Deterministic probe used to test application policy."""

    def __init__(self, result: DependencyHealth) -> None:
        self.result = result
        self.call_count = 0

    async def check(self) -> DependencyHealth:
        self.call_count += 1
        return self.result


def test_readiness_is_false_when_one_required_dependency_is_down() -> None:
    qdrant = FakeProbe(DependencyHealth("qdrant", DependencyState.DOWN, 1.0))
    ollama = FakeProbe(DependencyHealth("ollama", DependencyState.UP, 2.0))
    service = HealthService((qdrant, ollama))

    report = asyncio.run(service.readiness())

    assert report.ready is False
    assert qdrant.call_count == 1
    assert ollama.call_count == 1


def test_startup_state_changes_explicitly() -> None:
    service = HealthService(())

    assert service.startup_complete is False
    service.mark_started()
    assert service.startup_complete is True
    service.mark_stopped()
    assert service.startup_complete is False
