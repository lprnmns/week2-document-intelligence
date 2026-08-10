"""Framework-independent health state definitions."""

from dataclasses import dataclass
from enum import StrEnum


class DependencyState(StrEnum):
    """Availability state of an external dependency."""

    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class DependencyHealth:
    """Result returned by a dependency health probe."""

    name: str
    state: DependencyState
    latency_ms: float
    detail: str | None = None

    @property
    def available(self) -> bool:
        """Return whether the dependency can currently serve requests."""

        return self.state is DependencyState.UP


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Combined readiness result for all required dependencies."""

    checks: tuple[DependencyHealth, ...]

    @property
    def ready(self) -> bool:
        """Return true only when every required dependency is available."""

        return all(check.available for check in self.checks)
