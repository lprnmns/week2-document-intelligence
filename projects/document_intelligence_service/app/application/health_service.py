"""Health use cases independent from HTTP and concrete clients."""

import asyncio
from collections.abc import Sequence

from .ports import HealthProbe
from ..domain.health import ReadinessReport


class HealthService:
    """Orchestrate process, startup and dependency health decisions."""

    def __init__(self, probes: Sequence[HealthProbe]) -> None:
        self._probes = tuple(probes)
        self._startup_complete = False

    @property
    def startup_complete(self) -> bool:
        """Return whether application startup has completed."""

        return self._startup_complete

    def mark_started(self) -> None:
        """Mark startup complete after application wiring succeeds."""

        self._startup_complete = True

    def mark_stopped(self) -> None:
        """Mark the service unavailable during shutdown."""

        self._startup_complete = False

    async def readiness(self) -> ReadinessReport:
        """Check required dependencies concurrently."""

        checks = await asyncio.gather(*(probe.check() for probe in self._probes))
        return ReadinessReport(checks=tuple(checks))
