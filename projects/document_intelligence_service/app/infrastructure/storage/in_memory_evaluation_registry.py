"""Small process-local registry for asynchronous evaluation runs."""

import asyncio
from dataclasses import replace

from ...domain.evaluation import EvaluationRunSnapshot


class InMemoryEvaluationRegistry:
    """Keep evaluation lifecycle state observable without hiding failures."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._runs: dict[str, EvaluationRunSnapshot] = {}

    async def create(self, snapshot: EvaluationRunSnapshot) -> None:
        """Persist one new run and reject accidental ID reuse."""

        async with self._lock:
            if snapshot.run_id in self._runs:
                raise KeyError(f"evaluation run already exists: {snapshot.run_id}")
            self._runs[snapshot.run_id] = snapshot

    async def get(self, run_id: str) -> EvaluationRunSnapshot | None:
        """Return one run snapshot, if it exists."""

        async with self._lock:
            return self._runs.get(run_id)

    async def update(self, snapshot: EvaluationRunSnapshot) -> None:
        """Replace one existing run under the registry lock."""

        async with self._lock:
            if snapshot.run_id not in self._runs:
                raise KeyError(f"unknown evaluation run: {snapshot.run_id}")
            self._runs[snapshot.run_id] = replace(snapshot)

    async def list(self, limit: int) -> tuple[EvaluationRunSnapshot, ...]:
        """Return newest runs with a bounded response size."""

        if limit <= 0 or limit > 100:
            raise ValueError("evaluation limit must be between 1 and 100")
        async with self._lock:
            ordered = sorted(
                self._runs.values(),
                key=lambda run: (run.requested_at, run.run_id),
                reverse=True,
            )
            return tuple(ordered[:limit])
