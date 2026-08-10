"""Health probes for HTTP dependencies and the ingestion worker."""

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx

from ..domain.health import DependencyHealth, DependencyState
from ..domain.model_profile import RuntimeStatus


class HttpHealthProbe:
    """Check an HTTP dependency and convert failures into domain state."""

    def __init__(self, *, name: str, url: str, timeout_seconds: float) -> None:
        self._name = name
        self._url = url
        self._timeout_seconds = timeout_seconds

    async def check(self) -> DependencyHealth:
        """Return health state without leaking connection details."""

        started_at = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.get(self._url)
                response.raise_for_status()
        except (httpx.HTTPError, OSError):
            return DependencyHealth(
                name=self._name,
                state=DependencyState.DOWN,
                latency_ms=_elapsed_ms(started_at),
                detail="dependency unavailable",
            )

        return DependencyHealth(
            name=self._name,
            state=DependencyState.UP,
            latency_ms=_elapsed_ms(started_at),
        )


class LocalModelHealthProbe:
    """Check runtime reachability and selected-model installation separately."""

    def __init__(self, *, name: str, runtime: object, model_id: str) -> None:
        self._name = name
        self._runtime = runtime
        self._model_id = model_id

    async def check(self) -> DependencyHealth:
        """Return ``down`` with a safe reason for either failure mode."""

        started_at = perf_counter()
        try:
            status = await self._runtime.check_runtime()  # type: ignore[attr-defined]
        except Exception:
            status = RuntimeStatus(
                name="ollama",
                available=False,
                detail="runtime unavailable",
            )
        if not status.available:
            return DependencyHealth(
                name=self._name,
                state=DependencyState.DOWN,
                latency_ms=_elapsed_ms(started_at),
                detail="runtime unavailable",
            )
        try:
            installed = await self._runtime.list_installed_models()  # type: ignore[attr-defined]
        except Exception:
            return DependencyHealth(
                name=self._name,
                state=DependencyState.DOWN,
                latency_ms=_elapsed_ms(started_at),
                detail="runtime model listing unavailable",
            )
        if not any(item.model_id == self._model_id for item in installed):
            return DependencyHealth(
                name=self._name,
                state=DependencyState.DOWN,
                latency_ms=_elapsed_ms(started_at),
                detail="selected generation model is not installed",
            )
        return DependencyHealth(
            name=self._name,
            state=DependencyState.UP,
            latency_ms=_elapsed_ms(started_at),
            detail=f"selected model ready: {self._model_id}",
        )


class HeartbeatFileProbe:
    """Check a worker heartbeat written to a shared volume."""

    def __init__(
        self,
        *,
        name: str,
        path: str | Path,
        stale_after_seconds: float,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        self._name = name
        self._path = Path(path)
        self._stale_after_seconds = stale_after_seconds

    async def check(self) -> DependencyHealth:
        """Return down when the heartbeat is missing, invalid or stale."""

        started_at = perf_counter()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            updated_at = datetime.fromisoformat(str(payload["updated_at"]))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - updated_at).total_seconds(),
            )
            worker_state = str(payload.get("state", "unknown"))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return DependencyHealth(
                name=self._name,
                state=DependencyState.DOWN,
                latency_ms=_elapsed_ms(started_at),
                detail="heartbeat missing or invalid",
            )

        if age_seconds > self._stale_after_seconds or worker_state == "stopped":
            return DependencyHealth(
                name=self._name,
                state=DependencyState.DOWN,
                latency_ms=_elapsed_ms(started_at),
                detail=f"heartbeat stale ({age_seconds:.1f}s), state={worker_state}",
            )
        return DependencyHealth(
            name=self._name,
            state=DependencyState.UP,
            latency_ms=_elapsed_ms(started_at),
            detail=f"heartbeat {age_seconds:.1f}s ago, state={worker_state}",
        )


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
