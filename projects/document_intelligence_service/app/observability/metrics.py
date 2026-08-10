"""Small dependency-free metrics registry for the local-first service."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Mapping


class MetricsRegistry:
    """Collect bounded counters and latency samples without logging PII."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._histograms: dict[
            tuple[str, tuple[tuple[str, str], ...]], list[float]
        ] = defaultdict(list)

    def increment(
        self,
        name: str,
        labels: Mapping[str, str] | None = None,
        value: int = 1,
    ) -> None:
        """Increment one labeled counter."""

        if value < 0:
            raise ValueError("counter increment cannot be negative")
        key = (name, _label_key(labels))
        with self._lock:
            self._counters[key] += value

    def observe(
        self,
        name: str,
        value: float,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Append one bounded non-negative observation to a histogram."""

        if value < 0:
            raise ValueError("metric observations cannot be negative")
        key = (name, _label_key(labels))
        with self._lock:
            samples = self._histograms[key]
            samples.append(float(value))
            if len(samples) > 1000:
                del samples[: len(samples) - 1000]

    def snapshot(self) -> dict[str, object]:
        """Return JSON-safe counters and p50/p95 histogram summaries."""

        with self._lock:
            counters = [
                {
                    "name": name,
                    "labels": dict(labels),
                    "value": value,
                }
                for (name, labels), value in sorted(self._counters.items())
            ]
            histograms = [
                {
                    "name": name,
                    "labels": dict(labels),
                    "count": len(values),
                    "p50_ms": _percentile(values, 0.50),
                    "p95_ms": _percentile(values, 0.95),
                    "last_ms": values[-1] if values else None,
                }
                for (name, labels), values in sorted(self._histograms.items())
            ]
        return {"counters": counters, "histograms": histograms}


def _label_key(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight
