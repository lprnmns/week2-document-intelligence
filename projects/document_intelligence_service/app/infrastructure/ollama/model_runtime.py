"""Ollama runtime adapter for discovery and controlled local installation."""

from collections.abc import Awaitable, Callable
import json
import re

import httpx

from ...domain.model_profile import ModelMetadata, RuntimeStatus

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,127}$")


class OllamaModelRuntimeAdapter:
    """Use Ollama's HTTP API; arbitrary shell execution is impossible here."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 3.0,
        allowed_model_ids: tuple[str, ...] = (),
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._allowed_model_ids = frozenset(
            model_id for model_id in allowed_model_ids if _MODEL_ID.fullmatch(model_id)
        )

    async def check_runtime(self) -> RuntimeStatus:
        """Distinguish a reachable runtime from an unavailable runtime."""

        try:
            models = await self.list_installed_models()
        except (httpx.HTTPError, OSError, ValueError):
            return RuntimeStatus(
                name="ollama",
                available=False,
                detail="runtime unavailable",
            )
        return RuntimeStatus(
            name="ollama",
            available=True,
            detail="runtime available",
            installed_count=len(models),
        )

    async def list_installed_models(self) -> tuple[ModelMetadata, ...]:
        """Read installed model metadata from `/api/tags`."""

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("models"), list):
            raise ValueError("Ollama tags response has an invalid shape")
        return tuple(
            metadata
            for raw in body["models"]
            if isinstance(raw, dict)
            for metadata in (_parse_metadata(raw),)
            if metadata is not None
        )

    async def pull_model(
        self,
        model_id: str,
        on_progress: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    ) -> None:
        """Pull one allow-listed model and forward actual runtime progress."""

        self._validate_model_id(model_id)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json={"name": model_id, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if on_progress is not None and isinstance(payload, dict):
                        await on_progress(_safe_progress(payload))

    def _validate_model_id(self, model_id: str) -> None:
        if not _MODEL_ID.fullmatch(model_id):
            raise ValueError("model identifier is invalid")
        if self._allowed_model_ids and model_id not in self._allowed_model_ids:
            raise ValueError("model identifier is not allow-listed")


def _parse_metadata(raw: dict[str, object]) -> ModelMetadata | None:
    name = raw.get("name")
    if not isinstance(name, str) or not _MODEL_ID.fullmatch(name):
        return None
    details = raw.get("details")
    detail_map = details if isinstance(details, dict) else {}
    return ModelMetadata(
        model_id=name,
        size_bytes=_as_int(raw.get("size")),
        parameter_count_b=_parameter_count(detail_map.get("parameter_size")),
        quantization=_as_str(detail_map.get("quantization_level")),
        format=_as_str(detail_map.get("format")),
        family=_as_str(detail_map.get("family")),
    )


def _safe_progress(payload: dict[str, object]) -> dict[str, object]:
    """Keep only Ollama progress fields useful to a local UI."""

    allowed = {"status", "total", "completed", "digest"}
    return {
        key: value
        for key, value in payload.items()
        if key in allowed and isinstance(value, (str, int, float))
    }


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _parameter_count(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([BMKT]?)\s*", value, re.I)
    if match is None:
        return None
    number = float(match.group(1))
    multiplier = {"": 1.0, "K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1000.0}
    return number * multiplier[match.group(2).upper()]
