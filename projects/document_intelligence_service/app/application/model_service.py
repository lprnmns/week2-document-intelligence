"""System-aware local model discovery and compatibility application service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from .ports import HostProfilePort, ModelRuntimePort
from ..domain.model_profile import (
    CompatibilityClass,
    ModelCompatibility,
    ModelDescriptor,
    ModelMetadata,
    ModelRole,
    RuntimeStatus,
)
from ..domain.system_profile import SystemProfile


class ModelCompatibilityEstimator:
    """Deterministic heuristic estimator with explicit uncertainty."""

    def __init__(self, *, context_length: int = 4096) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        self._context_length = context_length

    def estimate(
        self,
        *,
        system: SystemProfile,
        metadata: ModelMetadata | None,
    ) -> ModelCompatibility:
        """Estimate memory/latency risk; never promise that a model will run."""

        if metadata is None:
            return ModelCompatibility(
                classification=CompatibilityClass.UNKNOWN,
                label="Resource estimate unavailable",
                reason="Model size or parameter metadata is unavailable",
                estimated_memory_gb=None,
                safe_memory_budget_gb=None,
                confidence="unknown",
            )
        estimated = _estimate_memory_gb(
            metadata,
            context_length=self._context_length,
        )
        if estimated is None or system.total_ram_gb is None:
            return ModelCompatibility(
                classification=CompatibilityClass.UNKNOWN,
                label="Resource estimate unavailable",
                reason="Hardware or model metadata is incomplete",
                estimated_memory_gb=estimated,
                safe_memory_budget_gb=None,
                confidence="unknown",
            )
        safe_budget = round(system.total_ram_gb * 0.65, 2)
        if system.gpu_available and system.gpu_vram_gb is not None:
            # Keep enough host RAM for the operating system and runtime; VRAM
            # helps, but does not make a CPU/GPU offload guarantee.
            safe_budget = round(max(safe_budget, system.gpu_vram_gb * 0.8), 2)
        ratio = estimated / safe_budget if safe_budget else 999.0
        if ratio <= 0.45:
            classification = CompatibilityClass.RECOMMENDED
            label = "Recommended"
            reason = "Comfortable estimated memory range with headroom"
        elif ratio <= 0.75:
            classification = CompatibilityClass.LIKELY_USABLE
            label = "Likely usable"
            reason = "Should fit the heuristic budget but leaves less headroom"
        elif ratio <= 1.0:
            classification = CompatibilityClass.MAY_RUN_SLOWLY
            label = "May run slowly"
            reason = "May fit, but latency and context/KV cache can be limiting"
        elif ratio <= 1.2:
            classification = CompatibilityClass.MEMORY_RISK
            label = "Memory risk"
            reason = "Estimated runtime footprint is close to or above safe budget"
        else:
            classification = CompatibilityClass.NOT_RECOMMENDED
            label = "Not recommended"
            reason = "Estimated memory exceeds the conservative local budget"
        return ModelCompatibility(
            classification=classification,
            label=label,
            reason=reason,
            estimated_memory_gb=estimated,
            safe_memory_budget_gb=safe_budget,
            confidence="heuristic",
        )


class ModelService:
    """Compose sanitized host facts with runtime models and role metadata."""

    def __init__(
        self,
        *,
        host_profile: HostProfilePort,
        runtime: ModelRuntimePort,
        generation_model: str,
        embedding_model: str,
        sparse_model: str,
        reranker_model: str,
        embedding_dimension: int | None = None,
        qdrant_collection: str | None = None,
        ollama_catalog: Sequence[str] = (),
        compatibility: ModelCompatibilityEstimator | None = None,
    ) -> None:
        self._host_profile = host_profile
        self._runtime = runtime
        self._generation_model = generation_model
        self._embedding_model = embedding_model
        self._sparse_model = sparse_model
        self._reranker_model = reranker_model
        self._embedding_dimension = embedding_dimension
        self._qdrant_collection = qdrant_collection
        self._ollama_catalog = tuple(dict.fromkeys(ollama_catalog))
        self._compatibility = compatibility or ModelCompatibilityEstimator()
        self._generation_probes: dict[str, tuple[str, str | None]] = {}

    def record_generation_probe(
        self,
        model_id: str,
        *,
        status: str,
        reason: str | None = None,
    ) -> None:
        """Record the last real generation outcome for the local UI."""

        if model_id not in set(self._ollama_catalog) | {self._generation_model}:
            return
        if status not in {"ready", "last_probe_failed"}:
            raise ValueError("unsupported generation probe status")
        self._generation_probes[model_id] = (
            status,
            reason[:120] if reason else None,
        )

    async def snapshot(self) -> dict[str, object]:
        """Return a safe system/model snapshot for the settings drawer."""

        system = self._host_profile.detect()
        runtime = await self._runtime.check_runtime()
        installed: tuple[ModelMetadata, ...] = ()
        if runtime.available:
            try:
                installed = await self._runtime.list_installed_models()
            except Exception:
                runtime = RuntimeStatus(
                    name=runtime.name,
                    available=False,
                    detail="runtime model listing unavailable",
                )
        installed_by_id = {item.model_id: item for item in installed}
        generation_ids = tuple(
            dict.fromkeys(
                (
                    self._generation_model,
                    *self._ollama_catalog,
                )
            )
        )
        descriptors = [
            self._descriptor(
                model_id=model_id,
                role=ModelRole.GENERATION,
                metadata=installed_by_id.get(model_id),
                installed=model_id in installed_by_id,
                source=(
                    "installed runtime"
                    if model_id in installed_by_id
                    else "configured/catalog"
                ),
                system=system,
                selected=model_id == self._generation_model,
                readiness=self._readiness_for(
                    model_id,
                    installed=model_id in installed_by_id,
                    runtime_available=runtime.available,
                ),
            )
            for model_id in generation_ids
        ]
        descriptors.extend(
            [
                self._descriptor(
                    model_id=self._embedding_model,
                    role=ModelRole.EMBEDDING,
                    metadata=None,
                    installed=True,
                    source="configured application adapter",
                    system=system,
                    selected=True,
                    role_compatibility="Configured embedding adapter; runtime installation is not applicable",
                ),
                self._descriptor(
                    model_id=self._sparse_model,
                    role=ModelRole.SPARSE,
                    metadata=None,
                    installed=True,
                    source="configured application implementation",
                    system=system,
                    selected=True,
                    role_compatibility="BM25 implementation, not an LLM",
                ),
                self._descriptor(
                    model_id=self._reranker_model,
                    role=ModelRole.RERANKER,
                    metadata=None,
                    installed=False,
                    source="configured sentence-transformer adapter",
                    system=system,
                    selected=False,
                    role_compatibility="Unverified until the adapter loads the model",
                ),
            ]
        )
        generation_installed = self._generation_model in installed_by_id
        probe_status, probe_reason = self._readiness_for(
            self._generation_model,
            installed=generation_installed,
            runtime_available=runtime.available,
        )
        if not runtime.available:
            generation_status = "runtime_unavailable"
        elif not generation_installed:
            generation_status = "model_missing"
        else:
            generation_status = "ready"
        return {
            "system": system.as_dict(),
            "runtime": {
                "name": runtime.name,
                "available": runtime.available,
                "detail": runtime.detail,
                "installed_count": runtime.installed_count,
            },
            "configured": {
                "generation": self._generation_model,
                "embedding": self._embedding_model,
                "sparse": self._sparse_model,
                "reranker": self._reranker_model,
            },
            "generation_readiness": {
                "model": self._generation_model,
                "status": generation_status,
            },
            "generation_probe": {
                "model": self._generation_model,
                "status": probe_status,
                "reason": probe_reason,
            },
            "index_compatibility": {
                "embedding_model": self._embedding_model,
                "embedding_dimension": self._embedding_dimension,
                "collection": self._qdrant_collection,
                "reindex_on_embedding_change": True,
                "message": "Embedding model changes require deliberate re-indexing",
            },
            "installed_models": [
                _metadata_dict(item) for item in installed
            ],
            "models": [descriptor.as_dict() for descriptor in descriptors],
        }

    async def pull_model(
        self,
        model_id: str,
        on_progress: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    ) -> None:
        """Delegate a validated local pull to the runtime adapter."""

        allowed = set(self._ollama_catalog) | {self._generation_model}
        if model_id not in allowed:
            raise ValueError("model is not in the configured local catalog")
        await self._runtime.pull_model(model_id, on_progress=on_progress)

    async def validate_model_pull(self, model_id: str) -> str:
        """Validate a pull before a background pull record is created."""

        allowed = set(self._ollama_catalog) | {self._generation_model}
        if model_id not in allowed:
            return "not_allowlisted"
        runtime = await self._runtime.check_runtime()
        return "ready" if runtime.available else "runtime_unavailable"

    async def validate_generation_model(self, model_id: str) -> str:
        """Validate a selected generation model before a demo query starts."""

        allowed = set(self._ollama_catalog) | {self._generation_model}
        if model_id not in allowed:
            return "not_allowlisted"
        runtime = await self._runtime.check_runtime()
        if not runtime.available:
            return "runtime_unavailable"
        try:
            installed = await self._runtime.list_installed_models()
        except Exception:
            return "runtime_unavailable"
        return "ready" if any(item.model_id == model_id for item in installed) else "model_missing"

    def _readiness_for(
        self,
        model_id: str,
        *,
        installed: bool,
        runtime_available: bool,
    ) -> tuple[str, str | None]:
        if not runtime_available:
            return "runtime_unavailable", "Ollama runtime is unavailable"
        if not installed:
            return "not_installed", "Model is not installed in the runtime"
        return self._generation_probes.get(
            model_id,
            ("installed_unverified", "Installed; no generation probe in this process"),
        )

    def _descriptor(
        self,
        *,
        model_id: str,
        role: ModelRole,
        metadata: ModelMetadata | None,
        installed: bool,
        source: str,
        system: SystemProfile,
        selected: bool,
        role_compatibility: str | None = None,
        readiness: tuple[str, str | None] = ("not_applicable", None),
    ) -> ModelDescriptor:
        return ModelDescriptor(
            model_id=model_id,
            role=role,
            installed=installed,
            source=source,
            metadata=metadata,
            compatibility=self._compatibility.estimate(
                system=system,
                metadata=metadata,
            ),
            role_compatibility=role_compatibility or "Generation model",
            selected=selected,
            readiness=readiness[0],
            readiness_reason=readiness[1],
        )


def _estimate_memory_gb(
    metadata: ModelMetadata,
    *,
    context_length: int,
) -> float | None:
    if metadata.size_bytes is not None and metadata.size_bytes > 0:
        # Ollama's blob size plus conservative runtime/KV headroom.
        base_memory = metadata.size_bytes / 1_073_741_824
        return round(base_memory * 1.15 + _kv_headroom_gb(context_length), 2)
    if metadata.parameter_count_b is not None and metadata.parameter_count_b > 0:
        quantization = (metadata.quantization or "").casefold()
        bits = 4 if "q4" in quantization else 8 if "q8" in quantization else None
        if bits is None:
            return None
        base_memory = metadata.parameter_count_b * bits / 8
        return round(base_memory * 1.15 + _kv_headroom_gb(context_length), 2)
    return None


def _kv_headroom_gb(context_length: int) -> float:
    """Use a small model-agnostic KV allowance without fake precision."""

    return min(4.0, max(0.25, context_length / 4096 * 0.25))


def _metadata_dict(metadata: ModelMetadata) -> dict[str, object]:
    return {
        "model_id": metadata.model_id,
        "size_bytes": metadata.size_bytes,
        "parameter_count_b": metadata.parameter_count_b,
        "quantization": metadata.quantization,
        "format": metadata.format,
        "family": metadata.family,
    }
