"""Model role and compatibility value objects."""

from dataclasses import dataclass
from enum import StrEnum


class ModelRole(StrEnum):
    """Distinct model responsibilities in the document service."""

    GENERATION = "generation"
    EMBEDDING = "embedding"
    SPARSE = "sparse"
    RERANKER = "reranker"


class CompatibilityClass(StrEnum):
    """Heuristic local resource estimate, never a runtime guarantee."""

    RECOMMENDED = "recommended"
    LIKELY_USABLE = "likely_usable"
    MAY_RUN_SLOWLY = "may_run_slowly"
    MEMORY_RISK = "memory_risk"
    NOT_RECOMMENDED = "not_recommended"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Runtime metadata returned by a local model runtime."""

    model_id: str
    size_bytes: int | None = None
    parameter_count_b: float | None = None
    quantization: str | None = None
    format: str | None = None
    family: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Reachability state distinct from an individual model's installation."""

    name: str
    available: bool
    detail: str | None = None
    installed_count: int = 0


@dataclass(frozen=True, slots=True)
class ModelCompatibility:
    """Transparent, approximate compatibility result."""

    classification: CompatibilityClass
    label: str
    reason: str
    estimated_memory_gb: float | None
    safe_memory_budget_gb: float | None
    confidence: str


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """A model candidate with role and runtime state separated."""

    model_id: str
    role: ModelRole
    installed: bool
    source: str
    metadata: ModelMetadata | None
    compatibility: ModelCompatibility
    role_compatibility: str
    selected: bool = False
    readiness: str = "not_applicable"
    readiness_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the safe UI/API representation."""

        metadata = self.metadata
        return {
            "model_id": self.model_id,
            "role": self.role.value,
            "installed": self.installed,
            "source": self.source,
            "selected": self.selected,
            "readiness": self.readiness,
            "readiness_reason": self.readiness_reason,
            "metadata": (
                {
                    "size_bytes": metadata.size_bytes,
                    "parameter_count_b": metadata.parameter_count_b,
                    "quantization": metadata.quantization,
                    "format": metadata.format,
                    "family": metadata.family,
                }
                if metadata is not None
                else None
            ),
            "compatibility": {
                "classification": self.compatibility.classification.value,
                "label": self.compatibility.label,
                "reason": self.compatibility.reason,
                "estimated_memory_gb": self.compatibility.estimated_memory_gb,
                "safe_memory_budget_gb": self.compatibility.safe_memory_budget_gb,
                "confidence": self.compatibility.confidence,
            },
            "role_compatibility": self.role_compatibility,
        }
