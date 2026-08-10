"""Sanitized host profile value objects for local-first model decisions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemProfile:
    """Operational hardware facts safe to expose to a local demo."""

    operating_system: str
    architecture: str
    cpu_model: str | None
    logical_cores: int | None
    physical_cores: int | None
    total_ram_gb: float | None
    available_ram_gb: float | None
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    acceleration: tuple[str, ...]
    containerized: bool

    def as_dict(self) -> dict[str, object]:
        """Return a stable, sanitized JSON projection."""

        return {
            "operating_system": self.operating_system,
            "architecture": self.architecture,
            "cpu": {
                "model": self.cpu_model,
                "cores": self.physical_cores,
                "threads": self.logical_cores,
            },
            "memory": {
                "total_gb": self.total_ram_gb,
                "available_gb": self.available_ram_gb,
            },
            "gpu": {
                "available": self.gpu_available,
                "name": self.gpu_name,
                "vram_gb": self.gpu_vram_gb,
            },
            "acceleration": list(self.acceleration),
            "containerized": self.containerized,
        }
