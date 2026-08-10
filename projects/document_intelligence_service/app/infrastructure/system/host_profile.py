"""Sanitized host profile detection for local model recommendations."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess

from ...domain.system_profile import SystemProfile


class HostProfileAdapter:
    """Inspect only operational hardware facts needed for model selection."""

    def detect(self) -> SystemProfile:
        """Return a best-effort profile; unknown facts stay ``None``."""

        cpu_model, physical_cores = _linux_cpu_info()
        logical_cores = os.cpu_count()
        total_ram_gb, available_ram_gb = _linux_memory()
        gpu_name, gpu_vram_gb, gpu_acceleration = _detect_gpu()
        acceleration = list(gpu_acceleration)
        if platform.system() == "Darwin":
            acceleration.append("metal")
        return SystemProfile(
            operating_system=platform.system() or "unknown",
            architecture=platform.machine() or "unknown",
            cpu_model=cpu_model or platform.processor() or None,
            logical_cores=logical_cores,
            physical_cores=physical_cores,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_available=gpu_name is not None,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            acceleration=tuple(dict.fromkeys(acceleration)),
            containerized=Path("/.dockerenv").is_file(),
        )


def _linux_cpu_info() -> tuple[str | None, int | None]:
    """Read model/physical topology without exposing the file path."""

    path = Path("/proc/cpuinfo")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    model: str | None = None
    topology: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in text.splitlines() + [""]:
        if line.startswith("model name") and model is None:
            model = line.split(":", 1)[-1].strip() or None
        elif line.startswith("Hardware") and model is None:
            model = line.split(":", 1)[-1].strip() or None
        elif line.startswith("physical id"):
            physical_id = line.split(":", 1)[-1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[-1].strip()
        elif not line.strip():
            if physical_id is not None and core_id is not None:
                topology.add((physical_id, core_id))
            physical_id = None
            core_id = None
    return model, len(topology) or None


def _linux_memory() -> tuple[float | None, float | None]:
    """Read MemTotal/MemAvailable when the host exposes procfs."""

    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None, None
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(":")] = float(parts[1]) / 1_048_576
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return _round_gb(total), _round_gb(available)


def _detect_gpu() -> tuple[str | None, float | None, tuple[str, ...]]:
    """Use fixed, read-only vendor commands; never accept user-supplied args."""

    if shutil.which("nvidia-smi"):
        output = _run_fixed(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if output:
            first = output.splitlines()[0].split(",", 1)
            name = first[0].strip() or None
            raw_vram = _parse_float(first[1]) if len(first) > 1 else None
            vram = raw_vram / 1024 if raw_vram is not None else None
            return name, _round_gb(vram), ("cuda",)
    if shutil.which("rocm-smi"):
        output = _run_fixed(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
        if output:
            return "AMD GPU", None, ("rocm",)
    return None, None, ()


def _run_fixed(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _round_gb(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None
