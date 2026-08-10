"""Atomic worker heartbeat writes for process-level health checks."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path


class WorkerHeartbeat:
    """Write a small shared-volume heartbeat without blocking ingestion."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def write(
        self,
        *,
        state: str,
        job_id: str | None = None,
    ) -> bool:
        """Atomically publish the current worker state."""

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "pid": os.getpid(),
            "job_id": job_id,
        }
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False,
                           separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, self._path)
        except OSError:
            return False
        return True
