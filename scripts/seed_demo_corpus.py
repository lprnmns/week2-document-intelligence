#!/usr/bin/env python3
"""Seed the fictional NOVA demo PDFs through the normal ingestion API.

This is a Compose init step, not a second ingestion implementation. It uses
the same ``POST /v1/documents`` contract as the UI and waits for each normal
worker job to finish. Existing active documents are left untouched so a
restart is safe and does not reset a tenant or rewrite retrieval state.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = os.environ.get("DEMO_SEED_API_URL", "http://api:8000").rstrip("/")
TENANT_ID = os.environ.get("DEMO_SEED_TENANT_ID", "final-demo-v1")
ACL_TAGS = os.environ.get("DEMO_SEED_ACL_TAGS", "public")
PDF_DIR = Path(os.environ.get("DEMO_SEED_PDF_DIR", "/app/demo_pdfs"))
ENABLED = os.environ.get("DEMO_SEED_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class SeedError(RuntimeError):
    """A bounded, operator-readable seed failure."""


def api_request(
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    request = Request(API_URL + path, data=body, method=method)
    request.add_header("Accept", "application/json")
    request.add_header("X-Tenant-ID", TENANT_ID)
    request.add_header("X-ACL-Tags", ACL_TAGS)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as error:
        raw = error.read()
        detail = raw.decode("utf-8", "replace")[:500]
        raise SeedError(f"{method} {path} -> HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise SeedError(f"{method} {path} unavailable: {error.reason}") from error
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def wait_for_ready(timeout_seconds: int = 600) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            payload = api_request("GET", "/v1/health/ready", timeout=5)
            if payload.get("status") == "ready":
                return
        except SeedError:
            pass
        time.sleep(2)
    raise SeedError(f"API did not become ready within {timeout_seconds}s")


def list_documents() -> list[dict[str, Any]]:
    payload = api_request("GET", "/v1/documents?limit=100")
    return list(payload.get("items", []))


def is_active(item: dict[str, Any]) -> bool:
    return item.get("status") == "active" and bool(item.get("active_version_id"))


def multipart_pdf(path: Path, idempotency_key: str) -> dict[str, Any]:
    boundary = f"----week2-demo-seed-{hashlib.sha256(path.name.encode()).hexdigest()[:16]}"
    content = path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="',
            path.name.encode("utf-8"),
            b'"\r\nContent-Type: application/pdf\r\n\r\n',
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    response = api_request(
        "POST",
        "/v1/documents",
        body=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": idempotency_key,
        },
        timeout=60,
    )
    if not isinstance(response, dict):
        raise SeedError(f"unexpected upload response for {path.name}")
    return response


def wait_for_job(job_id: str, path: Path, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = api_request("GET", f"/v1/jobs/{job_id}")
        status = payload.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            code = payload.get("error_code", "UNKNOWN")
            message = payload.get("error_message", "no detail")
            raise SeedError(f"ingestion failed for {path.name}: {code} - {message}")
        time.sleep(2)
    raise SeedError(f"ingestion timed out for {path.name}")


def seed() -> None:
    if not ENABLED:
        print("Demo corpus seed disabled (DEMO_SEED_ENABLED=false).")
        return

    paths = sorted(PDF_DIR.glob("nova_*.pdf"))
    if not paths:
        raise SeedError(f"no NOVA demo PDFs found in {PDF_DIR}")

    wait_for_ready()
    catalog = list_documents()
    active_by_title = {
        str(item.get("title")): item for item in catalog if is_active(item)
    }
    print(f"Demo seed target: tenant={TENANT_ID}, PDFs={len(paths)}")

    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        active = active_by_title.get(path.name)
        if active and active.get("content_hash") == digest:
            print(f"SKIP · {path.name} already has the current active version")
            continue
        key = f"week2-demo-seed-v1-{digest[:48]}"
        receipt = multipart_pdf(path, key)
        job_id = receipt.get("job_id")
        if isinstance(job_id, str) and job_id:
            wait_for_job(job_id, path)
        elif not receipt.get("idempotent_hit"):
            raise SeedError(f"upload response has no job_id for {path.name}")
        print(f"INGESTED · {path.name}")

    final_catalog = list_documents()
    missing = [
        path.name
        for path in paths
        if not any(
            item.get("title") == path.name and is_active(item)
            for item in final_catalog
        )
    ]
    if missing:
        raise SeedError(f"demo seed did not produce active documents: {missing}")
    print(f"READY · {len(paths)} NOVA demo PDFs are active in tenant {TENANT_ID}")


if __name__ == "__main__":
    try:
        seed()
    except SeedError as error:
        print(f"DEMO SEED FAILED · {error}")
        raise SystemExit(1) from error
