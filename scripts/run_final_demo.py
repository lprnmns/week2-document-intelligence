#!/usr/bin/env python3
"""Prepare and measure the Turkish final demo corpus through the real API.

The script is intentionally an HTTP client. It never imports application
internals, writes Qdrant, changes V11 settings, or substitutes synthetic
retrieval results for the product pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "demo" / "final_demo_pack"
PDF_DIR = PACK / "pdfs"
MANIFEST_PATH = PACK / "demo_cases.json"
RESULT_DIR = PACK / "results"

DEFAULT_API = "http://127.0.0.1:8010"
DEFAULT_TENANT = "final-demo-v1"
DEFAULT_ACL = "public"


class ApiFailure(RuntimeError):
    """A bounded API failure with status and body for release diagnostics."""

    def __init__(self, method: str, path: str, status: int, body: str) -> None:
        super().__init__(f"{method} {path} failed with HTTP {status}: {body[:500]}")
        self.method = method
        self.path = path
        self.status = status
        self.body = body


def _bounded_artifact_text(value: str, limit: int = 1_600) -> str:
    """Keep committed demo evidence readable without duplicating giant parents."""

    if len(value) <= limit:
        return value
    head = max(1, int(limit * 0.72))
    tail = max(1, limit - head - 24)
    return f"{value[:head]} … [bounded] … {value[-tail:]}"


def _compact_artifact(value: Any, key: str | None = None) -> Any:
    """Bound repeated runtime text in the stored demo report, not the API trace."""

    if isinstance(value, str):
        limit = 900 if key in {"parent_context", "context_text"} else 1_600
        return _bounded_artifact_text(value, limit)
    if isinstance(value, list):
        return [_compact_artifact(item, key) for item in value]
    if isinstance(value, dict):
        return {name: _compact_artifact(item, name) for name, item in value.items()}
    return value


def now_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def normalize(value: str) -> str:
    lowered = value.casefold().replace("İ", "i")
    lowered = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", " ", lowered, flags=re.UNICODE)
    return " ".join(lowered.split())


def tokens(value: str) -> set[str]:
    # Keep short numeric fragments such as ``40.000`` after punctuation
    # normalization; they are high-confidence facts in this controlled label
    # resolver, not retrieval terms.
    return {token for token in normalize(value).split() if len(token) >= 2}


def api_call(
    api: str,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> Any:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request_body = body
    if payload is not None:
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(api.rstrip("/") + path, data=request_body, method=method)
    for key, value in request_headers.items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read()
    except HTTPError as error:
        raw = error.read()
        raise ApiFailure(
            method, path, error.code, raw.decode("utf-8", "replace")
        ) from error
    except URLError as error:
        raise RuntimeError(f"{method} {path} unavailable: {error.reason}") from error
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def scoped_headers(tenant: str) -> dict[str, str]:
    return {"X-Tenant-ID": tenant, "X-ACL-Tags": DEFAULT_ACL}


def list_documents(api: str, tenant: str) -> list[dict[str, Any]]:
    payload = api_call(
        api,
        "GET",
        "/v1/documents?limit=100",
        headers={"X-Tenant-ID": tenant},
    )
    return list(payload.get("items", []))


def delete_current_documents(api: str, tenant: str, filenames: set[str]) -> list[str]:
    deleted: list[str] = []
    for item in list_documents(api, tenant):
        if item.get("title") not in filenames:
            continue
        document_id = item.get("document_id")
        if not isinstance(document_id, str):
            continue
        api_call(
            api,
            "DELETE",
            f"/v1/documents/{document_id}",
            headers={"X-Tenant-ID": tenant},
        )
        deleted.append(document_id)
    return deleted


def multipart_pdf(path: Path, key: str, tenant: str) -> dict[str, Any]:
    boundary = f"----week2-final-demo-{uuid.uuid4().hex}"
    content = path.read_bytes()
    name = path.name.encode("utf-8")
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="',
            name,
            b'"\r\nContent-Type: application/pdf\r\n\r\n',
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    return api_call(
        DEFAULT_API,
        "POST",
        "/v1/documents",
        headers={
            **scoped_headers(tenant),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": key,
        },
        body=body,
    )


def upload_pdf(api: str, path: Path, key: str, tenant: str) -> dict[str, Any]:
    boundary = f"----week2-final-demo-{uuid.uuid4().hex}"
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
    return api_call(
        api,
        "POST",
        "/v1/documents",
        headers={
            **scoped_headers(tenant),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": key,
        },
        body=body,
    )


def wait_job(api: str, job_id: str, timeout_seconds: int = 900) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = api_call(api, "GET", f"/v1/jobs/{job_id}")
        status = payload.get("status")
        if status in {"succeeded", "failed"}:
            if status == "failed":
                raise RuntimeError(
                    f"ingestion failed for {job_id}: {payload.get('error_code')} "
                    f"{payload.get('error_message')}"
                )
            return payload
        time.sleep(1.5)
    raise TimeoutError(f"timed out waiting for ingestion job {job_id}")


def find_scalar(value: Any, wanted: set[str]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in wanted and isinstance(item, (str, int, float, bool)):
                found[key] = item
            found.update(find_scalar(item, wanted))
    elif isinstance(value, list):
        for item in value:
            found.update(find_scalar(item, wanted))
    return found


def ingest_corpus(
    api: str, tenant: str, manifest: dict[str, Any], reset: bool
) -> dict[str, Any]:
    filenames = {item["filename"] for item in manifest["documents"]}
    deleted = delete_current_documents(api, tenant, filenames) if reset else []
    run_token = now_token()
    receipts: list[dict[str, Any]] = []
    for document in manifest["documents"]:
        path = PDF_DIR / document["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"final-demo-{run_token}-{digest[:24]}"
        first = upload_pdf(api, path, key, tenant)
        job = wait_job(api, first["job_id"])
        duplicate = upload_pdf(api, path, key, tenant)
        if not duplicate.get("idempotent_hit"):
            raise RuntimeError(f"idempotency verification failed for {path.name}")
        if any(
            first.get(key_name) != duplicate.get(key_name)
            for key_name in ("document_id", "version_id", "job_id")
        ):
            raise RuntimeError(f"duplicate identity changed for {path.name}")
        metadata = find_scalar(
            job,
            {
                "pipeline_fingerprint",
                "parent_count",
                "child_count",
                "page_count",
                "point_count",
            },
        )
        receipts.append(
            {
                "key": document["key"],
                "filename": path.name,
                "sha256": digest,
                "first_receipt": first,
                "duplicate_receipt": duplicate,
                "job": job,
                "metadata_found_in_job": metadata,
                "document_id": first["document_id"],
                "version_id": first["version_id"],
                "page_count": job.get("page_count"),
                "point_count": job.get("point_count"),
                "pipeline_fingerprint": metadata.get("pipeline_fingerprint"),
                "parent_count": metadata.get("parent_count"),
                "child_count": metadata.get("child_count"),
            }
        )
        print(
            f"ingested {path.name}: {job.get('page_count')} pages, {job.get('point_count')} points"
        )
    return {
        "tenant_id": tenant,
        "run_token": run_token,
        "deleted_document_ids": deleted,
        "receipts": receipts,
        "documents": list_documents(api, tenant),
    }


def browse_chunks(
    api: str,
    tenant: str,
    document_id: str,
    page: int,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = urlencode(
        [("document_ids", document_id), ("page", str(page)), ("limit", str(limit))]
    )
    payload = api_call(
        api,
        "GET",
        f"/v1/demo/gold/evidence?{query}",
        headers=scoped_headers(tenant),
    )
    return list(payload.get("items", []))


def resolve_trusted_sources(
    api: str,
    tenant: str,
    manifest: dict[str, Any],
    ingestion: dict[str, Any],
) -> dict[str, Any]:
    by_key = {item["key"]: item for item in ingestion["receipts"]}
    resolutions: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        source_ids: list[str] = []
        matches: list[dict[str, Any]] = []
        facts = [tokens(item) for item in case.get("required_facts", [])]
        for index, document_key in enumerate(case.get("gold_documents", [])):
            receipt = by_key.get(document_key)
            if receipt is None:
                continue
            pages = case.get("gold_pages", [])
            page = int(pages[index]) if index < len(pages) else int(pages[0])
            chunks = browse_chunks(api, tenant, receipt["document_id"], page)
            if len(case.get("gold_documents", [])) == 1:
                relevant_tokens = set().union(*facts) if facts else set()
            else:
                relevant_tokens = facts[min(index, len(facts) - 1)] if facts else set()
            ranked: list[tuple[int, dict[str, Any]]] = []
            for chunk in chunks:
                child_text = str(chunk.get("chunk_text", ""))
                # Prefer the canonical child. Parent context is intentionally
                # not used for the first pass because it is bounded context
                # shared by many children and would create false gold labels.
                overlap = len(relevant_tokens & tokens(child_text))
                if overlap == 0:
                    overlap = len(relevant_tokens & tokens(str(chunk.get("title", ""))))
                ranked.append((overlap, chunk))
            ranked.sort(key=lambda pair: pair[0], reverse=True)
            if ranked and ranked[0][0] > 0:
                score, chosen = ranked[0]
                source_id = chosen.get("source_id")
                if isinstance(source_id, str):
                    source_ids.append(source_id)
                    matches.append(
                        {
                            "document_key": document_key,
                            "document_id": receipt["document_id"],
                            "page": page,
                            "source_id": source_id,
                            "token_overlap": score,
                            "chunk_text": chosen.get("chunk_text", ""),
                        }
                    )
        case["trusted_source_ids"] = list(dict.fromkeys(source_ids))
        case["trusted_source_resolution"] = matches
        resolutions.append(
            {
                "case_id": case["case_id"],
                "source_ids": case["trusted_source_ids"],
                "matches": matches,
                "warning": None
                if matches or not case.get("gold_documents")
                else "No matching source found",
            }
        )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"resolutions": resolutions}


def run_query(
    api: str,
    tenant: str,
    case: dict[str, Any],
    document_ids: list[str],
    *,
    reranker_enabled: bool,
    expected_answer: bool,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": case["question"],
        "document_ids": document_ids,
        "retrieval_mode": case.get("recommended_retrieval_mode", "hybrid"),
        # Keep the demo request bounded on local CPU while retaining the
        # production retrieval/fusion/reranker implementation. Multi-document
        # cases need two evidence slots; single-fact cases need one.
        "top_k": int(
            case.get("demo_top_k", 2 if len(case.get("gold_documents", [])) > 1 else 1)
        ),
        "reranker_enabled": reranker_enabled,
        "tenant_id": tenant,
        "acl_tags": [DEFAULT_ACL],
    }
    if expected_answer and case.get("expected_answer"):
        payload["expected_answer"] = case["expected_answer"]
    started = time.monotonic()
    start = api_call(
        api,
        "POST",
        "/v1/demo/query-runs",
        payload=payload,
        headers=scoped_headers(tenant),
    )
    run_id = start["run_id"]
    while time.monotonic() - started < timeout_seconds:
        snapshot = api_call(api, "GET", f"/v1/demo/query-runs/{run_id}")
        if snapshot.get("status") in {"completed", "failed"}:
            return snapshot
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for query run {run_id}: {case['case_id']}")


def attach_trusted_diagnostic(
    api: str,
    tenant: str,
    case: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    source_ids = case.get("trusted_source_ids", [])
    expected = case.get("expected_answer", "")
    if (
        not source_ids
        or not expected
        or snapshot.get("status") not in {"completed", "failed"}
    ):
        return None
    payload = {
        "source_ids": source_ids,
        "expected_answer": expected,
        "question": case["question"],
    }
    try:
        return api_call(
            api,
            "POST",
            f"/v1/demo/query-runs/{snapshot['run_id']}/trusted-evidence",
            payload=payload,
            headers=scoped_headers(tenant),
        )
    except ApiFailure as error:
        # A failed generation run may not have a completed result object for
        # the legacy merge endpoint. Preserve the real run and report the
        # diagnostic transport limitation instead of hiding the failure.
        return {
            "status": "TRUSTED_DIAGNOSTIC_UNAVAILABLE",
            "http_status": error.status,
            "message": error.body[:500],
        }


def compact_snapshot(
    snapshot: dict[str, Any], diagnostic: dict[str, Any] | None
) -> dict[str, Any]:
    result = snapshot.get("result") if isinstance(snapshot.get("result"), dict) else {}
    raw_error = (
        snapshot.get("error") if isinstance(snapshot.get("error"), dict) else None
    )
    error = dict(raw_error) if raw_error is not None else None
    if error is not None:
        error.pop("expected_check", None)
    expected_check = (
        result.get("expected_check")
        if result
        else (raw_error.get("expected_check") if raw_error else None)
    )
    return {
        "run_id": snapshot.get("run_id"),
        "request_id": snapshot.get("request_id"),
        "status": snapshot.get("status"),
        "decision": result.get("decision") if result else None,
        "answer": result.get("answer") if result else None,
        "no_answer_reason": result.get("no_answer_reason") if result else None,
        "error": error,
        "events": _compact_artifact(snapshot.get("events", []), "events"),
        "result": _compact_artifact(result, "result"),
        "expected_check": expected_check,
        "trusted_diagnostic": _compact_artifact(diagnostic, "trusted_diagnostic"),
    }


def write_results(
    ingestion: dict[str, Any],
    resolutions: dict[str, Any],
    runs: list[dict[str, Any]],
    reranker_runs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "v11_behavior_unchanged": True,
        "api": DEFAULT_API,
        "tenant_id": ingestion["tenant_id"],
        "corpus": {
            "document_count": len(ingestion["receipts"]),
            "documents": ingestion["receipts"],
            "resolutions": resolutions,
        },
        "runs": runs,
        "reranker_ablation": reranker_runs,
        "case_count": len(manifest["cases"]),
    }
    (RESULT_DIR / "ingestion_receipts.json").write_text(
        json.dumps(ingestion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (RESULT_DIR / "demo_run_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Demo run results",
        "",
        f"- Generated: `{output['generated_at']}`",
        f"- Tenant: `{ingestion['tenant_id']}`",
        f"- Corpus: {len(ingestion['receipts'])} PDF",
        f"- Cases: {len(runs)}",
        "- Pipeline: normal `POST /v1/documents` + real V11 demo query path",
        "",
        "## Ingestion",
        "",
        "| PDF | Pages | Points | Active version | Idempotency |",
        "|---|---:|---:|---|---|",
    ]
    for receipt in ingestion["receipts"]:
        lines.append(
            f"| {receipt['filename']} | {receipt.get('page_count') or '-'} | "
            f"{receipt.get('point_count') or '-'} | `{receipt.get('version_id')}` | PASS |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Expected | Actual | Decision | Time |",
            "|---|---|---|---|---:|",
        ]
    )
    for item in runs:
        expected = next(
            (
                case["expected_decision"]
                for case in manifest["cases"]
                if case["case_id"] == item["case_id"]
            ),
            "",
        )
        actual = item.get("decision") or (item.get("error") or {}).get("reason") or "-"
        latency = (item.get("result") or {}).get("latency", {}).get("total_ms")
        lines.append(
            f"| {item['case_id']} | {expected} | {actual} | {item['status']} | {latency or '-'} ms |"
        )
    lines.extend(
        [
            "",
            "## Reranker ablation",
            "",
            "Two live Hybrid RRF comparisons are stored in `demo_run_results.json`.",
            "",
        ]
    )
    (RESULT_DIR / "DEMO_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--tenant", default=DEFAULT_TENANT)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Upload/verify the demo PDFs and resolve labels without running queries.",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ingestion = ingest_corpus(args.api, args.tenant, manifest, args.reset)
    resolutions = resolve_trusted_sources(args.api, args.tenant, manifest, ingestion)
    if args.ingest_only:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "ingestion_receipts.json").write_text(
            json.dumps(ingestion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (RESULT_DIR / "trusted_source_resolutions.json").write_text(
            json.dumps(resolutions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved ingestion receipts to {RESULT_DIR / 'ingestion_receipts.json'}")
        return
    document_ids = [receipt["document_id"] for receipt in ingestion["receipts"]]
    runs: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        print(f"running {case['case_id']} ({case['question']})")
        snapshot = run_query(
            args.api,
            args.tenant,
            case,
            document_ids,
            reranker_enabled=bool(case.get("recommended_reranker", False)),
            expected_answer=bool(case.get("expected_answer")),
        )
        diagnostic = attach_trusted_diagnostic(args.api, args.tenant, case, snapshot)
        item = {"case_id": case["case_id"], **compact_snapshot(snapshot, diagnostic)}
        runs.append(item)
        error = item.get("error") or {}
        print(
            f"  -> {item.get('decision') or error.get('reason') or error.get('code') or item.get('status')}"
        )
    ablation_cases = [
        case
        for case in manifest["cases"]
        if case["case_id"] in {"release_checklist", "release_time_and_rollback"}
    ]
    reranker_runs: list[dict[str, Any]] = []
    for case in ablation_cases:
        print(f"reranker ON {case['case_id']}")
        snapshot = run_query(
            args.api,
            args.tenant,
            case,
            document_ids,
            reranker_enabled=True,
            expected_answer=bool(case.get("expected_answer")),
        )
        diagnostic = attach_trusted_diagnostic(args.api, args.tenant, case, snapshot)
        reranker_runs.append(
            {"case_id": case["case_id"], **compact_snapshot(snapshot, diagnostic)}
        )
    write_results(ingestion, resolutions, runs, reranker_runs, manifest)
    print(f"saved measured results to {RESULT_DIR / 'demo_run_results.json'}")


if __name__ == "__main__":
    main()
