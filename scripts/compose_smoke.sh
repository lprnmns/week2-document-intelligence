#!/usr/bin/env bash

set -euo pipefail

compose=(docker compose -f compose.yaml)
if [[ "${BUNDLED_OLLAMA:-false}" == "true" ]]; then
  compose+=( -f compose.ollama.yaml --profile bundled-ollama )
  export DIS_OLLAMA_URL="http://ollama:11434"
fi
qdrant_host_port="${QDRANT_HOST_PORT:-6335}"
api_host_port="${API_HOST_PORT:-8010}"
ui_host_port="${UI_HOST_PORT:-8501}"
demo_collection="${DIS_QDRANT_COLLECTION:-document_chunks_week2_final_v1}"

if [[ -z "${DIS_SOURCE_REVISION:-}" ]]; then
  DIS_SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
  export DIS_SOURCE_REVISION
fi

"${compose[@]}" config --quiet
"${compose[@]}" up --build -d qdrant

cleanup() {
  "${compose[@]}" down --remove-orphans >/dev/null
}
trap cleanup EXIT

"${compose[@]}" up --build -d api worker demo-ui

wait_for() {
  local url="$1"
  local attempts=0
  until curl --fail --silent --show-error "$url" >/dev/null; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -ge 60 ]]; then
      echo "Timed out waiting for $url" >&2
      "${compose[@]}" ps
      exit 1
    fi
    sleep 2
  done
}

wait_for "http://127.0.0.1:${api_host_port}/v1/health/live"
wait_for "http://127.0.0.1:${ui_host_port}/"
wait_for "http://127.0.0.1:${qdrant_host_port}/readyz"

worker_attempts=0
until "${compose[@]}" ps --status running --services | rg -qx "worker"; do
  worker_attempts=$((worker_attempts + 1))
  if [[ "$worker_attempts" -ge 60 ]]; then
    echo "Timed out waiting for the ingestion worker" >&2
    "${compose[@]}" ps
    exit 1
  fi
  sleep 2
done

sample_pdf="${SMOKE_PDF:-}"
if [[ -n "$sample_pdf" && -f "$sample_pdf" ]]; then
  sample_digest="$(sha256sum "$sample_pdf" | awk '{print $1}')"
  smoke_idempotency_key="${SMOKE_IDEMPOTENCY_KEY:-compose-smoke-v3-${sample_digest:0:24}}"
  receipt="$(curl --fail --silent --show-error \
    -H "Idempotency-Key: ${smoke_idempotency_key}" \
    -H 'X-Tenant-ID: default' \
    -H 'X-ACL-Tags: public' \
    -F "file=@${sample_pdf};type=application/pdf" \
    "http://127.0.0.1:${api_host_port}/v1/documents")"
  job_id="$(printf '%s' "$receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
  job_attempts=0
  while true; do
    job="$(curl --fail --silent --show-error "http://127.0.0.1:${api_host_port}/v1/jobs/${job_id}")"
    job_status="$(printf '%s' "$job" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
    if [[ "$job_status" == "succeeded" ]]; then
      break
    fi
    if [[ "$job_status" == "failed" ]]; then
      printf '%s\n' "$job" >&2
      exit 1
    fi
    job_attempts=$((job_attempts + 1))
    if [[ "$job_attempts" -ge 180 ]]; then
      echo "Timed out waiting for smoke ingestion job ${job_id}" >&2
      exit 1
    fi
    sleep 2
  done

  first_document_id="$(printf '%s' "$receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')"
  first_version_id="$(printf '%s' "$receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["version_id"])')"
  first_job_id="$(printf '%s' "$receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
  points_before_duplicate="$(curl --fail --silent --show-error \
    "http://127.0.0.1:${qdrant_host_port}/collections/${demo_collection}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"].get("points_count", 0))')"
  duplicate_receipt="$(curl --fail --silent --show-error \
    -H "Idempotency-Key: ${smoke_idempotency_key}" \
    -H 'X-Tenant-ID: default' \
    -H 'X-ACL-Tags: public' \
    -F "file=@${sample_pdf};type=application/pdf" \
    "http://127.0.0.1:${api_host_port}/v1/documents")"
  duplicate_document_id="$(printf '%s' "$duplicate_receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')"
  duplicate_version_id="$(printf '%s' "$duplicate_receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["version_id"])')"
  duplicate_job_id="$(printf '%s' "$duplicate_receipt" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
  duplicate_hit="$(printf '%s' "$duplicate_receipt" | python3 -c 'import json,sys; print(str(json.load(sys.stdin)["idempotent_hit"]).lower())')"
  points_after_duplicate="$(curl --fail --silent --show-error \
    "http://127.0.0.1:${qdrant_host_port}/collections/${demo_collection}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"].get("points_count", 0))')"
  if [[ "$duplicate_hit" != "true" || "$first_document_id" != "$duplicate_document_id" || "$first_version_id" != "$duplicate_version_id" || "$first_job_id" != "$duplicate_job_id" ]]; then
    echo "Duplicate ingestion identity was not reused" >&2
    printf '%s\n' "$duplicate_receipt" >&2
    exit 1
  fi
  if [[ "$points_before_duplicate" != "$points_after_duplicate" ]]; then
    echo "Duplicate upload changed Qdrant point count: ${points_before_duplicate} -> ${points_after_duplicate}" >&2
    exit 1
  fi
elif [[ -n "$sample_pdf" ]]; then
  echo "SMOKE_PDF does not exist: ${sample_pdf}" >&2
  exit 1
else
  echo "Skipping sample PDF ingestion: set SMOKE_PDF to a local parseable PDF to exercise ingestion"
fi

# Readiness includes the selected Ollama runtime. A 503 is a real environment
# failure, not a reason to pretend that the query path is ready.
wait_for "http://127.0.0.1:${api_host_port}/v1/health/ready"

collection_snapshot() {
  curl --fail --silent --show-error "http://127.0.0.1:${qdrant_host_port}/collections" \
    | python3 -c 'import json, sys; payload=json.load(sys.stdin); print(json.dumps(sorted(item["name"] for item in payload.get("result", {}).get("collections", []))))'
}

before="$(collection_snapshot)"
before_points="$(curl --fail --silent --show-error "http://127.0.0.1:${qdrant_host_port}/collections/${demo_collection}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"].get("points_count", 0))')"
"${compose[@]}" restart qdrant >/dev/null
wait_for "http://127.0.0.1:${qdrant_host_port}/readyz"
after="$(collection_snapshot)"
after_points="$(curl --fail --silent --show-error "http://127.0.0.1:${qdrant_host_port}/collections/${demo_collection}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"].get("points_count", 0))')"

if [[ "$before" != "$after" ]]; then
  echo "Qdrant collection snapshot changed after restart" >&2
  exit 1
fi

if [[ "$before_points" != "$after_points" ]]; then
  echo "Qdrant point count changed after restart: ${before_points} -> ${after_points}" >&2
  exit 1
fi

echo "Compose smoke passed: live, ready, worker, demo UI, ingestion idempotency, and Qdrant restart persistence."
