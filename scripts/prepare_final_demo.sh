#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

api_url="${DEMO_API_URL:-http://127.0.0.1:8010}"
tenant_id="${DEMO_TENANT_ID:-final-demo-v1}"
ollama_url="${DEMO_OLLAMA_URL:-http://127.0.0.1:11434}"
manifest="demo/final_demo_pack/demo_cases.json"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command not found: $1" >&2
    exit 1
  }
}

require_command curl
require_command python3

wait_for() {
  local url="$1"
  local attempts=0
  local max_attempts="${2:-60}"
  until curl --fail --silent --show-error --max-time 3 "$url" >/dev/null; do
    attempts=$((attempts + 1))
    if (( attempts >= max_attempts )); then
      echo "Timed out waiting for ${url}" >&2
      exit 1
    fi
    sleep 2
  done
}

echo "[1/5] Checking Week-2 readiness"
wait_for "${api_url}/v1/health/live" 60
wait_for "${api_url}/v1/health/ready" 120

if [[ "${DEMO_REBUILD_PDFS:-false}" == "true" ]]; then
  echo "[2/5] Rebuilding fictional Turkish PDFs"
  python3 scripts/generate_final_demo_pdfs.py
else
  echo "[2/5] Using committed fictional Turkish PDFs"
fi

echo "[3/5] Resetting only tenant ${tenant_id} demo documents"
python3 scripts/run_final_demo.py \
  --api "$api_url" \
  --tenant "$tenant_id" \
  --reset \
  --ingest-only

python3 - "$manifest" "demo/final_demo_pack/results/ingestion_receipts.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {item["filename"] for item in manifest["documents"]}
actual = {item["filename"] for item in receipts["receipts"]}
if actual != expected:
    raise SystemExit(f"demo receipt mismatch: expected {sorted(expected)}, got {sorted(actual)}")
for receipt in receipts["receipts"]:
    if not receipt.get("document_id") or not receipt.get("version_id"):
        raise SystemExit(f"missing active identity for {receipt['filename']}")
    if receipt.get("duplicate_receipt", {}).get("idempotent_hit") is not True:
        raise SystemExit(f"idempotency did not pass for {receipt['filename']}")
print(f"PASS · {len(actual)} PDFs ingested through POST /v1/documents")
print("PASS · duplicate upload reused the original document/version/job identity")
PY

echo "[4/5] Warming Gemma when the local Ollama endpoint is exposed"
warmup_payload='{"model":"gemma3:4b","prompt":"Sadece hazır yaz.","stream":false,"options":{"num_predict":4}}'
if curl --fail --silent --show-error --max-time 5 "${ollama_url}/api/tags" >/dev/null 2>&1; then
  warmup_response="$(curl --fail --silent --show-error --max-time 120 \
    -H 'Content-Type: application/json' \
    -d "$warmup_payload" \
    "${ollama_url}/api/generate")"
  python3 - "$warmup_response" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
if not str(payload.get("response", "")).strip():
    raise SystemExit("Gemma warm-up returned no final response")
print("PASS · Gemma warm-up produced a final response")
PY
else
  echo "WARNING · host Ollama endpoint is not exposed; readiness still verifies the configured runtime"
fi

echo "[5/5] Final readiness check"
ready="$(curl --fail --silent --show-error "${api_url}/v1/health/ready")"
python3 - "$ready" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
if payload.get("status") != "ready":
    raise SystemExit(f"Week-2 is not ready: {payload}")
print(payload.get("checks", {}).get("llm", {}).get("detail", "LLM readiness confirmed"))
PY
echo "Final demo corpus is ready in tenant ${tenant_id}."
