#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

api_url="${DEMO_API_URL:-http://127.0.0.1:8010}"
tenant_id="${DEMO_TENANT_ID:-final-demo-v1}"
manifest="demo/final_demo_pack/demo_cases.json"
receipts="demo/final_demo_pack/results/ingestion_receipts.json"
results="demo/final_demo_pack/results/demo_run_results.json"

echo "[1/4] Service and UI pre-flight"
curl --fail --silent --show-error "${api_url}/v1/health/live" >/dev/null
ready="$(curl --fail --silent --show-error "${api_url}/v1/health/ready")"
curl --fail --silent --show-error -I http://127.0.0.1:8501/ >/dev/null

echo "[2/4] Corpus and idempotency receipts"
python3 - "$manifest" "$receipts" "$tenant_id" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if receipts.get("tenant_id") != sys.argv[3]:
    raise SystemExit(f"unexpected demo tenant: {receipts.get('tenant_id')}")
expected = {item["filename"] for item in manifest["documents"]}
actual = {item["filename"] for item in receipts.get("receipts", [])}
if expected != actual:
    raise SystemExit(f"PDF set mismatch: {sorted(expected)} != {sorted(actual)}")
for receipt in receipts["receipts"]:
    first = receipt.get("first_receipt", {})
    duplicate = receipt.get("duplicate_receipt", {})
    if duplicate.get("idempotent_hit") is not True:
        raise SystemExit(f"idempotency failed for {receipt['filename']}")
    for key in ("document_id", "version_id", "job_id"):
        if first.get(key) != duplicate.get(key):
            raise SystemExit(f"identity changed for {receipt['filename']}: {key}")
    if not receipt.get("pipeline_fingerprint"):
        raise SystemExit(f"missing pipeline fingerprint for {receipt['filename']}")
print(f"PASS · {len(actual)} fictional PDFs, active identities and idempotency")
PY

echo "[3/4] Best-demo-6 result pre-flight"
python3 - "$manifest" "$results" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
by_id = {item["case_id"]: item for item in results.get("runs", [])}
best = [
    "semantic_remote_days",
    "exact_rollback_code",
    "release_time_and_rollback",
    "education_2024_absent",
    "missing_rd_budget",
    "security_injection",
]
manifest_ids = {item["case_id"] for item in manifest["cases"]}
missing = [case_id for case_id in best if case_id not in manifest_ids or case_id not in by_id]
if missing:
    raise SystemExit(f"missing measured Best Demo 6 cases: {missing}")
for case_id in best:
    item = by_id[case_id]
    decision = item.get("decision") or (item.get("error") or {}).get("reason") or "PIPELINE_FAILED"
    print(f"{case_id}: {decision} · request_id={item.get('request_id') or '—'}")
    if not item.get("events"):
        raise SystemExit(f"no trace events recorded for {case_id}")
no_answer = {case_id: by_id[case_id] for case_id in ("education_2024_absent", "missing_rd_budget", "security_injection")}
for case_id, item in no_answer.items():
    stages = [event.get("stage") for event in item.get("events", []) if event.get("status") == "passed"]
    if "llm" in stages:
        raise SystemExit(f"LLM unexpectedly passed for no-answer case {case_id}")
print("PASS · no-answer cases retained an application trace without a successful LLM stage")
print("WARNING · this script reports measured failures honestly; it does not retune or rerun them")
PY

echo "[4/4] Artifact hygiene"
if rg -n --fixed-strings '[object Object]' demo/final_demo_pack demo_ui >/dev/null; then
  echo "Found frontend serialization noise" >&2
  exit 1
fi
if find demo/final_demo_pack/pdfs -maxdepth 1 -type f ! -name 'nova_*.pdf' -print -quit | rg . >/dev/null; then
  echo "Unexpected non-NOVA PDF in final demo corpus" >&2
  exit 1
fi
echo "PASS · no [object Object] and only fictional NOVA demo PDFs are in the pack"
echo "Demo verification complete for tenant ${tenant_id}."
