#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

bundled_ollama=false
while (($# > 0)); do
  case "$1" in
    --bundled-ollama)
      bundled_ollama=true
      ;;
    --host-ollama)
      bundled_ollama=false
      ;;
    -h|--help)
      echo "Usage: $0 [--bundled-ollama|--host-ollama]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--bundled-ollama|--host-ollama]" >&2
      exit 2
      ;;
  esac
  shift
done

compose=(docker compose -f compose.yaml)
if [[ "$bundled_ollama" == true ]]; then
  compose+=( -f compose.ollama.yaml --profile bundled-ollama )
  export DIS_OLLAMA_URL="http://ollama:11434"
fi

api_host_port="${API_HOST_PORT:-8010}"
qdrant_host_port="${QDRANT_HOST_PORT:-6335}"
ui_host_port="${UI_HOST_PORT:-8501}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_command docker
require_command curl

if [[ -z "${DIS_SOURCE_REVISION:-}" ]]; then
  DIS_SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
  export DIS_SOURCE_REVISION
fi

"${compose[@]}" config --quiet
echo "Starting Week-2 services..."
"${compose[@]}" up --build -d

wait_for_http() {
  local url="$1"
  local attempts=0
  local max_attempts="${2:-90}"
  until curl --fail --silent --show-error "$url" >/dev/null; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -ge "$max_attempts" ]]; then
      return 1
    fi
    sleep 2
  done
}

if ! wait_for_http "http://127.0.0.1:${api_host_port}/v1/health/live" 90; then
  echo "API liveness did not become available." >&2
  "${compose[@]}" ps >&2 || true
  exit 1
fi

# Readiness checks Qdrant, Ollama, the selected model and the worker heartbeat.
if ! wait_for_http "http://127.0.0.1:${api_host_port}/v1/health/ready" 120; then
  echo "Week-2 is not ready. The UI was not reported as usable." >&2
  echo "Readiness response:" >&2
  curl --silent "http://127.0.0.1:${api_host_port}/v1/health/ready" >&2 || true
  printf '\nCompose status:\n' >&2
  "${compose[@]}" ps >&2 || true
  printf '\nRecent API/worker logs:\n' >&2
  "${compose[@]}" logs --tail=80 api worker >&2 || true
  echo >&2
  if [[ "$bundled_ollama" == true ]]; then
    echo "Common fix: inspect 'ollama' and 'ollama-model' logs; the first model pull may take several minutes." >&2
  else
    echo "Common fix: make host Ollama reachable from Docker and install gemma3:4b." >&2
    echo "Linux example: OLLAMA_HOST=0.0.0.0:11434 ollama serve" >&2
  fi
  exit 1
fi

if ! wait_for_http "http://127.0.0.1:${ui_host_port}/" 60; then
  echo "Demo UI did not become available." >&2
  "${compose[@]}" ps >&2 || true
  exit 1
fi

echo "Week-2 is ready."
if [[ "$bundled_ollama" == true ]]; then
  echo "Ollama:  bundled Docker service (${DIS_LLM_MODEL:-gemma3:4b})"
else
  echo "Ollama:  host runtime"
fi
echo "API:     http://127.0.0.1:${api_host_port}"
echo "Health:  http://127.0.0.1:${api_host_port}/v1/health/ready"
echo "Qdrant:  http://127.0.0.1:${qdrant_host_port}"
echo "Demo UI: http://127.0.0.1:${ui_host_port}"
