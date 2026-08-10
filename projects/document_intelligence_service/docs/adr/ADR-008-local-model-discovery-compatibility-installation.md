# ADR-008: Local model discovery, compatibility estimation and controlled installation

## Decision

Add a small system/model application service backed by two infrastructure
ports: a sanitized host profile adapter and an Ollama runtime adapter. Expose
hardware facts, runtime reachability, installed models and role metadata to the
local Demo UI. Allow model pull only when explicitly enabled and only for a
configured/validated model identifier.

## Alternatives

1. Hard-code model names and hardware conclusions in frontend JavaScript.
2. Run arbitrary `ollama` shell commands from an API endpoint.
3. Build a full model marketplace/monitoring dashboard.
4. Keep model/runtime details invisible and let query failures reveal missing
   configuration later.

## Reason

The service is local-first and resource-constrained. Hardware-aware guidance
helps prevent obviously unsuitable choices, while explicit role separation
prevents confusing the generation LLM with dense, BM25 or reranker components.
The runtime adapter boundary keeps host/container concerns out of the domain
and eliminates arbitrary shell execution.

Only operationally useful hardware is collected: OS/architecture, CPU,
cores, memory, GPU/VRAM, acceleration and container context. Usernames, paths,
environment variables and secrets are intentionally excluded. Compatibility
is a deterministic heuristic with uncertainty labels, not a promise.

## Measurement / evidence

- `tests/unit/test_model_service.py` checks deterministic estimates, unknown
  metadata and runtime-available/model-missing distinction.
- `tests/unit/test_model_runtime.py` checks shell-like and non-allowlisted model
  IDs are rejected before network pull.
- `/v1/system/profile` exposes the sanitized profile and actual installed model
  listing.
- Ollama `/api/pull` progress is forwarded only from the runtime response.

## Known limitation

The estimate cannot know the exact quantized implementation, context/KV cache,
offload strategy, concurrent load or other processes. The reranker and dense
models may be managed by sentence-transformers rather than Ollama, so their
runtime installation state remains adapter-specific/unverified. Embedding model
changes still require a deliberate re-index flow; this extension does not
silently migrate an existing Qdrant collection.

## Numbering note

The repository already uses ADR-006 for the durable ingestion registry and
ADR-007 for the observability boundary. This local-model decision is therefore
recorded as ADR-008 instead of overwriting an existing decision.
