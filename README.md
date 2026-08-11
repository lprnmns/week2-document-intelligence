# Document Intelligence — Week 2 AI Engineering Project

This repository is a clean, standalone copy of the Week-2 Document
Intelligence service. It is a local-first PDF intelligence system designed to
make a wrong answer diagnosable: the trace separates ingestion, retrieval,
fusion, reranking, answerability, prompt construction and generation.

No private development PDFs, model weights, Qdrant databases, credentials or
parent-repository history are included.

## What the project demonstrates

The service accepts arbitrary parseable PDFs, indexes them with a deterministic
pipeline, and answers only when the retrieved evidence passes the configured
answerability policy. Every answer has application-generated canonical source
metadata; source cards are never reconstructed from model text.

The core flow is:

```text
PDF
 → parse → normalize → AUTO chunk selection
 → dense + BM25 → RRF
 → optional reranker → canonical evidence
 → answerability → structured prompt → local LLM
 → answer + canonical sources
```

The Query Trace runs one retrieval strategy at a time: Dense only, BM25 only,
or Hybrid RRF. Hybrid means `Dense + BM25 → RRF`; Evaluation is the separate
place where multiple strategies and reranker states are compared.

## Measured engineering decisions

The final frozen mentor corpus contains 26 points in snapshot
`c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25`.
The current final retrieval artifact reports:

| Variant | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Dense | 0.9011 | 0.8750 | 0.9296 |
| BM25 | 0.8178 | 0.7844 | 0.8376 |
| Hybrid RRF | 0.9233 | 0.8778 | 0.9518 |
| Hybrid + reranker | 0.9122 | 0.8333 | 0.9329 |

Therefore the demo default is Hybrid RRF with reranker OFF. The reranker
remains available for measured ablation; it is not disabled by assumption.

Other validated behaviors:

- no-answer and security-policy decisions skip the LLM;
- duplicate ingestion reuses the same document/version when content and the
  effective pipeline fingerprint match;
- arbitrary uploads default to `AUTO`, resolving to a structure-aware strategy
  only when reliable structure is present and otherwise to bounded `generic_v1`;
- the frozen mentor corpus remains an explicit `mentor_program_v1` evaluation
  membership, not a global admission rule;
- canonical evidence retains document, page, parent/child and rank metadata;
- installed models and ready models are reported separately.

## Quick start

Requirements:

- Docker Engine with Compose v2;
- Bash and `curl`;
- a local Ollama runtime reachable from Docker;
- `gemma3:4b` installed in that runtime for the measured demo path;
- Python 3.12 and the development tools for local tests.

Start Ollama first and install the recommended local model:

```bash
# Terminal 1 (skip if Ollama is already running with a Docker-reachable bind)
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2
ollama pull gemma3:4b
```

From this repository root:

```bash
./scripts/start_demo.sh
```

The launcher validates Compose configuration, starts the services, waits for
API liveness, then waits for readiness of Qdrant, Ollama, the selected
generation model and the ingestion worker. It exits non-zero and prints the
real dependency status if any required check fails. The Demo UI is opened only
after readiness succeeds at <http://127.0.0.1:8501>.

Stop the standalone stack without deleting its persisted data:

```bash
docker compose down --remove-orphans
```

The default Compose configuration uses `host.docker.internal:11434` for
Ollama and maps API/Qdrant/UI to `8010/6335/8501`. Override those host ports or
the Ollama URL when needed:

```bash
API_HOST_PORT=8011 QDRANT_HOST_PORT=6336 UI_HOST_PORT=8502 \
DIS_OLLAMA_URL=http://host.docker.internal:11434 \
./scripts/start_demo.sh
```

On Linux, the most portable option is to run Ollama on a separate local port
if an existing service already occupies `11434`:

```bash
# Terminal 1
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# Terminal 2
OLLAMA_HOST=http://127.0.0.1:11435 ollama pull gemma3:4b
DIS_OLLAMA_URL=http://host.docker.internal:11435 \
API_HOST_PORT=8011 QDRANT_HOST_PORT=6336 UI_HOST_PORT=8502 \
./scripts/start_demo.sh
```

Docker Desktop normally provides `host.docker.internal` on macOS and Windows.
The Compose file adds the same host-gateway mapping on Linux. Keep the Ollama
listener restricted to the local machine/network; this project does not
require public model exposure.

Readiness is intentionally not equivalent to process liveness. If the model,
Ollama or Qdrant check is unavailable, the API reports not-ready and the UI
does not pretend that generation is available. The API Compose healthcheck is
also readiness-based, so dependent UI startup is blocked until the selected
model is actually installed and reachable.

## Demo flow

1. Check live and ready health.
2. Upload a parseable PDF in the Ingestion tab. Product uploads use `AUTO` and
   can fall back to `generic_v1`; no mentor headings are required.
3. Select only documents with an active searchable version.
4. Run a direct fact, a paraphrase and an exact/numeric query.
5. Inspect Dense, BM25, RRF, evidence, answerability and the canonical source
   in the Query Trace.
6. Run the unrelated-question case and verify `NO_ANSWER` with the LLM skipped.
7. Run the prompt-injection regression and verify `SECURITY_POLICY` before
   generation.
8. Compare retrieval/reranker variants in Evaluation.

Unavailable historical ingestion records remain visible for diagnosis but are
not selectable and never enter retrieval. Re-upload or retry them through the
normal Ingestion tab so the current `AUTO` path is used.

## Evaluation and reproducibility

The immutable membership manifest is:

```text
data/evaluations/week2_final_corpus_snapshot_v1.json
```

It records the exact 26 point IDs, document/version membership, collection,
pipeline fingerprint and snapshot ID. The final evaluation dataset has 44
cases split into development (19), validation (11) and test (14). The latest
live test-split smoke recorded Recall@5 `0.9583`, MRR@10 `1.0000` and nDCG@10
`0.9834`; its evaluation run ID and raw evidence are preserved in
`projects/document_intelligence_service/eval/results/`.

The repository does not contain a live Qdrant dump or the private mentor PDF.
Large candidate reports that would repeat verbatim document text are also
excluded; the committed raw CSV/JSONL outputs retain IDs, ranks, decisions and
metrics without republishing source-document content.
To reconstruct the frozen collection, obtain an approved copy of the source
input, ingest it with the explicit `mentor_program_v1` evaluation profile, and
verify the resulting point manifest before running the benchmark. Normal
product ingestion must continue to use `AUTO`.

Typical benchmark invocation after the approved corpus has been reconstructed:

```bash
DIS_QDRANT_URL=http://127.0.0.1:6335 \
DIS_QDRANT_COLLECTION=document_chunks_week2_final_v1 \
DIS_SECTION_MARKER_PROFILE=mentor_program_v1 \
python -m projects.document_intelligence_service.eval.run_benchmark \
  --mode hybrid --top-k 5 --point-count 26 \
  --output projects/document_intelligence_service/eval/results/hybrid_baseline.json \
  --raw-output-dir projects/document_intelligence_service/eval/results
```

## Local development and tests

```bash
python3.12 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.13.0+cpu'
.venv/bin/pip install -e 'projects/document_intelligence_service[dev]'
.venv/bin/pytest -q projects/document_intelligence_service/tests
.venv/bin/ruff check projects/document_intelligence_service/app projects/document_intelligence_service/eval projects/document_intelligence_service/tests
.venv/bin/mypy projects/document_intelligence_service/app projects/document_intelligence_service/eval projects/document_intelligence_service/tests
docker compose config --quiet
```

The standalone Compose smoke validates health, the demo UI, optional PDF
ingestion, duplicate-ingestion idempotency and Qdrant restart persistence. It
does not publish a PDF in this repository; provide a local input explicitly:

```bash
SMOKE_PDF=/path/to/a/parseable.pdf ./scripts/compose_smoke.sh
```

Without `SMOKE_PDF`, the smoke still checks service startup, health, UI and
Qdrant persistence.

## Repository structure

```text
app source:  projects/document_intelligence_service/app/
tests:       projects/document_intelligence_service/tests/
evaluation:  projects/document_intelligence_service/eval/
datasets:    data/evaluations/
UI:          demo_ui/
docs/        architecture, compliance, model compatibility and ADR material
scripts/     standalone Compose smoke
```

The nested service path is intentional: it preserves the tested Python import
package and Docker build layout without adding parent-repository dependencies.

## Local models and limitations

`INSTALLED` does not mean `READY`. The service exposes model readiness from the
runtime probe and keeps Gemma as the measured default. On this development
machine, `qwen3:4b` was installed but a controlled `think=false,
stream=false` probe still ended with bounded, incomplete output and no reliable
final answer; it remains `INSTALLED · LAST PROBE FAILED` and is not selected by
default. Thinking text is never treated as a user-facing answer.

Known MVP limits include local CPU generation latency, a small generic-document
answerability calibration set, process-local model probe state, local ACL-ready
filters rather than a full identity provider, process-local metrics rather than
a metrics platform, and MVP-scoped retention/purge. These limitations do not
change the frozen benchmark labels or corpus membership.

## Architecture and decision records

- [Architecture](docs/architecture.md)
- [API examples](docs/api_examples.md)
- [Week-2 compliance](docs/week2_compliance.md)
- [Live evaluation smoke evidence](docs/week2_live_evaluation_smoke.md)
- [Model compatibility](docs/model_compatibility.md)
- [Service ADRs](projects/document_intelligence_service/docs/adr/)
