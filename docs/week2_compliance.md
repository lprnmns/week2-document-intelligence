# Week-2 Compliance Matrix

Kaynak: approved local Week-2 mentor specification PDF (28 pages; private source bytes excluded)

PDF tam olarak 28 sayfadır. Metin denetimi ve her sayfanın diyagram/tablo
görsel kontrolü şu kanıtlarla tutulur:

- `projects/document_intelligence_service/docs/acceptance/week2_pdf_visual_review.md`
- `projects/document_intelligence_service/docs/acceptance/week2_pdf_acceptance_matrix.md`
- `projects/document_intelligence_service/eval/verify_week2_pdf.py`

Durumlar: `PASS` kanıtlandı; `PARTIAL` ana davranış var fakat açık sınır veya
eksik acceptance detayı var; `GAP` zorunlu davranış kanıtlanmadı.

## A. REQUIRED WEEK-2 FUNCTIONALITY

| Requirement | PDF section/page | Current status | Code/test evidence | Action needed |
|---|---|---|---|---|
| PDF kabulü, kontrollü indexing, retrieval, kaynaklı cevap ve explicit no-answer | §§02–03, pp. 2–3 | PASS | `app/api/v1/documents.py`, `search.py`, `queries.py`; query/ingestion/security tests; final Compose smoke | Davranışı koru |
| `api → application → domain`, infrastructure adapter sınırı | §§04, 06, p. 4, 6 | PASS | `docs/architecture.md`; `app/api`, `application`, `domain`, `infrastructure` | Qdrant/Ollama kararlarını endpoint/UI içine taşımama |
| Query sequence: validation, scope, dense/sparse, RRF, bounded rerank, evidence, gate, LLM/no-answer | §05, p. 5 | PASS | `QueryService`, `RetrievalService`; `tests/unit/test_query_service.py`, `test_retrieval_service.py` | Yeni stage trace kanıtlarını koru |
| FastAPI lifespan/DI, startup/live/ready health, sync adapter offload | §07, p. 7 | PASS | `app/main.py`, health routes, `asyncio.to_thread`; health/worker tests | Ollama yokken readiness ayrımını smoke’ta göster |
| REST resources: documents, jobs, queries, search, evaluations | §08, p. 8 | PASS | `app/api/v1/*`, `contracts.py`; contract tests | Canonical `/v1/queries` sözleşmesini koru |
| Stable response/error models, safe error envelope, canonical source metadata | §09, p. 9 | PASS | `contracts.py`, `api/errors.py`; resource/security contract tests | Raw prompt/path/stack sızıntısı ekleme |
| Eight-stage ingestion: accept, identity, parse, normalize, chunk, embed, stage, activate | §10, p. 10 | PASS | `ingestion_worker.py`: `validate`, `inspect`, `parse`, `normalize`, `chunk`, `embed_dense`, `embed_sparse`, `stage_qdrant`, `verify`, `activate`; worker tests | `embed_dense`/`embed_sparse` alt aşamalarını dokümante edilmiş biçimde koru |
| Idempotency: `content_hash + pipeline_fingerprint`; duplicate point üretmeme | §10, p. 10; §24, p. 24 | PASS | SQLite/in-memory registry; duplicate contract test; final Compose smoke duplicate assertion | Final smoke evidence `docs/week2_live_evaluation_smoke.md` içinde |
| Versioning, staged inactive points, verification and atomic activation | §10, p. 10 | PASS (MVP scope) | `IngestionWorker`, `QdrantChunkStore`, `SqliteIngestionRegistry.activate_document_version`, `QdrantRetriever` authoritative version filter; activation regression tests; final 26-point ingest | Physical old-point cleanup is best effort; registry selection prevents mixed-version retrieval. Retention/purge remains bounded to the MVP |
| Named dense+sparse Qdrant schema, deterministic IDs, dimension validation | §11, p. 11 | PASS | `qdrant/schema.py`, `chunk_store.py`; Qdrant schema tests; compatible legacy collections receive missing additive indexes | Existing vector dimensions/sparse modifier remain strict; payload values still need explicit migration |
| Payload metadata: document/version/active/tenant/ACL/page/source/pipeline/model identity | §§11–12, pp. 11–12 | PASS (frozen membership) | `eval/verify_active_metadata.py`; `eval/results/final_corpus_metadata.json`; frozen manifest membership inspected, missing required metadata `0` | The live product/demo collection may contain additional active points; the preserved legacy collection is not part of the frozen claim |
| Pre-filter semantics: tenant → ACL → document scope → active version; source re-check | §12, p. 12 | PASS | `QdrantRetriever._active_filter`, `RetrievalService._filter_access`, `scope.py`; scope/retrieval/security tests | Document scope’u router gibi sunma; UI yalnız scope olarak göstermeli |
| Dense semantic retrieval | §13, p. 13 | PASS | `SentenceTransformerEmbedder`, retrieval service and candidate trace | Actual model/dimension’ı trace’te göstermeye devam et |
| BM25/sparse lexical retrieval | §§13–14, pp. 13–14 | PASS | `BM25SparseEncoder`, named Qdrant `bm25`; sparse tests and benchmark artifacts | Türkçe morfoloji sınırını değerlendirme raporunda tut |
| Hybrid RRF rank fusion; raw score spaces not mixed | §§13–14, pp. 13–14 | PASS | `_fuse`, RRF tests, ADR-002, live rank table | RRF `k`/candidate limits manifest’e yazılmalı |
| Bounded reranker and reranker ablation | §15, p. 15 | PASS | `CrossEncoderReranker`, bounded input tests; canonical `/v1/queries` ON/OFF regression; ablation artifacts | Model/CPU latency değişince ablation’ı yeniden çalıştır |
| Retrieval metrics: Recall@1/3/5, MRR, nDCG, retrieval p50/p95, failure rate, slices | §§14–16, pp. 14–16 | PASS | `eval/metrics.py`, `runner.py`, raw CSV/JSONL, `eval/results/week2_stabilization_v1/` | Keep all four same-snapshot variants and raw outputs together |
| Reproducible run manifest: SHA, dataset, corpus snapshot, models, config, host | §16, p. 16 | PASS (delivery path) | `eval/reporting.py`, final-delivery manifest, `/v1/evaluations/config`, `DIS_SOURCE_REVISION` and packaged `DELIVERY_SHA.txt` fallback | Historical artifacts retain their original SHA; final-delivery artifacts are tied to the final source SHA |
| 40+ golden cases and required categories | §17, p. 17 | PASS | `data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl`: 44 case; dataset tests | Split/category sayımlarını CI’de koru |
| Development/validation/test separation; no test leakage in threshold selection | §§17–18, pp. 17–18 | PASS | `eval/calibration.py`, calibration tests, split validation | Threshold provenance’u manifest’te tut |
| Answerability signals and stable reason codes | §18, p. 18 | PARTIAL (calibrated dense threshold) | `AnswerabilityPolicy`; validation-only calibration `0.337857395 → 0.338`; frozen final-corpus test: FP `0`, FN `2`; reason-code/unit tests | Margin/coverage remain diagnostic provisional gates (`0.0`); test result is reported, never used for threshold selection |
| LLM is actually skipped when gate/security fails | §18, p. 18 | PASS | `QueryService` emits `llm=skipped`; generator call-count tests; `latency.llm_ms=0` contract tests | Empty/nonexistent document scope ile demo acceptance’ını tekrar et |
| Direct/indirect prompt injection, cross-document leakage, safe rendering | §19, p. 19 | PASS (local MVP) | `prompt_safety.py`, `evidence_safety.py`; injection/leakage/security matrix tests; UI `textContent` | Unknown attacks, authenticated principal and full URL/image sanitizer sonraki güvenlik kapsamı |
| Request correlation, structured logs, stage latency, metrics and audit | §20, p. 20 | PASS (local MVP) | `request_id.py`, `query_trace.py`, `metrics.py`, `audit.py`; live trace/metrics/audit tests | Metrics process-local; merkezi collector/exporter zorunlu değil fakat sonraki operasyon sınırı |
| Compose topology, worker, health checks, persistent Qdrant volume | §21, p. 21 | PASS | `compose.yaml`, Dockerfiles, final clean-start/restart smoke | Final smoke evidence `docs/week2_live_evaluation_smoke.md` içinde |
| CI: lint, type, unit/contract/security/evaluation/image checks | §22, p. 22 | PARTIAL | `.github/workflows/document-intelligence-service.yml`, `.github/pull_request_template.md`, `pyproject.toml` | Ruff, Mypy, non-integration tests, frozen-manifest smoke, Docker build, advisory `pip-audit` and secret scan run. Qdrant HTTP integration, SBOM and image vulnerability scan are not implemented; branch protection is not claimed/configured here |
| Five-day gates and evaluation/idempotency priority | §23, p. 23 | PASS | `docs/demo_runbook_20min.md`, `docs/release_manifest.md`, service tests and evaluation artifacts | Mentor interaction remains a manual review; no external score is claimed |
| Required deliverables: source, UI, Compose, dataset, raw outputs, report, ADR, README, API examples | §24, p. 24 | PASS | Repository source, `demo_ui`, `eval/results`, docs, ADRs, README, final live smoke report | Known limitations açıkça korunmalı |
| Review/demo flow: health → upload → duplicate → query → no-answer → injection → benchmark | §25, p. 25 | PASS (repeatable local evidence) | UI tabs, final-corpus compose smoke, live trace, security/eval artifacts and deterministic flip examples | Mentor interaction itself remains a manual demonstration; the underlying cases are runnable |
| Critical-fail rubric: clean setup, duplicate safety, provenance, ACL, no-answer, security | §26, p. 26 | PASS (automated local evidence) | Contract/unit/security/Qdrant tests and smoke assertions | Real external review/mentor score otomatikleştirilemez |
| Technical interview questions with decision/evidence/alternative/limitation | §27, p. 27 | PASS | `docs/mentor_technical_questions.md`, ADRs | Answers are project-specific; external mentor review is not fabricated |
| `.env.example`, at least five ADRs, official references, final SHA note | §28, p. 28 | PASS | `.env.example`; ADR-001..008; `docs/references.md`; `docs/release_manifest.md` | Release tag is authoritative; historical evaluation SHA remains unchanged |

### Final corpus consistency evidence

The final Compose default is now the same clean, section-aware corpus used by
the golden benchmark:

- collection: `document_chunks_week2_final_v1`
- active points: `26`
- pipeline fingerprint: `132e52a3e8358e66906a7dd9bcfd0c8b57aa228dd3102e9b3d8f39ccfb4c41a4`
- corpus snapshot: `c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25`
- metadata verifier: `26/26` inspected, `missing_required_metadata=0`

The old `document_chunks_v2_bm25` collection and its 135-point snapshot are
preserved separately for rollback/audit. They are not mixed into the final
benchmark claim. The 28-page Week-2 PDF was rejected by the Week-1-specific
section profile and was not silently indexed as a different document family.

## B. OPTIONAL / DEMO USABILITY EXTENSIONS

These extensions are not substitutes for the required Week-2 work.

| Requirement | PDF section/page | Current status | Code/test evidence | Action needed |
|---|---|---|---|---|
| Compact three-tab engineering console: ASK / DOCUMENTS / BENCHMARKS | Demo extension related to §§20, 25 | PASS | `demo_ui/index.html` | Keep raw JSON under Details |
| Live query trace while the real use-case runs | Demo extension related to §§05, 20 | PASS | `/v1/demo/query-runs`, `LiveQueryTraceStore`, `test_demo_system_api.py` | Transport is polling and in-memory; SSE/durable store not needed for local demo |
| Real stages: scope → representation → dense/BM25 → RRF → rerank → evidence → gate → LLM → response | Demo extension | PASS | `QueryService`/`RetrievalService` callback events; UI renders application events | Do not add frontend-simulated stages |
| Document distribution and rank movement without fake document routing | Demo extension related to §§12–15 | PASS | retrieval distribution/candidate traces; UI default compact insights | Visualize dominance as retrieval result, never “PDF chosen/rejected” |
| Short educational tooltips for Dense/BM25/RRF/Reranker/Answerability/Parent | Demo extension | PASS | `demo_ui/index.html` info titles and compact hints | Keep explanations one sentence |
| Compact header health strip and detailed drawer | Demo extension | PASS | health endpoints and UI refresh every second; System/Models drawer | Health details remain behind drawer |
| Sanitized host profile: CPU/RAM/GPU/VRAM/acceleration/container | Local-first extension | PASS | `HostProfileAdapter`, `ModelService`, `/v1/system/profile`; sanitization test | Never expose paths, users, env or secrets |
| Runtime reachability and installed model discovery | Local-first extension | PASS | `ModelRuntimePort`, `OllamaModelRuntimeAdapter`, model tests | Runtime-installed models are not automatically assigned unrelated roles |
| Heuristic compatibility estimation and uncertainty labels | Local-first extension | PASS | `ModelCompatibilityEstimator`, `docs/model_compatibility.md`; deterministic/unknown tests | Never claim guaranteed execution |
| Safe optional model pull with actual progress | Local development extension | PASS | HTTP-only Ollama `/api/pull`, allowlist validation before pull state, pull tests | Disabled by default; not exercised when local management is off |
| Separate generation/embedding/sparse/reranker roles | Local-first extension | PARTIAL | Domain `ModelRole`, API profile and config line; generation selector | Dense/reranker runtime role verification remains adapter-specific/unverified |
| Missing selected model versus unavailable runtime | Local-first extension | PASS | `ModelService`, readiness probe, demo validation contract | Keep both states visible in UI |
| Embedding model/dimension change protection and deliberate re-index path | Required safety extension of §11 | PARTIAL | Qdrant dimension validation; pipeline fingerprint/model metadata; system profile warning | No user-facing re-index workflow yet; never silently switch/query incompatible vectors |
| Actual model/config identity in trace, logs and evaluation manifest | Reproducibility extension of §16 | PARTIAL | retrieval model fields, evaluation configuration, trace model fields, frozen benchmark manifest | Generation/embedding/sparse/reranker role verification remains adapter-specific; unknown must remain explicit |

## Final audit conclusion

The required retrieval, ingestion, security, evaluation and API core is present
with automated evidence. The live Compose corpus and stored benchmark now refer
to the same frozen snapshot; the old 135-point corpus is preserved but excluded
from the final claim. Remaining bounded gaps are: retention/purge policy is
still MVP-scoped, the preserved legacy corpus is not migrated, the small
generic-profile answerability calibration has limited statistical confidence,
authenticated production ACL and central metrics are outside this local MVP,
and there is intentionally no one-click embedding re-index UI. Generic
qualifier coverage now rejects the recorded wrong-year, wrong-program,
wrong-discount and year-attribute near-miss regressions without changing the
generic dense threshold or frozen mentor calibration. The local model panel is
an optional operational extension and does not replace those required
controls.
