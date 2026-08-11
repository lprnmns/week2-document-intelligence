# Mentor technical questions — Week 2

This is a project-specific defense sheet. Every answer names the chosen design,
the alternative considered, evidence from this repository, and a known limit.
The frozen benchmark remains immutable; historical measurements keep their
original source SHA.

## 1. Why is the service split into API, application, domain and infrastructure?

**Chosen design.** FastAPI adapters live under `projects/document_intelligence_service/app/api`; use cases are in `app/application`; policy and data contracts are in `app/domain`; Qdrant, pypdf, sentence-transformers and Ollama are infrastructure adapters behind ports.

**Alternative.** Put the workflow in the route handler, which is shorter initially but makes retrieval, ingestion and failure tests depend on HTTP and external services.

**Evidence.** `app/main.py` composes the services, while `QueryService` and `RetrievalService` are exercised with fake ports in `tests/unit/`.

**Limitation.** The service is a local MVP; it does not provide a full deployment platform, identity provider or centralized telemetry backend.

## 2. How does an arbitrary PDF enter the system?

**Chosen design.** The upload is validated and inspected before acceptance. The default `AUTO` profile resolves reliable known structure when available and otherwise selects bounded `generic_v1`; the resolved profile enters the pipeline fingerprint before idempotent acceptance. The worker then runs parse → normalize → chunk → embed → stage → verify → activate.

**Alternative.** Require one mentor-specific heading contract globally. That was rejected because a valid unrelated PDF must not fail merely because those headings are absent.

**Evidence.** `IngestionPreparationService`, `KnownSectionMarkerProfileResolver` and `IngestionWorker`; the recorded arbitrary-PDF smoke is 8 pages, 5 bounded parents and 82 children with `AUTO → generic_v1`.

**Limitation.** The parser is selectable-text pypdf processing. OCR and advanced layout/table reconstruction are intentionally outside this scope.

## 3. Why are version activation and idempotency separate concerns?

**Chosen design.** Content hash plus effective pipeline fingerprint identifies an ingestion version. Qdrant first publishes a complete verified version; the registry atomically changes the authoritative active status; retrieval filters by the registry's active version IDs. Old physical payload flags are cleaned after the switch.

**Alternative.** Toggle Qdrant `active=true/false` in two sequential calls and treat that as atomic. A cleanup failure can expose both versions, so it is not sufficient.

**Evidence.** `IngestionWorker`, `SqliteIngestionRegistry.activate_document_version`, `InMemoryIngestionRegistry.activate_document_version` and `QdrantRetriever.snapshot_active_version_ids`.

**Limitation.** Physical old-point cleanup is best effort; the authoritative filter prevents stale points from being searchable while cleanup is pending.

## 4. What is the difference between Dense, BM25 and Hybrid RRF?

**Chosen design.** Dense uses multilingual sentence embeddings for semantic similarity. BM25 uses the named Qdrant sparse vector for lexical terms. Hybrid runs both branches and combines their ranks with Reciprocal Rank Fusion; raw score magnitudes are never added together.

**Alternative.** Use only cosine scores or only lexical search. The former mixes incompatible score spaces; the latter misses paraphrase and semantic matches.

**Evidence.** `RetrievalService._fuse` and ADR-002. On the frozen 44-case set: Dense Recall@5 `0.9011`, BM25 `0.8178`, Hybrid `0.9233`.

**Limitation.** The measurements are corpus/model/configuration-specific and are not a universal claim about every language or document type.

## 5. Why is the reranker OFF by default?

**Chosen design.** Reranking remains an explicit ablation after a bounded RRF window. The measured mentor configuration keeps it OFF by default.

**Alternative.** Always rerank because a cross-encoder sounds more accurate. That would ignore the observed quality and latency trade-off.

**Evidence.** Hybrid + reranker changed Recall@5 `0.9233 → 0.9122`, MRR@10 `0.8778 → 0.8333`, nDCG@10 `0.9518 → 0.9329`; warm reranker latency was measured separately and receives at most `20 → 5` candidates.

**Limitation.** A different corpus, model or hardware can change the result; the ON path stays available for remeasurement.

## 6. How does answerability decide whether to call the LLM?

**Chosen design.** The gate combines evidence existence, calibrated score, margin, lexical/qualifier coverage, filters and profile-aware policy. A failed gate returns a stable no-answer reason and the LLM is skipped.

**Alternative.** Always call the LLM and ask it to decide whether the evidence is enough. That increases cost and makes no-answer behavior less deterministic.

**Evidence.** `app/domain/answerability.py`, `QueryService` and the frozen calibration artifact. Mentor calibration uses validation only and keeps dense threshold `0.338`; generic calibration is separately scoped and uses dense `0.247` plus coverage `0.367`.

**Limitation.** The generic validation set is small, so statistical confidence is limited; coverage is a deterministic MVP signal, not a learned universal judge.

## 7. How do you protect against prompt injection and unsupported answers?

**Chosen design.** Direct prompt-safety checks run before retrieval; evidence safety removes indirect injection-like chunks before generation; prompt construction uses bounded canonical evidence; canonical source cards are generated by the application, not parsed from model text.

**Alternative.** Trust the model to ignore instructions embedded in retrieved text and extract citations from its answer.

**Evidence.** `PromptSafetyPolicy`, `EvidenceSafetyPolicy`, `QueryService`, `evidence_validation.py`, and the prompt-injection/security tests. Security-policy cases have retrieval/LLM skipped.

**Limitation.** Output validation is strongest for explicit numeric/qualifier grounding. General entity/name grounding and poisoning quarantine are deliberately partial MVP scope.

## 8. How can a reviewer diagnose a wrong answer?

**Chosen design.** The UI presents a Run Result, a clickable Run Diagnosis / Decision Path, a Candidate Journey, and an Evidence Inspector. The same candidate can be followed through Dense, BM25, RRF, reranker and evidence selection.

**Alternative.** Show only the final answer or a raw JSON wall. Neither proves where the candidate changed.

**Evidence.** `demo_ui/index.html` renders application trace events, rank fields and canonical excerpts using `textContent`. For arbitrary queries it shows diagnostic rules rather than claiming ground-truth attribution; labeled evaluation cases can use expected evidence IDs.

**Limitation.** Without trusted expected evidence, the UI cannot honestly say that a retrieved answer is wrong or assign blame to one stage.

## 9. Why is the frozen benchmark identified by a snapshot and not only a fingerprint?

**Chosen design.** `data/evaluations/week2_final_corpus_snapshot_v1.json` fixes the exact 26 Qdrant point IDs and document/version membership. The pipeline fingerprint is retained as reproducibility metadata.

**Alternative.** Filter evaluation only by `pipeline_fingerprint`. A future product document can legitimately use the same configuration and would then leak into the benchmark.

**Evidence.** `EvaluationCorpusSnapshot`, `_load_evaluation_corpus_snapshot`, the snapshot regression tests and the evaluation configuration endpoint.

**Limitation.** Reconstructing the frozen points requires an approved copy of the private source PDF; the repository intentionally ships the manifest, not a Qdrant dump or private PDF.

## 10. What does reproducibility mean here?

**Chosen design.** Manifests record dataset SHA, corpus snapshot ID, point count, profile/fingerprint, models, RRF limits, reranker state, host facts and historical evaluation SHA. The final release tag identifies the source tree without pretending that old measurements ran on a later commit.

**Alternative.** Rewrite historical JSON results after each release to show the latest SHA. That destroys provenance.

**Evidence.** `eval/results/week2_stabilization_v1/run_manifest.json`, `docs/release_manifest.md` and `data/evaluations/week2_final_corpus_snapshot_v1.json`.

**Limitation.** CPU LLM latency and runtime model availability are machine-dependent; exact benchmark reproduction also depends on the approved corpus and local model/runtime versions.

## 11. How is model readiness different from installation?

**Chosen design.** Ollama discovery reports installed models, while a bounded generation probe records readiness separately. The UI distinguishes READY, INSTALLED · UNVERIFIED, INSTALLED · LAST PROBE FAILED, NOT INSTALLED and RUNTIME UNAVAILABLE.

**Alternative.** Treat an installed tag as proof that generation will work.

**Evidence.** `ModelService`, `OllamaModelRuntimeAdapter`, `/v1/system/profile` and the Qwen probe history. Gemma remains the measured default when ready; Qwen is not promoted without a successful final-response probe.

**Limitation.** Probe state is process-local and runtime behavior can change after model updates.

## 12. Which security and operational claims are intentionally not made?

**Chosen design.** The service enforces upload/page bounds, ACL prefilter plus application re-check, direct/indirect injection defenses, safe UI text rendering, structured request IDs and bounded candidate/evidence sizes.

**Alternative.** Call the local MVP enterprise-ready or claim complete poisoning, DoS and centralized-log controls.

**Evidence.** `docs/security_attack_matrix_v1.md`, `app/domain/*safety*`, Compose resource/log limits and the security test artifacts.

**Limitation.** Provenance allowlists/quarantine, rate/concurrency quotas, parser timeouts, authenticated principal integration and centralized retention/access policy remain partial and are documented as such.
