# Document Intelligence Service

Hafta 1'deki RAG çekirdeğini, PDF kabul eden ve kanıt yolunu görünür kılan
local-first bir servis haline getirir.

## Çalışan topoloji

```text
demo-ui :8501 → api :8000 → SQLite job registry
                    │
                    ├── POST /v1/queries → dense/BM25/RRF → gate → Ollama
                    ├── POST /v1/documents → worker → Qdrant :6333
                    └── /v1/demo/query-runs → bounded live trace transport

worker ve api aynı image'i kullanır; Ollama API image'ının dışında ayrı bir
model runtime'ıdır. Önerilen delivery Compose yolu `ollama` servisini ve
`http://ollama:11434` adresini kullanır; host runtime isteğe bağlıdır. Qdrant named volume
ile kalıcıdır.
```

Host portları çakışmayı önlemek için varsayılan olarak API `8010`, UI `8501`,
Qdrant `6335`'tir. Container içindeki portlar API `8000`, Qdrant `6333` olarak
kalır.

## Başlatma

Repo kökünden, önerilen clean delivery yolu:

```bash
docker compose -f compose.yaml -f compose.ollama.yaml \
  --profile bundled-ollama up --build -d
docker compose -f compose.yaml -f compose.ollama.yaml \
  --profile bundled-ollama ps
curl -i http://127.0.0.1:8010/v1/health/live
curl -i http://127.0.0.1:8010/v1/health/ready
# Open http://127.0.0.1:8501 after readiness passes.
```

Readiness `503` ise bu bir “cevap vermeyi dene” durumu değildir. Response içindeki
Qdrant/Ollama check'lerini oku. Önerilen bundled Compose yolu Ollama'yı
`http://ollama:11434` üzerinden sağlar. Host üzerinde çalışan Ollama kullanacaksan
(optional host mode):

```bash
DIS_OLLAMA_URL=http://host.docker.internal:11434 \
  docker compose up --build -d
```

Smoke script API, worker ve UI'yi açar; örnek PDF'i upload eder, job'ın
`validate → inspect → parse → normalize → chunk → embed_dense → embed_sparse →
stage_qdrant → verify → activate → complete` timeline'ını bekler. Aynı PDF'i
ikinci kez de göndererek `idempotent_hit=true`, aynı document/version/job
kimlikleri ve değişmeyen Qdrant point count'ını doğrular; ardından Qdrant
restart sonrası kalıcılığı kontrol eder.

32 GB RAM için:

- Ollama ayrı model container'ı olarak çoğaltılmaz.
- Compose API/worker CPU ve RAM limitleriyle çalışır.
- Reranker varsayılan kapalıdır; açılırsa en fazla 20 aday üzerinde çalışır.
- LLM yalnız answerability geçerse çağrılır ve `max_output_tokens` bounded'dır.
- Host portu doluysa `API_HOST_PORT` veya `QDRANT_HOST_PORT` değiştirilebilir.

## API akışı

1. `POST /v1/documents` PDF'i boyut/MIME/magic-byte/sayfa kontrolünden geçirir ve
   `202 + job_id` döndürür.
2. SQLite registry content hash + etkin pipeline fingerprint ile idempotent
   identity tutar. Upload profili varsayılan olarak `auto`'dur; bilinen marker
   sözleşmesi yoksa `generic_v1` fallback seçilir. Aynı upload tekrarında aynı
   document/version receipt'i döner.
3. Worker parent/child chunk üretir, named dense/sparse Qdrant point'lerini
   inactive stage'e yazar, count/metadata doğrular ve sonra active eder.
4. `POST /v1/queries` tenant/ACL/document filtrelerini normalize eder; dense,
   BM25 veya hybrid RRF ile bounded candidate listesi üretir.
5. Reranker açıksa RRF sonrası en fazla 20 aday üzerinde çalışır ve final top-5
   evidence döner.
6. No-answer veya security policy kararında Ollama çağrılmaz. `sources` her
   zaman retrieval'dan gelen canonical evidence nesnelerinden üretilir.

Query için `POST /v1/queries` kullanılır; eski `POST /v1/query` uyumluluk alias'ı
olarak korunur. `POST /v1/search` yalnız retrieval yapar ve `llm_ms=0` döner.

Demo UI, canonical query response'unu değiştirmeden `/v1/demo/query-runs`
üzerinden aynı `QueryService` use-case'ini polling ile izler. Request, scope,
query representation, dense, BM25, RRF, reranker, evidence, answerability,
prompt, LLM ve response event'leri gerçek application kodundan gelir;
frontend pipeline simüle etmez. `DIS_DEMO_TRACE_ENABLED=false` ile kapatılabilir.

System / Models paneli `/v1/system/profile` üzerinden sanitized CPU/RAM/GPU,
Ollama runtime ve installed model listesini gösterir. Generation, embedding,
BM25 ve reranker rolleri ayrı tutulur. Compatibility sınıfları heuristiktir;
“will run” garantisi değildir. Model pull varsayılan olarak kapalıdır ve
allow-list + Ollama HTTP adapter'ı ile sınırlandırılmıştır.

## Sözleşme ve gözlemleme

- Health: `/v1/health/live`, `/v1/health/startup`, `/v1/health/ready`
- Catalog: `/v1/documents`, `/v1/documents/{id}`
- Job: `/v1/jobs/{id}`
- Query/search: `/v1/queries`, `/v1/query`, `/v1/search`
- Metrics: `/v1/metrics`
- Evaluation: `/v1/evaluations/runs`
- Demo trace (OpenAPI dışı): `/v1/demo/query-runs`
- System/models (OpenAPI dışı): `/v1/system/profile`, `/v1/system/models/pulls`

Job response'unda `attempt_count`, `max_attempts`, `current_stage`, her stage'in
`duration_ms`, input/output özeti, decision ve hata alanları bulunur. Query
trace; request ID, question hash, retrieval adayları, answerability kararı ve
embed/search/rerank/LLM sürelerini JSON log olarak taşır. Audit event'leri
`document.audit` adıyla kabul/activate/fail/delete işlemlerini raw PDF veya raw
soru yazmadan kaydeder.

`/v1/metrics` process-local JSON registry'dir; Prometheus sunucusu değildir.
Worker metrikleri worker logları ve job timeline üzerinden izlenir. Bu local MVP
sınırı bilinçli olarak belgelenmiştir.

## Geliştirme kontrolleri

```bash
.venv/bin/pytest -q projects/document_intelligence_service/tests
.venv/bin/ruff check projects/document_intelligence_service/app projects/document_intelligence_service/eval projects/document_intelligence_service/tests
.venv/bin/mypy projects/document_intelligence_service/app projects/document_intelligence_service/eval projects/document_intelligence_service/tests
docker compose config --quiet
```

## Benchmarkı yeniden üretme

Dedicated Week 2 Qdrant `6335` portunda açık ve benchmark PDF'i section-aware
profil ile indekslenmiş olmalıdır. Worker volume'ündeki `/data/bm25_state.json`
dosyası hostta erişilebilir bir kopyaya alınır; sparse query vocabulary'si ile
ingestion state'i aynı kalmalıdır.

```bash
export DIS_QDRANT_URL=http://127.0.0.1:6335
export DIS_QDRANT_COLLECTION=document_chunks_week2_final_v1
export DIS_BM25_STATE_PATH=/tmp/week2-benchmark/bm25_state.json
export DIS_SECTION_MARKER_PROFILE=mentor_program_v1
# For Alperen's 28-page Week 2 program PDF, use:
# export DIS_SECTION_MARKER_PROFILE=mentor_program_week2_v1

.venv/bin/python -m projects.document_intelligence_service.eval.run_benchmark \
  --mode hybrid --top-k 5 --point-count 26 \
  --output projects/document_intelligence_service/eval/results/hybrid_baseline.json \
  --raw-output-dir projects/document_intelligence_service/eval/results

.venv/bin/python -m projects.document_intelligence_service.eval.run_evidence_coverage \
  --benchmark projects/document_intelligence_service/eval/results/hybrid_baseline.json \
  --output projects/document_intelligence_service/eval/results/hybrid_evidence_coverage.json
```

Dense/BM25 ve reranker varyantları aynı komutun `--mode`/`--reranker`
seçenekleriyle çalıştırılır. Final A/B/C/D özeti
`eval/results/week2_stabilization_v1/` altında tutulur; bu koşu Ollama
çağırmaz. Container image `.git` içermediği için `DIS_SOURCE_REVISION` canlı
evaluation manifest'ine kaynak revision'ını taşır; değişken verilmezse değer
bilinçli olarak `unknown` kalır.

Ephemeral Qdrant entegrasyon testi yalnız URL verilirse çalışır:

```bash
QDRANT_INTEGRATION_URL=http://127.0.0.1:6333 \
  pytest -q projects/document_intelligence_service/tests/integration
```

## Sınırlar

Canlı Compose trace/evaluation kanıtı repo kökündeki
`docs/week2_live_evaluation_smoke.md` dosyasında tutulur. Compose varsayılanı
ve benchmark aynı `document_chunks_week2_final_v1` snapshot'ını kullanır;
eski generic corpus yalnız korunmuş audit verisidir.

Bu servis production authentication/authorization, object storage, rate limit,
çoklu worker koordinasyonu, Kubernetes, tıbbi karar ve tool calling iddiasında
bulunmaz. `tenant_id` ve `acl_tags` local ACL-ready filtre sınırıdır; gerçek
request principal/policy store bir sonraki güvenlik kapsamıdır. 44-vakalık golden
set ve mevcut benchmark sonuçları aynı corpus/config için geçerlidir; model,
chunk veya collection değişince yeniden çalıştırılmalıdır.

Detaylı karar kayıtları `docs/adr/`, API örnekleri `docs/api_examples.md`,
benchmark `docs/benchmark_report.md`, security matrisi
`docs/security_attack_matrix_v1.md`, 28 sayfalık kabul matrisi
`docs/acceptance/week2_pdf_acceptance_matrix.md` ve sayfa sayfa görsel inceleme
`docs/acceptance/week2_pdf_visual_review.md` altındadır.

Güncel compliance matrisi repo kökünde `docs/week2_compliance.md`, model
uyumluluk varsayımları ise `docs/model_compatibility.md` dosyasındadır.
