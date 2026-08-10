# API v1 Examples

- Base URL (host Python): `http://127.0.0.1:8000`
- Base URL (default Compose mapping): `http://127.0.0.1:8010`

## Compose demo

From the repository root:

```bash
docker compose up --build -d
docker compose ps
open http://127.0.0.1:8501
```

The API container reaches Qdrant through the Compose network at
`http://qdrant:6333`. The default local setup reaches the existing Ollama
container at `http://ai-journey-ollama:11434`. If Ollama runs as a host
process instead, set `DIS_OLLAMA_URL=http://host.docker.internal:11434`; the
Compose file provides the `host-gateway` mapping. The host-facing API port is
`8010` and Qdrant port is `6335` by default to avoid colliding with the earlier
local Qdrant demo on `6333`.

If Ollama itself is not running, readiness correctly remains `503`; liveness
alone does not prove that model generation is available.

Compose API ve worker aynı image'i kullanır. API `202 + job_id` ile SQLite
registry'ye bırakır; ayrı worker queued/retryable/stale job'ları alır. Redis
bu local MVP'de opsiyonel bir sonraki scale-out kararıdır.

## Health

```bash
curl -i http://127.0.0.1:8000/v1/health/live
curl -i http://127.0.0.1:8000/v1/health/startup
curl -i http://127.0.0.1:8000/v1/health/ready
```

`live` yalnızca API sürecini kontrol eder. `ready` Qdrant ve Ollama gibi zorunlu bağımlılıkları da kontrol eder.

Local durable worker composition'ını açmak için:

```bash
export DIS_INGESTION_REGISTRY_BACKEND=sqlite
export DIS_INGESTION_DATABASE_PATH=data/ingestions.sqlite3
# Normal uploads: auto (known structure or generic_v1 fallback).
export DIS_SECTION_MARKER_PROFILE=auto
# Frozen benchmark / explicit mentor profiles only:
# export DIS_SECTION_MARKER_PROFILE=mentor_program_v1
# export DIS_SECTION_MARKER_PROFILE=mentor_program_week2_v1
# Baseline için false; ölçümlü reranker deneyi için true.
export DIS_RERANKER_ENABLED=false
export DIS_LLM_MODEL=gemma3:4b
export DIS_LLM_MAX_OUTPUT_TOKENS=256
```

## Document upload

```bash
curl -i \
  -H 'Idempotency-Key: upload-demo-001' \
  -H 'X-Tenant-ID: default' \
  -H 'X-ACL-Tags: public' \
  -F 'file=@sample.pdf;type=application/pdf' \
  http://127.0.0.1:8010/v1/documents
```

Beklenen çalışan akış:

```json
{
  "document_id": "doc_...",
  "version_id": "ver_...",
  "job_id": "job_...",
  "status": "indexing",
  "request_id": "req_..."
}
```

Status: `202 Accepted`. Compose'ta SQLite registry ve ayrı worker kullanılır;
job response'u `attempt_count`, `max_attempts`, `current_stage`, stage listesi,
duration ve hata/decision alanlarını taşır. Aynı content hash + pipeline
fingerprint tekrarında duplicate point üretilmez.

## Job status

```bash
curl -i http://127.0.0.1:8000/v1/jobs/job_...
```

## Query

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: mentor-query-001' \
  -H 'X-Tenant-ID: default' \
  -d '{
    "question": "Yerel model karşılaştırmasında hangi değerler ölçülmelidir?",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "include_debug": false
  }' \
  http://127.0.0.1:8010/v1/queries
```

İstek gövdesinde `tenant_id`, `acl_tags` ve `document_ids` de verilebilir.
Header kullanıldığında `X-Tenant-ID` ve `X-ACL-Tags` transport-level scope olarak
canonical kabul edilir. Body ile header aynı anda gönderilirse değerler aynı
olmalıdır; farklı tenant veya ACL değerleri `400 INVALID_REQUEST` döndürür.
Body alanları header kullanmayan geriye dönük istemciler için desteklenir.
Filtreler retrieval çağrısından önce normalize edilir; Qdrant pre-filter ve
application source re-check birlikte çalışır. `tenant_id` burada local ACL-ready
izolasyon filtresidir, authentication değildir.

## Evidence search

`/v1/search` yalnız retrieval kanıtlarını döndürür; LLM çağırmaz. Local SQLite composition açıkken:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: search-demo-001' \
  -H 'X-Tenant-ID: default' \
  -H 'X-ACL-Tags: public' \
  -d '{
    "question": "Yerel model karşılaştırmasında hangi değerler ölçülmelidir?",
    "retrieval_mode": "hybrid",
    "top_k": 5
  }' \
  http://127.0.0.1:8010/v1/search
```

Response'taki `sources` canonical Qdrant payload'ından, `retrieval` ise dense/sparse/RRF trace'inden gelir. `llm_ms` bu endpointte her zaman `0` olmalıdır.

`DIS_RERANKER_ENABLED=true` seçilirse RRF sonrası bounded cross-encoder devreye girer; en fazla 20 adayı skorlayıp en fazla 5 final kaynak döndürür. CPU cold-start ve inference latency'si baseline ile ayrı ölçülmelidir.

Query önce direct-injection safety policy'sinden, sonra answerability gate'ten geçer. `SECURITY_POLICY`, `NO_EVIDENCE` veya `LOW_RELEVANCE` kararında retrieval veya Ollama çağrısı atlanır ve `llm_ms=0` kalır. Kanıt yeterliyse `gemma3:4b` yalnız bounded evidence prompt'u ile çağrılır. Qdrant/embedding çalıştığı halde Ollama üretimi başarısızsa bu no-answer değildir; güvenli `503 DEPENDENCY_UNAVAILABLE` döner.

Answered response'ta `warnings` alanı, model çıktısının final evidence ile
karşılaştırılmasından doğan structured output/evidence uyarılarını taşır.
İlk sürüm `UNSUPPORTED_NUMBER` koduyla evidence'ta bulunmayan sayıları bildirir.
Bu warning cevabı otomatik olarak reddetmez; `sources` ise her zaman retrieval
payload'ından canonical olarak üretilir.

## No-answer response

```json
{
  "decision": "no_answer",
  "answer": null,
  "no_answer_reason": "LOW_RELEVANCE",
  "sources": [],
  "retrieval": {
    "mode": "hybrid",
    "dense_candidates": 30,
    "sparse_candidates": 30,
    "rrf_candidates": 20,
    "reranked_candidates": 5
  },
  "model": {"provider": null, "model": null},
  "warnings": [],
  "latency": {
    "embedding_ms": 12.4,
    "search_ms": 18.1,
    "rerank_ms": 38.2,
    "llm_ms": 0,
    "total_ms": 70.1
  },
  "request_id": "mentor-query-001"
}
```

No-answer kararını operasyonel olarak kontrol etmek için response'taki
`no_answer.reason_code`, `model.model: null`, `latency.llm_ms: 0` ve `/v1/metrics`
çıktısındaki `rag_no_answer_total` birlikte okunur. Query trace raw soru/evidence
yerine question hash ve stage span'leri taşır; lifecycle audit ise document,
version, action ve result alanlarını taşır.

## Metrics ve katalog

```bash
curl -sS http://127.0.0.1:8010/v1/metrics
curl -sS -H 'X-Tenant-ID: default' \
  'http://127.0.0.1:8010/v1/documents?limit=100'
```

Metrics endpoint local process snapshot'ıdır; merkezi Prometheus exporter'ı
değildir. Worker stage sürelerinin asıl kanıtı `/v1/jobs/{job_id}` timeline'ı ve
worker structured loglarıdır.

## Live demo trace (development transport)

Canonical `/v1/queries` sözleşmesi değişmeden, mentor demosu gerçek use-case
event'lerini polling ile izleyebilir:

```bash
curl -sS -X POST \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: default' \
  -H 'X-ACL-Tags: public' \
  -d '{
    "question": "Yerel model karşılaştırmasında hangi değerler ölçülmelidir?",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "reranker_enabled": false,
    "document_ids": []
  }' \
  http://127.0.0.1:8010/v1/demo/query-runs

curl -sS http://127.0.0.1:8010/v1/demo/query-runs/trace_...
```

Trace içindeki `reranker` event'i ON ise bounded input/output sayısını, OFF ise
`skipped / configuration` durumunu gösterir. Dense/BM25 document dağılımı,
RRF rank'ı, reranker rank movement, evidence IDs, answerability ve LLM
skipped/running kararı aynı request ID ile izlenir.

## System / Models

```bash
curl -sS http://127.0.0.1:8010/v1/system/profile
```

Bu response sanitized CPU/RAM/GPU bilgisi, Ollama erişilebilirliği, installed
models, role ayrımı ve heuristik compatibility sınıfı döndürür. Runtime
reachable fakat seçili generation model yoksa `generation_readiness.status`:
`model_missing`; runtime yoksa `runtime_unavailable` olur. Model pull yalnız
`DIS_LOCAL_MODEL_MANAGEMENT_ENABLED=true` ve configured catalog ile açılır.
