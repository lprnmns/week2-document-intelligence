# Hafta 2 PDF — Sayfa Sayfa Görsel ve Şablon İnceleme Kaydı

Kaynak: approved local Week-2 mentor PDF supplied separately; its private
filesystem path is intentionally omitted.

Görsel doğrulama: PDF bilgisi `28` sayfa; tüm sayfalar `pdftoppm -r 150`
ile ayrı PNG olarak render edildi ve `page-01.png`–`page-28.png` aralığı tek
tek incelendi. Kaynak PDF SHA-256: `df95170524478fbea62140bc76b97e521749d9f0c3928a4a64769f39ba7aee19`.

Tekrar üretilebilir yapısal kontrol:

```bash
PYTHONPATH=. .venv/bin/python -m \
  projects.document_intelligence_service.eval.verify_week2_pdf \
  /path/to/approved_week2_mentor.pdf
```

Bu kontrol 28 page heading marker'ını, 28 parent'ı ve child chunk'ların
`1..28` sayfa aralığı metadata'sını doğrular; görsel incelemenin yerine değil,
aynı kabul kaydını yeniden çalıştıran teknik tamamlayıcıdır.

Bu kayıt 28 sayfanın metin akışını, tablolarını, kod şablonlarını, mimari/sequence
diyagramlarını, checklist ve teslim tablolarını tek tek repo kanıtlarıyla
eşleştirir. `PASS`, ilgili öğenin kod/test/artefact ile gösterildiğini; `BOUNDARY`
ise PDF'in production kapsamına taşan kısmın bilinçli olarak açık bırakıldığını
belirtir.

| Sayfa | Görsel/şablon öğesi ve beklenen mesaj | Repo karşılığı | Durum |
| ---: | --- | --- | :---: |
| 1 | Kapak; 5 günlük local-first ürünleştirme hedefi | `README.md`, Week 2 service README | PASS |
| 2 | “İlk haftayı tekrar etme”; servis hedefi ve ana mühendislik sorusu | `docs/architecture.md`, layered service, benchmark/eval docs | PASS |
| 3 | İlke tablosu: kanıt, sözleşme, ölçüm, tekrar üretim, güvenlik, local-first | API contracts, manifests, security matrix, README | PASS |
| 4 | Layered architecture diyagramı ve “kesinlikle yapmamalı” tablosu | `docs/architecture.md`; API/application/domain/infrastructure sınırları | PASS |
| 5 | Query sequence diyagramı: normalize → dense/sparse → RRF → rerank → gate → LLM/no-answer | `RetrievalService`, `QueryService`, `QueryTraceEvent`, LLM-skip testleri | PASS |
| 6 | Klasör ağacı ve dependency-direction notu | `app/`, `eval/`, `tests/`, `docs/adr/`; import sınırları | PASS |
| 7 | FastAPI lifespan kod şablonu, üç health endpointi ve 202 upload notu | `main.py`, health routes, preload, `POST /v1/documents` | PASS |
| 8 | REST kaynak tablosu, query JSON örneği, pagination/idempotency/debug kuralları | `contracts.py`, document/job/query/search/evaluation routes, OpenAPI tests | PASS |
| 9 | QueryResponse/SourceEvidence sınıf şablonu ve hata taksonomisi tablosu | Stable Pydantic contracts, unified error mapping, response leak tests | PASS |
| 10 | Kabul → identity → parse → normalize → chunk → embed → stage → verify → activate pipeline tablosu ve fingerprint formülü | `IngestionWorker`, SQLite registry, `PipelineConfig`, Qdrant verify/activate, `eval/verify_week2_pdf.py` | PASS |
| 11 | Named dense/sparse Qdrant şema çizimi, payload alanları ve index tablosu | `QdrantSchema`, `QdrantChunkStore`, schema/integration tests; `dense` + `bm25` | PASS |
| 12 | Metadata sınıfları, filtre sırası ve privacy notları | tenant/ACL/document/active pre-filter, `X-Tenant-ID`/`X-ACL-Tags` scope çözümleme, source re-check, trace hash/audit | PASS |
| 13 | Dense top-30 + sparse top-30 → RRF top-20 → rerank top-5 tablosu | `RetrievalService`, bounded limits, RRF trace, reranker disabled default | PASS |
| 14 | Benchmark metric tablosu, query slice ve latency budget beklentileri | 44-case raw CSV/JSONL, Recall/MRR/nDCG, p50/p95, slice report | PASS |
| 15 | A/B/C/D reranker ablation ve gain/loss beklentisi | `week2_report_v2/ablation_matrix.csv`, `reranker_flips.jsonl`, summary | PASS |
| 16 | Run manifest, warm-up, random order, bootstrap CI, failure-rate ve latency bütçesi şablonu | `run_manifest.json`, reporting helpers, raw observations, bootstrap slice CI | PASS |
| 17 | 40+ balanced golden set tablosu, JSONL kayıt şablonu, validation/test leakage kuralı | 44 vaka; category/split contract; calibration yalnız validation’da | PASS |
| 18 | Answerability sinyal tablosu ve decision tree; canonical source/output warning sözleşmesi | `AnswerabilityPolicy`, coverage trace, evidence coverage artifact, canonical sources, numeric warnings | PASS* |
| 19 | Threat matrix ve structured prompt DATA/QUESTION ayrımı | prompt/evidence safety, security matrix, untrusted evidence prompt, safe UI text rendering | PASS* |
| 20 | JSON query event örneği, log/metric/trace/audit sinyal tablosu ve PII/maliyet notu | `query_trace.py`, `metrics.py`, `audit.py`, `/v1/metrics`, stage timeline | PASS* |
| 21 | Compose topoloji diyagramı: API, worker, Qdrant, UI, Ollama host, volume, healthcheck, limits | `compose.yaml`, named volumes, resource limits, smoke script | PASS |
| 22 | CI pipeline tablosu ve coding-standard maddeleri | GitHub workflow: lint/type/unit/contract/integration/security/eval/image/SBOM | PASS |
| 23 | 5 günlük gate tablosu ve “evaluation/idempotency azaltılmaz” kuralı | `docs/internship/week2/hafta2_uygulama_plani.md` ve acceptance matrix | PASS |
| 24 | 10 teslim kalemi ve zorunlu acceptance maddeleri | Source/UI/Compose/dataset/raw/benchmark/architecture/README/API/demo | PASS |
| 25 | Review checklist ve 20 dakikalık demo timeline’ı | Live Compose smoke + UI + demo flow; video capture insan tarafından yapılır | BOUNDARY |
| 26 | 100 puan rubriği, seviye bantları ve critical-fail satırı | Test gates ve review checklist; mentor puanı otomatikleştirilmez | BOUNDARY |
| 27 | 12 teknik görüşme sorusu | `../../../../docs/mentor_technical_questions.md` | PASS |
| 28 | `.env.example`, 5 ADR, resmi kaynaklar ve final SHA teslim notu | `.env.example`, ADR-001..007, README links, manifests | PASS* |

## Açık sınırlar

`PASS*` local-first MVP kapsamının otomatik kanıtlandığını gösterir. Merkezi
metrics/retention, authenticated request principal, provenance allowlist ve
quarantine, URL/image sanitizer, video kaydı ve mentorun insan puanı repo dışı
veya sonraki operasyon kapsamıdır. Bunlar acceptance matrix’te saklanmaz; açıkça
`PARTIAL`/`BOUNDARY` olarak gösterilir.

Artifact’ler source kod SHA’sını manifest içinde taşır. Artifact dosyalarının
kendilerini commit eden paketleme commit’i doğal olarak farklı bir SHA olur;
Git’in kendi hash tanımı nedeniyle tek bir dosyanın “içinde bulunduğu son commit
SHA’sını” self-referential biçimde taşıması mümkün değildir.
