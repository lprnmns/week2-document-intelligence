# Hafta 2 PDF Kabul Matrisi — Sayfa 1–28

Kaynak PDF: approved local Week-2 mentor PDF supplied separately; its private
filesystem path is intentionally omitted.
İnceleme kapsamı: 28 sayfanın metni, tabloları, şablonları ve diyagramları tek
tek kontrol edildi. Görsel diyagramların kod karşılıkları `architecture.md`,
`compose.yaml`, `demo_ui/index.html` ve bu matristeki kanıt yollarıyla eşleştirildi.

## Durum anlamı

- `PASS`: İstenen davranış kod, test veya yeniden üretilebilir artifact ile gösterildi.
- `PARTIAL`: Temel davranış var; PDF'in production/operasyon ayrıntısının bir kısmı açık sınır olarak kaldı.
- `NOT_READY`: İstenen davranış için güvenilir kanıt henüz yok; tamamlandı kabul edilmez.

## Sayfa bazlı eşleştirme

| Sayfa | PDF'teki söz/şablon/diyagram | Repo karşılığı ve kanıt | Durum | Kalan sınır / kontrol |
| ---: | --- | --- | :---: | --- |
| 1 | Ürünü ölçülebilir, kaynaklı ve yeniden kurulabilir küçük servise dönüştürme | `projects/document_intelligence_service/README.md`, Compose, API, eval artifacts | PASS | Son teslim SHA'sı final smoke sonrası yeniden yazılacak |
| 2 | Hafta 1 parçalarını tek uygulamada birleştirme; ana mühendislik sorusu | `app/application/`, `app/domain/`, `app/infrastructure/`, `architecture.md` | PASS | Scope local-first; HA/Kubernetes yok |
| 3 | Kanıt önce, sözleşme önce, ölçmeden karar yok, güvenli varsayılan, local-first | API contracts, golden set, security matrix, no-answer tests | PASS | Production retention/access policy dış sistem sınırı |
| 4 | Layered architecture diyagramı; API → Application → Domain; en az 3 ADR | `docs/architecture.md` katman diyagramı; `docs/adr/ADR-001..008` | PASS | PDF minimumunu aşan observability ve local-model ADR'leri mevcut |
| 5 | Query sequence diyagramı; normalize → dense/sparse → RRF → rerank → gate → LLM/no-answer; canonical source; LLM skip testi | `RetrievalService`, `QueryService`, `QueryTraceEvent`, contract/unit tests | PASS | Candidate limitleri trace count olarak görülüyor; ayrı config snapshotı docs'ta |
| 6 | Önerilen klasör ağacı ve dependency direction | `app/api`, `application`, `domain`, `infrastructure`, `observability`, `eval`, `tests` | PASS | PDF'teki bazı dosya adları port/adaptor tasarımına uyarlanmış |
| 7 | FastAPI lifespan/DI; CPU offload; live/ready/startup; upload `202 + job_id` | `app/main.py`, health routes, `asyncio.to_thread`, documents/jobs routes, Compose preload | PASS | Development/test ortamında preload açıkça kapatılabilir; Compose gerçek model preload ile çalışıyor |
| 8 | REST tablo: documents/jobs/queries/search/evaluation; pagination, idempotency, debug, delete conflict | `contracts.py`, routes, OpenAPI contract testleri | PASS | Delete `204`; aktif ingestion sırasında `409 DOCUMENT_BUSY` |
| 9 | Kararlı response modelleri, source evidence, hata taksonomisi, stack/system prompt/path sızıntısı yok | `contracts.py`, `api/errors.py`, contract tests | PASS | Response versioning bir sonraki API v2 kararı |
| 10 | Kabul → identity → parse → normalize → chunk → embed → stage → verify → activate | `ingestion_worker.py`, `ingestion.py`, job timeline, SQLite registry, `eval/verify_week2_pdf.py` | PASS | Eski version retention/purge politikası henüz manuel |
| 11 | Named dense/sparse Qdrant, deterministic point ID, payload/index, dimension validation | `qdrant/schema.py`, `chunk_store.py`, Qdrant unit/integration tests | PASS | Named sparse vector `bm25`; hash tabanlı `version_id` PDF'teki integer ingestion_version yerine daha güçlü eşdeğer sürüm kimliği |
| 12 | Metadata sınıfları; tenant → ACL → document → active filtre; source re-check; corpus snapshot; privacy | `QdrantRetriever`, `RetrievalService._filter_access`, `api/v1/scope.py`, reporting manifest, trace/audit | PASS | `X-Tenant-ID`/`X-ACL-Tags` local canonical scope'tur; authenticated principal yerine geçmez |
| 13 | Dense top-30 + sparse top-30 → RRF top-20 → rerank top-5 → parent evidence | `retrieval_service.py`, RRF tests, UI debug candidate trace | PASS | Reranker local CPU'da varsayılan kapalı; karar benchmark ile belgeli |
| 14 | BM25/dense/hybrid query slices; Recall/MRR/nDCG; p50/p95; 5 gain/5 loss | `../../../../docs/evaluation_method_slices.md`, `benchmark_report.md`, raw CSV/JSONL | PASS | 2026-08-10 clean snapshot: 44 vaka, 26 immutable point; BM25-vs-Dense disagreements are reported without cherry-picking |
| 15 | A/B/C/D reranker ablation; candidate recall; positive/negative flip; p95 gate | `run_ablation_report.py`, `eval/results/week2_stabilization_v1/ablation_summary_v2.json`, ADR-002 | PASS | Final snapshot'ta 8 positive/12 negative flip; input/output `20 → 5`; cold/warm latency ayrıca raporlandı |
| 16 | Run manifest, warm-up, random order, bootstrap CI, failure rate, latency budgets | `eval/reporting.py`, `eval/results/week2_stabilization_v1/run_manifest.json`, manifest/raw artifact tests | PASS | Manifest source SHA `90900ae`; corpus snapshot, dataset SHA, point count, pipeline fingerprint, model/config, host RAM, CI ve metric version alanları mevcut |
| 17 | 40+ dengeli golden set; evidence labels; adjudication; validation-only threshold; test leakage yok | `data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl` (44), split tests, `data/evaluations/mentor_program_blind_review_packet_v1.json`, calibration report | PARTIAL | Blind packet covers 10/44 and is explicitly `PENDING HUMAN REVIEW`; no human completion is claimed |
| 18 | Answerability sinyalleri, canonical source, no-answer reason, output warning | `answerability.py`, `query_service.py`, `contracts.py`, validation artifact ve testler | PARTIAL | Mentor/frozen policy dense threshold `0.337857395 → 0.338` ve test FP `0`, FN `2` olarak değişmedi. Generic `generic_v1` policy ayrı validation kalibrasyonuyla dense `0.247`, lexical coverage `0.367` kullanıyor; açık yıl/yüzde/tırnak qualifier coverage near-miss gate'i threshold değiştirmeden `INSUFFICIENT_COVERAGE` üretiyor. Genel semantik coverage hâlâ calibrated değildir. |
| 19 | Direct/indirect injection, leakage, unsafe render, poisoning, DoS; structured prompt; tool-off | `prompt_safety.py`, `evidence_safety.py`, security matrix, safe UI text rendering | PARTIAL | Frozen test security gate `4/4`; shipped UI'da HTML/Markdown/link/image sink'i yok ve payload text olarak inert kalıyor. Provenance allowlist/quarantine, authenticated principal, rate limit ve ingestion timeout production kapsamına bırakıldı |
| 20 | JSON log, metric, trace, audit, correlation ID, PII/cost policy | `observability/{query_trace,metrics,audit}.py`, `/v1/metrics`, worker stage logs | PARTIAL | Metrics process-local; retention/sampling/central collector yok |
| 21 | Compose diagramı: API, worker, Qdrant, UI, host Ollama, volume, healthcheck, limits | `compose.yaml`, `.env.example`, `scripts/compose_smoke.sh` | PASS | Ollama bağlantısı makine/network'e bağlı; smoke bunu açıkça kontrol ediyor |
| 22 | CI: lint/type/unit/contract/integration/security/eval/image/SBOM/scan; coding standard; PR fields | `.github/workflows/document-intelligence-service.yml`, pyproject, PR template | PASS | Full eval nightly/manual kapsamda tutuldu; her push'ta bounded dataset smoke çalışıyor |
| 23 | 5 günlük gate planı ve “evaluation/idempotency azaltılmaz” kuralı | `../../../../docs/demo_runbook_20min.md`, service tests and release manifest | PASS | Mentor demo planı ve reproducibility evidence actual tree'de tutuluyor |
| 24 | 10 teslim kalemi ve zorunlu acceptance kriterleri | Source, UI, Compose, 44 golden, raw results, benchmark, architecture/ADR, README, API examples, demo | PASS | Video teslimi yok; canlı demo akışı var |
| 25 | Review checklist ve 20 dakikalık demo: health → upload → query → no-answer → injection → benchmark | UI, smoke script, benchmark/security artifacts, `hafta2_uygulama_plani.md` demo tablosu | PARTIAL | Canlı akış ve smoke mevcut; tek komutla ekran görüntüsü/video capture otomasyonu yok |
| 26 | 100 puan rubrik ve critical fail koşulları | Duplicate, source provenance, ACL isolation, no-answer, clean setup testleri | PARTIAL | Mentor puanlaması insan değerlendirmesidir; kod bunu otomatik puanlamaz |
| 27 | 12 teknik görüşme sorusunu ölçüm/kod/alternatif/sınır ile cevaplama | `../../../../docs/mentor_technical_questions.md` | PASS | Cevaplar gerçek code path, ölçüm ve limitation ile bağlıdır; blind review ayrı olarak pending'dir |
| 28 | `.env.example`, 5 ADR, resmi kaynaklar, aynı Git SHA ile final teslim | `.env.example`, `docs/adr/ADR-001..008`, README/API docs, `run_manifest.json`, Week 2 marker profile | PARTIAL | Offline artifact'lar `90900ae` base revision'ını taşır. Container'da `.git` olmadığı için canlı SHA `DIS_SOURCE_REVISION` ile enjekte edilir; exact final tree için final commit/rebuild gerekir |

## Sonuç

Çekirdek ürün akışı ve 28 sayfanın ana teknik beklentileri repo içinde
kanıtlandı. Week 2 PDF'i için `mentor_program_week2_v1` profili 28 başlığı,
28 parent'ı ve 97 child'ı sayfa sınırlarıyla doğrular. `PARTIAL` satırlar
bilinçli sınırları gösterir: validation ile kalibre edilmemiş ikincil
answerability sinyalleri, production provenance ve authentication, merkezi
metrics/retention, mentorun insan değerlendirmesi ve artifact paketleme SHA'sı.
Bunlar tamamlandı gibi gösterilmeyecek; teslimde hangi kapsamın otomatik,
hangisinin manuel veya sonraki sprint olduğunu açıkça ayıracağız.
