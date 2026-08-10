# ADR-007: Local observability and privacy boundary

## Bağlam

Mentor programı bir yanlış cevabın ingestion, retrieval, reranker, answerability,
prompt veya model katmanından geldiğinin kanıtla ayrılmasını istiyor. 32 GB RAM'li
local makinede merkezi telemetry stack kurmak bu haftanın zorunlu kapsamı değil;
ancak log/metric/trace/audit sinyalleri görünür olmalı.

## Alternatifler

1. Yalnız `print` ve ham soru/evidence loglamak.
2. İlk günden OpenTelemetry + Prometheus + merkezi collector kurmak.
3. Privacy-safe JSON trace, bounded process-local metrics ve lifecycle audit ile
   exporter sınırını sonraki haftaya bırakmak.

## Karar

Üçüncü seçenek seçildi:

- Query trace `request_id`, question hash, decision/reason, candidate sayıları,
  answerability sinyalleri ve `embed/search/rerank/llm/total` sürelerini taşır.
- `/v1/metrics` process-local counter ve bounded latency sample snapshot'ı döner.
- Ingestion job response'u stage duration, input/output özeti, decision ve retry
  bilgisini taşır.
- `document.audit` kabul, activate, fail ve delete olaylarını document/version/
  action/result kimlikleriyle kaydeder.
- Raw soru, prompt, PDF veya chunk metni varsayılan log alanına girmez; UI de
  text-only rendering kullanır.

## Ölçüm/kanıt

`tests/unit/test_query_trace.py`, `tests/unit/test_metrics.py`,
`tests/unit/test_audit.py`, `/v1/metrics`, `/v1/jobs/{job_id}` ve
`docs/architecture.md` trace/audit sınırını doğrular.

## Sonuçlar

Bir query'nin no-answer kararı ve LLM skip'i, stage süreleri ve reason code ile
ayrıştırılabilir. Worker ile API ayrı process olduğundan metrics registry'leri
şimdilik process-local'dır; worker'ın kalıcı operasyon kanıtı job timeline ve
structured loglardır.

## Bilinen sınır

Retention, sampling, log erişim politikası, merkezi Prometheus/OpenTelemetry
exporter ve authenticated audit reader sonraki operasyon kapsamıdır. SHA-256
question hash düşük entropili sorular için tam anonimlik garantisi değildir.
