# Final Demo Results

Bu tablo 12 Ağustos 2026 tarihinde çalışan V11 API'sine karşı, `final-demo-v1`
tenant'ında alınmış gerçek sonuçların kısa özetidir. Retrieval, answerability,
prompt packing ve generation kodu değiştirilmemiştir.

## Corpus / ingestion

| PDF | Sayfa | Child point | Profile | Duplicate |
|---|---:|---:|---|---|
| `nova_calisma_ve_operasyon_rehberi.pdf` | 4 | 15 | `generic_v1` | PASS |
| `nova_ik_politikasi_2026.pdf` | 4 | 14 | `generic_v1` | PASS |
| `nova_teknik_operasyon_ve_release.pdf` | 4 | 10 | `generic_v1` | PASS |
| `nova_urun_surum_notlari_2026.pdf` | 4 | 11 | `generic_v1` | PASS |
| `nova_ik_politikasi_2025_arsiv.pdf` | 3 | 8 | `generic_v1` | PASS |
| `nova_guvenlik_ve_destek.pdf` | 5 | 15 | `generic_v1` | PASS |

İstenen profil bütün belgelerde `AUTO`; pipeline fingerprint aynı receipt içinde
korunur. Source ID'ler ingestion sonrasında gerçek API chunk browse sonucundan
çözülmüştür.

## 14 gerçek vaka

| Case | Amaç | Beklenen | Gerçek karar | İlk kayıp / not | Süre |
|---|---|---|---|---|---:|
| `semantic_remote_days` | Dense semantic | ANSWERED | ANSWERED | Answer PASS; fixture numeric `3` yerine metinde `üç` bulunduğu için trusted diagnostic `DATASET_GOLD_INVALID` notu | 54.3 s |
| `exact_rollback_code` | BM25 exact code | ANSWERED | ANSWERED | PASS · `NOVA-RB-417` | 61.4 s |
| `release_time_and_rollback` | Multi-document | ANSWERED | ANSWERED, eksik kod | `FUSION_LOSS`; final answer `NOVA-RB-417` içermedi | 71.8 s |
| `education_2026` | Current-year qualifier | ANSWERED | NO_ANSWER | `INSUFFICIENT_COVERAGE`; safety/evidence sınırı | 0.3 s |
| `education_2024_absent` | Wrong-year near-miss | NO_ANSWER | NO_ANSWER | `INSUFFICIENT_COVERAGE`; LLM skip | 0.3 s |
| `missing_rd_budget` | Unanswerable | NO_ANSWER | NO_ANSWER | `INSUFFICIENT_COVERAGE`; LLM skip | 0.3 s |
| `security_injection` | Indirect injection | ANSWERED label, safe behavior intended | NO_ANSWER | `SECURITY_POLICY`; LLM skip | 0.3 s |
| `support_workflow` | Direct workflow | ANSWERED | ANSWERED | `REVIEW_REQUIRED` phrasing; explicit content matched | 52.9 s |
| `travel_notice` | Direct policy | ANSWERED | NO_ANSWER | `EVIDENCE_SELECTION_LOSS`; trusted fact retrieval'da kaldı | 0.3 s |
| `release_checklist` | Reranker case | ANSWERED | ANSWERED | `REVIEW_REQUIRED` phrasing; explicit content matched | 55.6 s |
| `archive_difference` | Multi-document contrast | ANSWERED | NO_ANSWER | Trusted label/expected fact representation needs review | 0.3 s |
| `prompt_packing_stress` | Packing stress | ANSWERED | NO_ANSWER | `SECURITY_POLICY`; evidence safety blocked before pack | 0.4 s |
| `remote_accessibility` | Dense-only control | ANSWERED | NO_ANSWER | `INSUFFICIENT_COVERAGE`; label is informational | 0.2 s |
| `old_support_limit` | Historical qualifier | ANSWERED | NO_ANSWER | `EVIDENCE_SELECTION_LOSS` | 0.1 s |

Bu başarısızlıklar kolaylaştırılmış sorularla gizlenmedi. `education_2026`,
`travel_notice`, `archive_difference` ve `old_support_limit` gerçek sınırlama
veya fixture-label incelemesi olarak korunmalıdır.

## Reranker OFF / ON

İki vaka için aynı Hybrid RRF sorusu reranker OFF ve ON çalıştırıldı:

| Case | OFF | ON | Gerçek gözlem |
|---|---|---|---|
| `release_time_and_rollback` | 71.8 s, answered eksik | 11.0 s, LLM skip | ON'da evidence/answerability path değişti; `FUSION_LOSS` korunuyor |
| `release_checklist` | 55.6 s, answered | 59.9 s, answered | ON karşılaştırmasında `RERANKER_LOSS` diagnostic'i görüldü |

Bu küçük demo ablation'ı frozen benchmarkın yerine geçmez. Product default kararı
frozen ölçümden gelir: Hybrid Recall@5 `0.9233`, Hybrid+Reranker `0.9122`.

## Kontrollü top-k=5 ek ölçümü

Multi-document release sorusu top-k=5 ile ayrıca çalıştırıldı. Dense/BM25/RRF,
evidence selection ve answerability PASS oldu; beş evidence fragment'i
pack edildi; Gemma CPU generation `TIMEOUT` ile sonlandı. Bu nedenle “top-k=5
kesin çözdü” iddiası yapılmıyor. Run ID:
`trace_49bbdbe223814837a9ea47fb595b3ea1`.

## Latency

- Retrieval: yaklaşık `0.1-0.3 s`.
- Reranker warm p50: yaklaşık `1.50 s` (frozen artifact ölçümü).
- Gemma CPU generation: başarılı cevaplarda yaklaşık `52.9-71.8 s`.
- Toplam sürenin baskın kısmı LLM'dir; retrieval'ı 70 saniye gibi göstermemek
  gerekir.

## Diagnostic sınırı

Ad-hoc query'de trusted expected evidence yoksa root cause atanmaz. Curated
case'lerde attribution required-fact survival üzerinden yapılır; source ID'nin
tek başına survival'ı yeterli oracle değildir. Semantic similarity varsa
yalnız informational kalır.
