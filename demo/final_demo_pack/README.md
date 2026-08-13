# NOVA Final Demo Pack

Bu klasör, Week-2 Document Intelligence servisinin 20 dakikalık mentor
sunumunda kullanılmak üzere hazırlanmış, tamamen kurgusal ve Türkçe bir
çok-doküman corpus'udur. PDF'ler NOVA Yazılım ve Teknoloji A.Ş. adını kullanan
demo materyalleridir; gerçek şirket, kişi, erişim kodu veya kullanıcı belgesi
içermez.

## Ne gösterir?

- Aynı corpus içinde doğru PDF ve sayfanın bulunması.
- Dense semantic, BM25 exact-term ve Hybrid RRF davranışının ayrılması.
- İki PDF'den gereken bilgilerin birleştirilmesi.
- Yıl/numara near-miss durumlarında `NO_ANSWER` ve LLM skip.
- Belge içindeki prompt-injection metninin veri olarak kalması.
- Trusted evidence verildiğinde ilk kanıtlanabilir kaybın Stage Explorer'da
  görünmesi.
- Reranker ON/OFF karşılaştırmasının gerçek rank hareketleri ve latency ile
  incelenmesi.

## Corpus

| PDF | Sayfa | Demo rolü |
|---|---:|---|
| `nova_calisma_ve_operasyon_rehberi.pdf` | 4 | Semantik çalışma modeli, seyahat, erişim |
| `nova_ik_politikasi_2026.pdf` | 4 | Güncel eğitim desteği ve başvuru politikası |
| `nova_teknik_operasyon_ve_release.pdf` | 4 | `NOVA-RB-417`, rollback ve incident prosedürü |
| `nova_urun_surum_notlari_2026.pdf` | 4 | 18 Eylül 2026 22:30 release bilgisi |
| `nova_ik_politikasi_2025_arsiv.pdf` | 3 | 2025 near-miss ve tarihli arşiv karşılaştırması |
| `nova_guvenlik_ve_destek.pdf` | 5 | `NOVA-SEC-882`, güvenlik akışı ve injection verisi |

PDF'ler normal ürün akışındaki `POST /v1/documents` API'si üzerinden
`AUTO` ingestion ile yüklenir. Demo tenant'ı `final-demo-v1` olarak ayrıdır;
frozen benchmark veya diğer tenant'lardaki belgeler silinmez.

## Hazırlama

Normal Compose startup bu corpus'u otomatik olarak hazırlar: `demo-seed`
container'ı altı PDF'yi normal `POST /v1/documents` API'siyle `final-demo-v1`
tenant'ına yükler, ingestion job'larını bekler ve UI'yi ancak tamamlandıktan
sonra açar. Zaten aynı içerikte aktif sürüm varsa dosya atlanır.

Explicit reset/re-ingestion gerektiğinde:

Servisler ayaktayken:

```bash
./scripts/prepare_final_demo.sh
./scripts/verify_final_demo.sh
```

İlk komut yalnız `final-demo-v1` tenant'ındaki aynı adlara sahip demo
belgelerini temizler, altı PDF'yi yeniden yükler, duplicate upload idempotency
kontrolünü yapar ve readiness'ı doğrular. İkinci komut ölçülmüş sonuçları,
no-answer kararlarını, UI'ı ve artifact hijyenini hızlıca kontrol eder.

Tüm 14 vakayı yeniden ölçmek için:

```bash
python3 scripts/run_final_demo.py --tenant final-demo-v1
```

Bu çalışma Gemma CPU generation nedeniyle birkaç dakika sürebilir. Üretim
pipeline'ı değiştirilmez; demo runner yalnızca HTTP API istemcisidir.

## Soru seti

14 vaka `demo_cases.json` içinde tutulur. Her vakanın amacı, beklenen karar
tipi, required facts'i ve trusted document/page bilgisi bulunur. Ingestion
tamamlandıktan sonra gerçek `source_id` değerleri API'den çözülür; sahte ID
üretilmez.

Son ölçülmüş sonuçlar `results/DEMO_RESULTS.md` ve ayrıntılı trace verisi
`results/demo_run_results.json` içindedir. `expected_answer` veya trusted
source, retrieval/generation akışına gönderilmez; yalnızca tamamlanmış run'ın
deterministik karşılaştırmasıdır.

## Önerilen mentor akışı

1. `semantic_remote_days`: Dense semantic retrieval ve canonical source.
2. `exact_rollback_code`: BM25 ile birebir kod araması.
3. `release_time_and_rollback`: iki dokümanlı vaka ve gerçek failure trace.
4. `education_2024_absent`: yakın yıl bilgisi varken güvenli no-answer.
5. `missing_rd_budget`: konu dışı bilgi için LLM skip.
6. `security_injection`: belge içi talimat benzeri metnin güvenlik politikasında
   bloke edilmesi.

Bu altı vakanın sunum metni `BEST_DEMO_6.md`, tam zaman çizelgesi
`20_DAKIKA_DEMO_PLANI.md`, doğal konuşma metni ise `DEMO_KONUSMA_METNI.md`
dosyasındadır.

## Gerçek ölçüm sınırı

Bu makinede retrieval yaklaşık 0.1-0.3 saniye, Gemma CPU generation yaklaşık
53-72 saniyedir. Tek belgeli vakalar runner tarafından top-k=1, çok belgeli
vakalar top-k=2 ile çalıştırılmıştır; bu, retrieval algoritmasını değil canlı
demo süresini sınırlar. Kontrollü bir top-k=5 release denemesi retrieval ve
answerability sonrasında LLM timeout ile bitmiştir. Bu durum sonuçlarda
başarıya çevrilmemiş, açık limitation olarak korunmuştur.

Reranker ölçümü `reranker_ablation` altında gerçek ON koşularıyla saklanır.
Frozen 26-point benchmarka, threshold'lara, chunking'e veya core retrieval
koduna bu demo paketi için müdahale edilmemiştir.
