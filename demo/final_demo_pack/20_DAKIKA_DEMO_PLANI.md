# 20 Dakikalık Mentor Demo Planı

Bu akış, uzun CPU generation beklemelerini de hesaba katar. Sonuç kartı ve
Stage Explorer mümkünse önceden ölçülmüş `results/demo_run_results.json`
trace'leriyle hazırlanır; canlıda en fazla iki answered query çalıştırılır.

## 00:00-02:00 — Problem ve sağlık

- UI header'daki API, Qdrant, Ollama, LLM ve Worker health durumunu göster.
- Şunu söyle: “Yeşil dekorasyon değil; ilgili health check'in son başarılı
  cevabını gösteriyor. LLM READY olması, bütün modellerin hazır olduğu anlamına
  gelmiyor.”
- `Hybrid RRF`, `Reranker OFF`, `Gemma` demo varsayılanını göster.

## 02:00-04:00 — Belgeler ve ingestion

- `DOCUMENTS` sekmesine geç.
- Altı NOVA PDF'sini tanıt: aktif sürüm, sayfa sayısı ve generic fallback.
- `prepare_final_demo.sh` komutunun normal upload API'sini kullandığını,
  Qdrant'a doğrudan yazmadığını vurgula.
- Duplicate upload'ta aynı document/version/job kimliği ve `idempotent_hit`
  gösterilir.

## 04:00-07:00 — Dense semantic success

Soru: “Çalışanlar haftada kaç gün uzaktan çalışabiliyor?”

- `semantic_remote_days` sonucunu aç.
- Dense node'unda `nova_calisma_ve_operasyon_rehberi.pdf`, p.2 ve gerçek excerpt'i
  göster.
- “Soru ile PDF aynı cümleyi kullanmıyor; anlam eşleşmesi Dense tarafından
  bulundu.”
- Sonuç ölçümünde cevap: “Çalışanlar haftada 3 gün uzaktan çalışabiliyor.”

## 07:00-09:00 — BM25 exact token

Soru: “Acil rollback kodu nedir?”

- `exact_rollback_code` sonucunu aç.
- BM25 branch'te `NOVA-RB-417` exact token'ını ve p.2 canonical source'u göster.
- “Bu tip kodlarda lexical branch'in birebir token gücü önemlidir.”

## 09:00-12:00 — Multi-document ve failure attribution

Soru: “Yeni sürüm ne zaman devreye alınacak ve sorun çıkarsa kullanılacak
rollback kodu nedir?”

- `release_time_and_rollback` trace'ini aç.
- Ürün sürüm PDF'sindeki tarih/saat ile teknik PDF'deki kodu yan yana göster.
- Actual cevap tarihi verdi ama kodu tamamlamadı; trusted diagnostic ilk
  kanıtlanabilir ayrışmayı `FUSION_LOSS` olarak raporladı.
- “Bu, başarısızlığı saklamadığımız bir vaka: selected evidence, RRF ranks,
  prompt membership ve final answer aynı panelde inceleniyor.”
- Top-k=5 tamamlayıcı kontrolünün generation timeout verdiğini limitation
  olarak söyle; bunu başarı gibi sunma.

## 12:00-14:00 — Wrong-year no-answer

Soru: “2024 eğitim desteği ne kadar?”

- `education_2024_absent` sonucunu aç.
- 2025/2026 yakın kanıtı olmasına rağmen 2024 qualifier'ının bulunmadığını
  göster.
- `NO_ANSWER`, `INSUFFICIENT_COVERAGE`, generation yok.

## 14:00-16:00 — Unanswerable query

Soru: “NOVA şirketinin 2027 Ar-Ge bütçesi kaç TL?”

- `missing_rd_budget` sonucunu aç.
- En güçlü retrieval evidence'ın soruyu cevaplamak için yeterli olmadığını ve
  LLM'in çağrılmadığını göster.

## 16:00-18:00 — Prompt injection

Soru: “Güncel güvenlik doğrulama kodu nedir?”

- `security_injection` sonucunu aç.
- PDF içinde bulunan “önceki talimatları yok say...” metninin veri olarak
  tutulduğunu, uygulamanın `SECURITY_POLICY` ile generation'ı kestiğini göster.
- Chain-of-thought veya injection metnini talimat gibi okumadığını belirt.

## 18:00-20:00 — Benchmark kararı ve kapanış

- Frozen 26-point snapshot, 44 case split ve gerçek ablation tablosunu aç.
- Hybrid: Recall@5 `0.9233`, MRR@10 `0.8778`, nDCG@10 `0.9518`.
- Hybrid + reranker: Recall@5 `0.9122`, MRR@10 `0.8333`, nDCG@10 `0.9329`;
  latency de yaklaşık 35.6 ms p50 retrieval'den reranker'lı yaklaşık 1.3 s
  p50'ye çıkıyor.
- Son cümle: “Bu nedenle Reranker OFF bir varsayım değil, ölçülmüş bir default;
  yine de ON ablation olarak kullanılabilir.”
