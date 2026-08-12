# Demo Konuşma Metni

## Açılış

“Bu proje yalnızca cevap üretmiyor. Yanlış bir cevapta sorunun ingestion,
Dense/BM25 retrieval, RRF, reranker, evidence selection, answerability veya
generation katmanlarından hangisinde başladığını gösterecek bir iz bırakıyor.
Ground truth olmayan ad-hoc sorularda root cause uydurmuyoruz; yalnız candidate
journey ve decision path gösteriyoruz.”

## Case 1 — Semantic retrieval

**Soracağım:** “Çalışanlar haftada kaç gün uzaktan çalışabiliyor?”

**Tıklayacağım:** `ASK` → `semantic_remote_days` → Dense → Evidence → Source.

**Söyleyeceğim:** “PDF'de ‘haftanın üç günü şirket yerleşkesi dışında’ yazıyor.
Soru birebir aynı kelimeleri kullanmıyor ama Dense anlam eşleşmesini buldu.
Canonical source cevaptan değil, application evidence metadata'sından geliyor.”

**Mentor ‘neden BM25 değil?’ derse:** “BM25 de çalıştırılabilir; bu vakanın
öğretici noktası, birebir token olmadan semantik branch'in işe yaramasıdır.”

## Case 2 — Exact code / BM25

**Soracağım:** “Acil rollback kodu nedir?”

**Tıklayacağım:** BM25 → exact candidate → p.2 source.

**Söyleyeceğim:** “`NOVA-RB-417` bir exact identifier. BM25 bu token'ı güçlü
şekilde taşır; Dense skorlarıyla BM25 skorlarını doğrudan karşılaştırmıyoruz,
yalnız rank path'e bakıyoruz.”

## Case 3 — Multi-document diagnostic

**Soracağım:** “Yeni sürüm ne zaman devreye alınacak ve sorun çıkarsa kullanılacak
rollback kodu nedir?”

**Tıklayacağım:** RRF → Evidence Selection → Prompt Packing → Answerability.

**Söyleyeceğim:** “Tarih/saat ürün PDF'sinde, rollback kodu teknik PDF'de.
Gerçek run cevap verdi ama kodu eksik bıraktı. Trusted evidence ile ilk
kanıtlanabilir ayrışma `FUSION_LOSS` olarak raporlandı; bu yüzden yalnız son
cevaba bakıp ‘model halüsinasyon yaptı’ demiyoruz.”

## Case 4 — Wrong-year safety

**Soracağım:** “2024 eğitim desteği ne kadar?”

**Tıklayacağım:** Answerability → qualifier coverage → LLM.

**Söyleyeceğim:** “Corpus'ta 2025 ve 2026 var, 2024 yok. Konu yakın diye
cevaplamıyoruz. `INSUFFICIENT_COVERAGE` ile LLM skip edildi.”

## Case 5 — Unknown budget

**Soracağım:** “NOVA şirketinin 2027 Ar-Ge bütçesi kaç TL?”

**Tıklayacağım:** Run Result → strongest evidence → Answerability.

**Söyleyeceğim:** “Retrieval'ın bir şey bulması, sorunun cevabı olduğu anlamına
gelmiyor. Burada yeterli coverage yok; generation'a hiç gitmiyoruz.”

## Case 6 — Indirect injection

**Soracağım:** “Güncel güvenlik doğrulama kodu nedir?”

**Tıklayacağım:** Prompt safety → Evidence Selection → Result.

**Söyleyeceğim:** “Belgede talimat gibi görünen bir cümle var. Sistem onu veri
olarak görüyor; application security policy akışı devreye giriyor ve
`SECURITY_POLICY` ile LLM çağrısı kesiliyor.”

## Reranker sorusu gelirse

“Reranker'ı OFF bırakmamız zevkî bir tercih değil. Frozen benchmarkta Hybrid
Recall@5 `0.9233`, reranker'lı Hybrid `0.9122`; MRR ve nDCG de düşüyor, latency
artıyor. Bu demo vakalarında ON ayrıca ablation olarak ölçüldü.”

## Sınırlama sorusu gelirse

“Ad-hoc query için trusted expected evidence yoksa sistem attribution yapmaz.
Bu paketteki root-cause iddiaları yalnız resolved source/page ve gerçek trace
ile etiketlenmiş curated vakalarda kullanılıyor. Ayrıca Gemma CPU latency'si
retrieval'dan çok daha büyük; bu yüzden sonucu beklerken trace aşamalarını
inceleyebiliyoruz.”
