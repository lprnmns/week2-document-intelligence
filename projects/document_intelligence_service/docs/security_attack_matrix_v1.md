# Security Attack Matrix v1

Generated: `2026-08-10`
Git SHA: `6e0c748354a755e073e99f27ce1fa663a1e42e5b`
Scope: Hafta 2 local MVP: query, evidence, ingestion ve API sınırları

Bu rapor tehdit sınıflarını mevcut kod ve test kanıtlarıyla eşleştirir. `pass` yalnız belirtilen dar kapsamın kanıtlandığını, genel güvenlik garantisi olmadığını ifade eder.

## Özet

- Kontrol sayısı: `8`
- `pass`: `5`
- `partial`: `3`
- `not_ready`: `0`
- Eksik kanıt yolu: `0`
- Release-ready: `False`

## Matris

| ID | Tehdit | Durum | Mevcut kontrol | Sonuç | Sonraki adım |
| --- | --- | --- | --- | --- | --- |
| SEC-01 | Direct prompt injection | `pass` | PromptSafetyPolicy retrieval ve LLM çağrısından önce çalışıyor. | Yüksek güvenli direct injection vakaları SECURITY_POLICY no-answer ile kesiliyor; generator çağrılmıyor. | Paraphrase, multilingual ve adversarial near-miss setini genişletip false-positive oranını ölçmek. |
| SEC-02 | System-prompt extraction | `pass` | System prompt, gizli kural ve iç talimat çıkarma kalıpları PromptSafetyPolicy içinde ayrı karar olarak ele alınıyor. | System-prompt gösterme ve kaynak dışı kesin iddia üretme talebi LLM'e ulaşmadan SECURITY_POLICY döndürüyor. | Extraction paraphrase seti, canary secret ve gerçek local-model smoke'u eklemek. |
| SEC-03 | Indirect injection in retrieved evidence | `pass` | EvidenceSafetyPolicy, evidence'ı structured prompt'a girmeden önce yüksek güvenli talimat kalıplarından filtreliyor. | 3/3 indirect saldırı çıkarıldı; benign prompt açıklaması 1/1 korundu; tüm evidence zararlıysa LLM çağrılmadı. | Malicious evidence ile gerçek Ollama smoke'u, provenance allowlist ve daha geniş adversarial corpus eklemek. |
| SEC-04 | Cross-document leakage / ACL bypass | `pass` | Tenant/ACL filtreleri application sınırında normalize edilir, Qdrant pre-filter olarak uygulanır ve dönen source metadata'sı ikinci kez doğrulanır. | Tenant veya ACL kapsamı olmayan candidate source listesine giremiyor; cross-tenant ve yetkisiz-tag contract/unit testleri geçiyor. Bu, authentication/authorization sistemi değil, local ACL-ready izolasyon kontrolüdür. | Bir sonraki kapsamda authenticated request principal, policy store ve signed provenance ile filtre değerlerinin istemciden beyan edilmesini kaldırmak. |
| SEC-05 | HTML/Markdown exfiltration and unsafe rendering | `pass` | API cevabı JSON sözleşmesi olarak döndürüyor; demo UI answer ve source alanlarını yalnız textContent ile gösteriyor, HTML/Markdown/link/image yorumlamıyor. | Response application/json; demo UI replaceChildren/textContent kullanıyor ve innerHTML, Markdown renderer, link veya image sink'i içermiyor. Bu nedenle HTML, URL ve image payload'ları local UI'da inert text olarak kalıyor. | Rich text ileride açılırsa external client'lar için server-side output policy ve URL/image scheme allowlist testlerini eklemek. |
| SEC-06 | RAG poisoning and untrusted source provenance | `partial` | Content hash, pipeline fingerprint, deterministic point ID ve stage → verify → activate akışı kullanılıyor. | Eksik/bozuk staged version active olmuyor; eski active version korunuyor. Ancak kaynak güven seviyesi, signed provenance ve içerik poisoning tespiti yok. | Kaynak allowlist/provenance metadata, quarantine, değişiklik incelemesi ve poisoning regression seti eklemek. |
| SEC-07 | DoS through oversized or expensive PDF upload | `partial` | Upload byte limiti, bounded read, MIME allowlist, %PDF magic-byte ve page limit kontrolleri pahalı parse öncesinde uygulanıyor. | 10 MiB ve 200 sayfa varsayılan sınırları; invalid MIME/signature, fazla boyut ve fazla sayfa testleri mevcut. | Rate/concurrency quota, bounded worker queue, parse timeout ve per-tenant resource budget ölçmek. |
| SEC-08 | Sensitive data leakage through query observability | `partial` | Trace ham soru ve evidence yerine request ID, question SHA-256, karar, skor ve latency metadata'sı tutuyor. | Raw query/evidence JSON trace'e yazılmıyor; retention, sampling ve log erişim politikası henüz repository içinde tanımlı değil. | Retention/sampling/access policy, düşük entropili alanlar için keyed hash ve PII redaction değerlendirmesi. |

## Okuma notu

Özellikle `SEC-04` için `pass`, local MVP kapsamındaki tenant/ACL pre-filter ve source re-check izolasyonunun test edildiği anlamına gelir; authentication veya merkezi authorization sistemi anlamına gelmez. Filtre değerleri şu an istemci tarafından beyan edilir ve bir sonraki kapsamda authenticated request principal ile bağlanmalıdır.
