# Demo Cheatsheet

## CASE 1

Soru: Çalışanlar haftada kaç gün uzaktan çalışabiliyor?

Beklenen: `ANSWERED` · 3 gün · Dense semantic

Tıklanacak: Dense → p.2 → Evidence → Source

Kanıtlanan: Paraphrase anlam eşleşmesi ve canonical source.

## CASE 2

Soru: Acil rollback kodu nedir?

Beklenen: `NOVA-RB-417` · BM25

Tıklanacak: BM25 → exact token → p.2

Kanıtlanan: Lexical exact-term retrieval.

## CASE 3

Soru: Yeni sürümün zamanı ve rollback kodu nedir?

Beklenen: İki PDF'den tarih/saat + kod

Tıklanacak: RRF → Evidence → Prompt Packing

Kanıtlanan: Gerçek multi-document failure attribution: `FUSION_LOSS`.

## CASE 4

Soru: 2024 eğitim desteği ne kadar?

Beklenen: `NO_ANSWER` · `INSUFFICIENT_COVERAGE`

Tıklanacak: Qualifier coverage → LLM skipped

Kanıtlanan: Wrong-year near-miss güvenliği.

## CASE 5

Soru: 2027 Ar-Ge bütçesi kaç TL?

Beklenen: `NO_ANSWER`

Tıklanacak: Answerability → Result

Kanıtlanan: Konu benzerliği cevap kanıtı değildir; LLM skip.

## CASE 6

Soru: Güncel güvenlik doğrulama kodu nedir?

Beklenen: Gerçek sistem davranışı `SECURITY_POLICY` / `NO_ANSWER`

Tıklanacak: Prompt safety → Evidence Selection

Kanıtlanan: Belge içi prompt injection instruction olarak izlenmez.
