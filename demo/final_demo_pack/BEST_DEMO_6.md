# Best Demo 6

Seçim, gerçek run sonuçları üzerinden yapıldı. Puanlar 1-5 arasındadır;
başarıyı yapay olarak artırmak için failure case'ler çıkarılmadı.

| Vaka | Reliability | Teaching | Visual clarity | Runtime | Mentor relevance | Risk | Seçim nedeni |
|---|---:|---:|---:|---:|---:|---:|---|
| `semantic_remote_days` | 5 | 5 | 5 | 4 | 5 | 1 | Temiz Dense semantic success |
| `exact_rollback_code` | 5 | 5 | 5 | 4 | 5 | 1 | Temiz BM25 exact identifier success |
| `release_time_and_rollback` | 2 | 5 | 5 | 2 | 5 | 4 | Gerçek multi-document failure attribution |
| `education_2024_absent` | 5 | 5 | 5 | 5 | 5 | 1 | Wrong-year near-miss, generation skip |
| `missing_rd_budget` | 5 | 4 | 4 | 5 | 5 | 1 | Unanswerable query ve no-answer gate |
| `security_injection` | 5 | 5 | 4 | 5 | 5 | 2 | Belge içi injection ve security policy |

`release_time_and_rollback` başarı örneği gibi anlatılmamalıdır. Gerçek
ölçümde cevap tarihi/saat verdi, rollback kodunu vermedi; trusted evidence
diagnostic'i `FUSION_LOSS` raporladı. Bu, mentorun “yanlışsa hangi katman?”
sorusuna en iyi canlı örnektir.

Reranker ablation için alternatif vaka `release_checklist`'tir. Aynı soru
OFF/ON ölçülmüş, ON koşulunda `RERANKER_LOSS` gözlenmiş ve ablation sonuçları
`results/demo_run_results.json` içinde saklanmıştır.
