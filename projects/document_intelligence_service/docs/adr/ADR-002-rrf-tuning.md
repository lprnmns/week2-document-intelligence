# ADR-002: RRF parametresi ve tuning politikası

## Bağlam

Dense ve sparse rank listelerinde skor ölçekleri farklıdır. Ham skorları toplamak bir retriever'ı haksız biçimde öne çıkarabilir.

## Alternatifler

1. Ham dense/BM25 skorlarını toplamak.
2. Rank Fusion ile RRF kullanmak.
3. Öğrenilmiş ağırlıklı ranker kullanmak.

## Karar

Üretim/demo adayı dense top-30 + sparse top-30 → RRF top-20 akışıdır.
Reranker top-5 bounded bir ablation seçeneği olarak kalır. RRF sabiti ve
candidate limitleri settings üzerinden tutulacak; magic constant olarak
endpoint içine gömülmeyecektir.

## Ölçüm/kanıt

Golden set üzerinde Recall@k, MRR, nDCG ve p95 latency; query type ve dil
slice'larıyla raporlanacak. Temiz 44-vaka smoke'unda hybrid RRF Recall@5
`0.9233`, MRR@10 `0.8778`, nDCG@10 `0.9518`, p95 `42.74 ms` verdi. Hybrid +
reranker Recall@5 `0.9122`, MRR@10 `0.8333`, nDCG@10 `0.9329`, p95
`1485.39 ms` ölçüldü. Aynı snapshot'ta `8` positive ve `12` negative flip
vardır; reranker girdisi/çıktısı `20 → 5` ile sınırlandırılmıştır.

Bu nedenle local varsayılan `Hybrid RRF + reranker OFF` olarak sabitlenmiştir.
Bu, reranker'ların genel olarak kötü olduğu iddiası değildir; bu corpus,
model, candidate limitleri ve CPU latency bütçesi için ölçülmüş bir karardır.
Reranker yalnız gerçek ON/OFF ablation ve mentor teşhisi için açılabilir.

## Bilinen sınır

RRF iyi bir başlangıçtır; veri, embedding modeli veya host değiştiğinde
candidate limitleri ve reranker maliyeti yeniden ölçülmelidir. Reranker modeli
process içinde cache'lense de CPU-only host'ta ilk kullanım cold latency'si
`7113.62 ms`, warm p50 `1436.13 ms`, warm p95 `3084.34 ms` ölçülmüştür.
Ham ölçüm `eval/results/week2_stabilization_v1/reranker_latency.json` içindedir.
