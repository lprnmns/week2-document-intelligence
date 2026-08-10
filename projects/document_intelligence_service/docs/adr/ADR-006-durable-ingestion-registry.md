# ADR-006: Durable ingestion registry for the local worker boundary

## Bağlam

Upload API'si `202 + job_id` döndürüyor. In-memory registry aynı process içindeki test ve demo için yeterliydi fakat ayrı worker process'i veya restart sonrası job, idempotency kaydı ve staged PDF kayboluyordu.

## Alternatifler

1. In-memory registry'yi koruyup API ile worker'ı aynı process'e zorlamak.
2. Local MVP için SQLite; daha yüksek hacimde ayrı metadata DB, object storage ve queue'ya geçmek.
3. İlk günden Redis/PostgreSQL/object storage zorunluluğu getirmek.

## Karar

`IngestionRegistry` portuna SQLite adapter eklendi. Content hash + pipeline fingerprint identity, idempotency key, job progress ve staged PDF aynı durable dosyada tutuluyor. API composition'ında adapter seçimi ayrı bir wiring adımı olarak kalacak; default test/demo adapter'ı in-memory olabilir.

## Ölçüm/kanıt

Bir registry instance'ı upload kabul ediyor; ikinci instance aynı SQLite dosyasından job, staged content ve idempotent receipt'i okuyabiliyor. Job status güncellemesi restart sonrası korunuyor.

## Bilinen sınır

SQLite local single-node MVP içindir. Çoklu worker, yüksek throughput ve büyük binary dosyalar için PostgreSQL/object storage/queue karşılaştırması yapılmadan production kararı verilmez.
