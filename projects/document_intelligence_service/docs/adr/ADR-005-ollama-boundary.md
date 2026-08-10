# ADR-005: Ollama host/container sınırı

## Bağlam

Geliştirme makinesinde 32 GB RAM vardır. API container'ı içinde ikinci bir model
runtime'ı çalıştırmak bellek ve operasyon maliyetini artırır; bu nedenle Ollama
API image'ından ayrı tek bir local runtime olarak çalışır.

## Alternatifler

1. Ollama'yı API container'ına almak.
2. Ollama'yı ayrı host/container runtime'ında tutup API/worker'dan erişmek.

## Karar

Bu makinede mevcut `ai-journey-ollama` container'ı Compose ağına bağlıdır; API ve
worker `http://ai-journey-ollama:11434` üzerinden `gemma3:4b` modeline erişir.
Host üzerinde çalışan Ollama için `DIS_OLLAMA_URL` ile
`http://host.docker.internal:11434` override edilebilir. Model request başına
yüklenmeyecek; startup/warm-up ve latency ölçümleri ayrı raporlanacaktır.

## Ölçüm/kanıt

Warm/cold latency, resident memory, failure davranışı ve Firefox açıkken sistem stabilitesi ölçülecek.

## Bilinen sınır

Runtime erişim adresi işletim sistemi ve Docker network ayarlarına bağlıdır;
Compose health check bunu açıkça raporlamalıdır.
