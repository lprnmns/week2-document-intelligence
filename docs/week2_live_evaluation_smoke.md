# Week-2 final live Compose evidence

Bu kayıt final mentor demo durumunu anlatır. Çalıştırma tarihi: 2026-08-10.
Final kalite iddiası tek corpus, tek pipeline ve tek evaluation snapshot üzerine
kuruludur.

## Final corpus

- collection: `document_chunks_week2_final_v1`
- frozen snapshot points: `26` (the live collection contains additional product points)
- source: Week-1 mentor program PDF’i, content hash
  `b20e5ee9255db127f8394092773d5e1c17b5a9e258849e82db27273d44fe9898`
- pipeline fingerprint: `132e52a3e8358e66906a7dd9bcfd0c8b57aa228dd3102e9b3d8f39ccfb4c41a4`
- corpus snapshot: `c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25`
- immutable membership manifest: `data/evaluations/week2_final_corpus_snapshot_v1.json`
- membership boundary: exact point-ID list; document/version membership is recorded
  in the manifest
- dataset: `mentor_program_pdf_rag_golden_v1`, 44 cases
- split: development `19`, validation `11`, test `14`
- metadata verifier: `active_points=26`, `missing_required_metadata=0`
- metadata identities: one pipeline fingerprint, one embedding model, one
  sparse encoder and one ingestion version
- source revision in offline manifest: `90900ae35b167100f9eef9d6b759e3b4b38a2c38`

The older `document_chunks_v2_bm25` collection with 135 legacy points and its
Qdrant snapshot remain preserved, but are not used by the final Compose default
or benchmark claim.

## Retrieval ablation on the same snapshot

| Variant | Recall@5 | MRR@10 | nDCG@10 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| Dense | 0.9011 | 0.8750 | 0.9296 | 38.65 ms | 50.84 ms |
| BM25 | 0.8178 | 0.7844 | 0.8376 | 7.57 ms | 10.76 ms |
| Hybrid RRF | 0.9233 | 0.8778 | 0.9518 | 35.59 ms | 42.74 ms |
| Dense + reranker | 0.9122 | 0.8361 | 0.9329 | 1285.38 ms | 1430.38 ms |
| Hybrid + reranker | 0.9122 | 0.8333 | 0.9329 | 1325.10 ms | 1485.39 ms |

Raw outputs, slices, rank traces and manifest:
`projects/document_intelligence_service/eval/results/week2_stabilization_v1/`.

Live API evaluation run IDs against this exact snapshot:

```text
Hybrid / reranker OFF: eval_4da2f59af5934582b527d4168c593603
Dense  / reranker OFF: eval_76a6ef12de044fff802cc00ae323f070
BM25   / reranker OFF: eval_0f50102ff6924691a95fc255c4824882
Hybrid / reranker ON:  eval_08bd5a6a2e0e438b9fdb8a37ca00de38
Dense  / reranker ON:  eval_60cfeb57fa4941ff9475d8b0d649e2c2
```

Final rebuilt-image test-split smoke: `eval_99d3f4e597984747acc270676d39c8cc`.
It succeeded with `case_count=14`, `Recall@5=0.9583`, retrieval p50
`118.74 ms`, retrieval p95 `132.84 ms`, and live `git_sha=
90900ae35b167100f9eef9d6b759e3b4b38a2c38`.

Latest final-stack recheck after the clean-start Compose smoke:
`eval_1e2580fd78f04cd493c2d1cfca98d0b8`. It succeeded with the same
snapshot, `case_count=14`, `Recall@5=0.9583`, `MRR@10=1.0000`,
`nDCG@10=0.9834`, retrieval p50 `116.76 ms`, retrieval p95 `123.42 ms`,
and `git_sha=90900ae35b167100f9eef9d6b759e3b4b38a2c38`. The API artifact is
stored under `/data/evaluation_runs` in Compose and records the same corpus
configuration and source revision.

Each run response recorded the dataset SHA, snapshot ID, collection, active
point count and pipeline fingerprint. The Compose smoke now injects
`DIS_SOURCE_REVISION` from `git rev-parse HEAD`; direct image starts without
that variable intentionally report an explicit `unknown` provenance value.

## Reranker decision and diagnosis

The interactive default is **Hybrid RRF + reranker OFF**. On this exact
configuration, reranking reduced Recall@5 (`0.9233 → 0.9122`), MRR@10
(`0.8778 → 0.8333`) and nDCG@10 (`0.9518 → 0.9329`). There were `8` positive
and `12` negative real per-case flips. This is a measured configuration
decision, not a claim that rerankers are generally bad.

The latency probe measured bounded input `20 → 5`:

- cold request: rerank `7113.62 ms` (lazy model initialization included);
- warm rerank p50: `1436.13 ms`;
- warm rerank p95: `3084.34 ms`.

The adapter caches the model for the process lifetime. The latency artifact is
`eval/results/week2_stabilization_v1/reranker_latency.json`.

Real flip examples, including query, category, page/chunk, rank movement and
gold relevance, are in `reranker_flips.jsonl` and are summarized in the Demo
BENCHMARKS tab.

The preserved mentor-demo examples are `near_miss_01` (positive, gold
`embedding`, rank `5 → 1`) and `direct_08` (negative, gold `purpose` while
irrelevant candidates are promoted). They are real records from the same
snapshot, not UI fixtures.

## Answerability and security

- calibration split: validation only, 9 non-security score-bearing cases;
- selected threshold: `0.337857395`, frozen runtime value `0.338`;
- validation: false positives `0`, false negatives `0`;
- untouched test: 14 cases, false positives `0`, false negatives `2`;
- test reason distribution: `ANSWERED=10`, `LOW_RELEVANCE=1`,
  `SECURITY_POLICY=3`.

Existing unit/contract tests separately cover `NO_EVIDENCE` and
`INSUFFICIENT_COVERAGE`; they remain stable reason codes. A no-answer trace
shows the LLM adapter call count stays zero when the gate fails.

The refreshed test security artifact reports prompt-injection and leakage
cases `4/4` passed with `llm_called=false`.

Live demo checks also covered an answered query with canonical sources, a
`LOW_RELEVANCE` no-answer with `llm_ms=0`, and a direct prompt-injection query
returning `SECURITY_POLICY` without retrieval or LLM invocation.

Final rebuilt-image request IDs:

```text
answered/paraphrase: req_91b2e8881b204b70ac344cac6999ee6a
  model: gemma3:4b; canonical sections: corporate_problem p.4, deliverables p.5, purpose p.1–2
no-answer: req_49abc066fae44380b42b6b02b073915b
  reason: LOW_RELEVANCE; llm_ms: 0
prompt-injection: req_d431db18691343ebaf84f7f9c1df5971
  reason: SECURITY_POLICY; retrieval/LLM: skipped; llm_ms: 0
live trace: trace_9f5d6fb55bbe43c3ad5e8abcbe64eb7b
  request_id: req_3aec7de7223d430fb7d8a9cd9cb6192d; status: completed
```

Latest final-stack verification IDs:

```text
answered: req_3008c1bfe58041789d041b69a8f9506c
  Hybrid, reranker OFF, gemma3:4b; canonical evidence includes rag p.3
no-answer: req_149fd2ba33884bf7b5f874118ac3f278
  LOW_RELEVANCE; llm_ms=0
prompt-injection: req_3da2c5f777e24948a85cea13bd02a3d6
  SECURITY_POLICY; retrieval/LLM skipped; llm_ms=0
live trace: trace_e1f9cd387eb84a65a2c68db3d2abe8f
  request_id=req_ced59127e5b24db583f5f14cf47127a8; real stages completed
evidence-only ablation: req_e31f443e05f44adaa61860c43bdecb1e (OFF),
  req_5a6e3a7975724c738290ef4690e6ffbc (ON); bounded reranker 20 -> 5
```

The trace contained real `scope → representation → dense → sparse → RRF →
reranker skipped → evidence → answerability → prompt/LLM skipped → response`
events. Trace transport status is intentionally `completed`; stage outcomes
are carried by the event list.

The final rebuilt image also verified the same real query through the evidence-
only API:

```text
reranker OFF: req_8a28db04c3ff4bdbac3b7d924ade7212
  total 124.91 ms; RRF top-5 starts deliverables p.5, reranker skipped by configuration
reranker ON:  req_910c12b3832a40a68ae1950a578a035d
  total 5908.97 ms; bounded 20 → 5; embedding p.3 moved fusion rank 5 → rerank rank 1
```

This is an interactive ablation smoke, not a replacement for the frozen
44-case quality benchmark.

## Idempotency and persistence

The Compose smoke uploads the final PDF twice. The second request reuses the
same document/version identity with `idempotent_hit=true`; active Qdrant point
count remains unchanged. A Qdrant restart preserves the collection and count.

## System/model smoke

The sanitized profile reports Linux x86_64, Intel i7-1165G7, 4C/8T, 31.03 GB
RAM, no supported compute GPU, reachable Ollama and three installed models:
`gemma3:4b`, `qwen3:4b`, `qwen3:4b-instruct-local`. Compatibility labels are
heuristic (`Recommended`, `Likely usable`, `May run slowly`, `Memory risk`,
`Unknown`); they are not execution guarantees. Local model management remains
disabled by default.

## PDF family boundary

The 28-page Week-2 specification PDF was independently verified by
`eval/verify_week2_pdf.py`, but its section markers do not belong to the
Week-1 final corpus profile. The frozen benchmark still uses explicit
`mentor_program_v1` ingestion and is unchanged. Normal product uploads use
`auto`; if no complete known structure is detected they are indexed with
`generic_v1` instead of being rejected. The effective profile is included in
the version fingerprint and persisted with the job/point metadata. In Compose,
interactive product queries use the normal `auto` retrieval service while the
evaluation executor uses a separate explicit `mentor_program_v1` retrieval
service filtered by the frozen fingerprint.

## AUTO arbitrary-PDF regression (2026-08-10)

The real 8-page `Alperen_35K_Tip_Tercih_Raporu_2026.pdf` was uploaded to the
same Compose stack after the profile fix.

- final active version: `ver_85e1ab863ddd10bfa986f4f807e189169b01239c4d5c073eb638360d6e15f5d3`
- document: `doc_48cca08afefe26080c76717e7e2ab484018abb3552fda7e154e87ec5eedee32c`
- result: `SUCCEEDED`, 8 pages, 5 bounded parents, 82 children, 82 active points
- requested/resolved: `auto` → `generic_v1`
- fallback: no reliable known section markers; confidence `low`
- pipeline fingerprint: `1ed73a335fa738fccbca968bcb29a29c9b31066f2c2c04705625994db863058b`
- generic parent bound: `4000` characters; page windows `1–2`, `3`, `4–5`, `6–7`, `8`
- parent character sizes: `3866`, `2574`, `3983`, `3866`, `2488`
- duplicate upload: `idempotent_hit=true`, same active version and unchanged point identity
- page-3 retrieval: `Ben olsam hangi sırada yazardım?` returned child
  `...parent:001:child:001` at fusion rank `1`; its expanded parent is
  `...parent:001`, page `3`, `2574` characters / `389` words. The evidence-only
  trace selected `5` candidates. A separate answer-generation attempt was
  correctly gated as `LOW_RELEVANCE` and did not call the LLM.

The frozen mentor benchmark remained separate by explicit fingerprint: 26
active points and snapshot
`c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25`.
