# Week-2 Retrieval and Answerability Benchmark

## Status

This is the final Week-2 stabilization report for the reproducible local demo
state. All retrieval variants below use the same active Qdrant snapshot, the
same 44-case golden dataset and the same section-aware ingestion configuration.
The benchmark does not call the LLM; generation latency is therefore not mixed
into retrieval quality measurements.

## Frozen evaluation identity

```text
dataset: mentor_program_pdf_rag_golden_v1
dataset_sha256: 5e822afa5d648656b18339b0d552c53a2c234c8e4e8213c5da782f51a53e369e
cases: 44 (development 19, validation 11, test 14)
quality denominator: 30 answerable cases
qdrant_collection: document_chunks_week2_final_v1
frozen_membership_point_count: 26
corpus_snapshot_id: c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25
pipeline_fingerprint: 132e52a3e8358e66906a7dd9bcfd0c8b57aa228dd3102e9b3d8f39ccfb4c41a4
source_revision: 90900ae35b167100f9eef9d6b759e3b4b38a2c38
membership_manifest: data/evaluations/week2_final_corpus_snapshot_v1.json
membership_boundary: immutable point-ID list (26 points)
```

Evaluation queries apply the manifest's exact point-ID membership. The pipeline
fingerprint remains reproducibility metadata and an additional compatibility
filter; it is not the corpus boundary. Therefore a later document produced by
the same pipeline cannot enter this frozen snapshot unless a new snapshot is
explicitly created. The active-point metadata verifier inspected the 26
manifest members and reported `missing_required_metadata=0`. The live
product/demo collection may contain additional active points. The old 135-point
`document_chunks_v2_bm25` collection is preserved separately for audit/rollback
and is not part of this benchmark claim.

Pipeline identity:

```text
parser: pypdf v1
normalizer: unicode_whitespace_v1
chunker: section_aware_v1, 3 sentences, overlap 1
section profile: mentor_program_v1
dense: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
sparse: bm25_qdrant_idf_v2
candidate/fusion/rerank: 30 / 20 / 5
RRF k: 60
```

The 28-page Week-2 specification PDF was verified independently, but its
section profile is not the Week-1 mentor corpus profile. It was rejected before
point creation and was not silently mixed into this snapshot.

## Retrieval ablation

Latency is retrieval-only and is reported before any LLM call. `Recall@1/3/5`
and ranking metrics use the 30 answerable cases; latency uses all 44 executed
cases. Candidate Recall@20 measures whether the correct section reached the
bounded candidate window.

| Variant | Candidate Recall@20 | Recall@1 | Recall@3 | Recall@5 | MRR@10 | nDCG@10 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.9933 | 0.7122 | 0.8567 | 0.9011 | 0.8750 | 0.9296 | 38.65 ms | 50.84 ms |
| BM25 | 0.9933 | 0.6622 | 0.7622 | 0.8178 | 0.7844 | 0.8376 | 7.57 ms | 10.76 ms |
| Hybrid RRF | 0.9933 | 0.7289 | 0.8567 | **0.9233** | **0.8778** | **0.9518** | 35.59 ms | 42.74 ms |
| Dense + reranker | 0.9933 | 0.6900 | 0.8567 | 0.9122 | 0.8361 | 0.9329 | 1285.38 ms | 1430.38 ms |
| Hybrid + reranker | 0.9933 | 0.6900 | 0.8233 | 0.9122 | 0.8333 | 0.9329 | 1325.10 ms | 1485.39 ms |

Raw CSV/JSONL observations, slices, confidence intervals and manifests are in
`eval/results/week2_stabilization_v1/`. The four live ablation runs used the
same snapshot and were recorded as:

```text
Hybrid OFF: eval_4da2f59af5934582b527d4168c593603
Dense OFF:  eval_76a6ef12de044fff802cc00ae323f070
BM25 OFF:   eval_0f50102ff6924691a95fc255c4824882
Hybrid ON:  eval_08bd5a6a2e0e438b9fdb8a37ca00de38
Dense ON:   eval_60cfeb57fa4941ff9475d8b0d649e2c2
```

## Reranker decision and latency diagnosis

The interactive default is **Hybrid RRF with reranker OFF**. On this exact
corpus/configuration, enabling the reranker lowered Recall@5 from `0.9233` to
`0.9122`, MRR@10 from `0.8778` to `0.8333`, and nDCG@10 from `0.9518` to
`0.9329`. There were 8 real positive flips and 12 real negative flips. This is
a measured configuration decision, not a general statement about rerankers.

The reranker received a bounded `20` candidates and returned `5`. The adapter
caches its model for the process lifetime; model initialization is not repeated
on each warm request. On this CPU-only host:

```text
cold rerank: 7113.62 ms (lazy model initialization included)
warm p50:    1436.13 ms
warm p95:    3084.34 ms
```

The first live ON request after an API restart was approximately `8620 ms`; a
subsequent warm request was approximately `1715 ms`. The main remaining cost is
the local CPU cross-encoder, not an unbounded candidate list or repeated model
download. Reranker remains available for a real ON/OFF diagnosis and evaluation
ablation.

Real flip records are in `eval/results/week2_stabilization_v1/reranker_flips.jsonl`.
Examples retained for the mentor demo:

- Positive: `near_miss_01`, gold `embedding`, rank `5 → 1`.
- Negative: `direct_08`, the gold `purpose` evidence remains but irrelevant
  candidates are promoted into the final window.

## Answerability calibration

Calibration was performed only on the validation split, after the final corpus
was frozen. Nine non-security, score-bearing validation cases were used. The
selected dense-score threshold is `0.337857395`, rounded to runtime `0.338`.
Validation result: false positives `0/7`, false negatives `0/2`. The test split
was not used to select the threshold.

The untouched 14-case test result is recorded separately: false positives `0`,
false negatives `2`. Its reason distribution is:

```text
ANSWERED: 10
LOW_RELEVANCE: 1
SECURITY_POLICY: 3
```

`NO_EVIDENCE`, `LOW_RELEVANCE` and `INSUFFICIENT_COVERAGE` remain stable domain
reason codes. Margin and phrase-coverage gates are still provisional diagnostics
(`0.0` runtime rejection thresholds), so this area remains `PARTIAL` in the
compliance matrix rather than being overstated as fully calibrated.

Evidence:
`eval/results/week2_stabilization_v1/hybrid_threshold_calibration.json` and
`hybrid_answerability_test_frozen.json`.

## Generic product-profile calibration

The arbitrary medical PDF is indexed as `generic_v1` and is deliberately kept
outside the frozen mentor calibration. The small real-document set is
`data/evaluations/generic_document_answerability_v1.jsonl`; its active corpus
scope is the medical PDF's 82 generic points and its pipeline fingerprint is
`1ed73a335fa738fccbca968bcb29a29c9b31066f2c2c04705625994db863058b`.

The generic policy was selected on six validation cases only:

```text
dense threshold:    0.24668321 -> runtime 0.247
coverage threshold: 0.36666667 -> runtime 0.367
validation FP/FN:   0/0
```

The untouched test now has six cases: one answerable and five
unanswerable/near-miss cases. It produced `TP=1`, `TN=5`, `FP=0`, `FN=0`.
The wrong-year case (2024 requested, 2025 available) is rejected as
`INSUFFICIENT_COVERAGE`; wrong-program, wrong-discount and year-attribute
mismatch cases are also retained as real negative regressions. The clearly
external astronomy query remains rejected as `LOW_RELEVANCE`.

This update changed neither the generic dense threshold (`0.247`) nor the
validation-only calibration policy. The generic sample remains small and is
not evidence of universal semantic coverage.

At runtime, policy selection uses resolved Qdrant `chunking_profile_resolved`
metadata. A missing or mixed profile uses the conservative mentor/default
policy. The actual profile, calibration id, score threshold and coverage
threshold are emitted in the live trace and structured query event.

## Security and no-answer evidence

- Direct prompt injection is rejected as `SECURITY_POLICY` before retrieval;
  no LLM call is made.
- Indirect injection-like evidence is filtered by `EvidenceSafetyPolicy` and
  is marked as untrusted in the structured prompt boundary.
- Frozen test security gate: `4/4` passed (`2/2` leakage/ACL and `2/2`
  prompt-injection cases), with `llm_called=false`.
- A live no-answer demo query returned `LOW_RELEVANCE` with `llm_ms=0`.
- Canonical sources are built from application evidence records, never parsed
  from generated text.

## Reproduction notes

Offline manifests record `source_revision`, dataset SHA, corpus snapshot,
collection, point count, pipeline fingerprint, model roles, retrieval limits,
host summary and metric implementation version. The production image excludes
`.git`; the Compose smoke injects `DIS_SOURCE_REVISION=$(git rev-parse HEAD)` so
live API evaluation records the same base revision. A direct `docker compose`
run without that variable reports `unknown` deliberately; the final tree must
be committed and rebuilt for a byte-for-byte final release identity.
