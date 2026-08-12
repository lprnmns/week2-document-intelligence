# Mentor demo runbook — 20 minutes

The demo is designed around one question: if a result is wrong, can we locate
the failing layer without reading raw JSON?

## 0:00–2:00 — Startup and health

Run `./scripts/start_demo.sh --bundled-ollama`, then open
`http://127.0.0.1:8501`.
The launcher does not report success until `/v1/health/ready` passes. Show the
header health strip and `/v1/health/ready`. Explain that green means
the corresponding health check responded successfully at the last poll; it is
not decorative UI state and it is not proof that every model is READY.

## 2:00–5:00 — Ingest an arbitrary PDF

Use any approved parseable PDF supplied locally. Show the eight ingestion groups:
Accept, Identity, Parse, Normalize, Chunk, Embed, Stage upsert/verify and
Activate. For a normal upload the requested profile is `AUTO`; a document with
no reliable specialized structure resolves to `generic_v1`. A mentor-specific
marker profile is an explicit benchmark/reproducibility option, not a product
admission requirement.

Point out the active-version rule: points are staged inactive, verified, then
the authoritative registry switches one version. Historical failed records are
kept but are excluded from search.

## 5:00–8:00 — Duplicate safety and document scope

Upload the same file again. Show `idempotent_hit=true`, the same content/pipeline
identity and unchanged active point identity. In ASK, show SEARCHABLE
DOCUMENTS separately from Unavailable documents. Only active versions can be
selected; unavailable records remain visible for diagnosis.

## 8:00–12:00 — Grounded query and Stage Explorer

Use the measured configuration: Hybrid RRF, reranker OFF and a genuinely READY
generation model. Run a direct or paraphrase question. Start at the top RUN
RESULT, then click the decision path and Stage Explorer. Explain:

`Dense + BM25 → RRF → Evidence → Answerability → Prompt Packing → LLM`.

Click Dense, BM25, RRF, Evidence Selection and Prompt Packing in the single
Stage Explorer. Each selection updates the same detail panel with the actual
file/page/child excerpt, ranks, selected evidence and real PromptPackResult
fragment metadata. The final source is application-generated from evidence
metadata, not copied from model text.

## 12:00–14:00 — No-answer and qualifier protection

Run the unrelated astronomy question. Show `NO ANSWER`, the reason code and
`LLM skipped`. If available, run the generic near-miss asking for a year/number
that is absent while an adjacent year/number exists; show
`INSUFFICIENT_COVERAGE` and the missing qualifier.

## 14:00–16:00 — Prompt injection

Run the existing direct or indirect injection regression. Show the security
decision before generation and the absence of an LLM call. Explain that ACL and
source checks are enforced in the application boundary as well as the vector
filter.

## 16:00–18:00 — Evaluation, not query ablation

Open BENCHMARKS. Unlike ASK, this tab allows multiple strategy and
reranker selections. Identify the immutable corpus snapshot, dataset split and
26-point membership. Show the measured result: Hybrid RRF outperformed Dense
and BM25 on the frozen set; reranker ON reduced this measured configuration's
quality and increased latency, so OFF is the demo default.

## 18:00–20:00 — Defense questions and limitations

Use `docs/mentor_technical_questions.md` for the 12 decision/evidence/
alternative/limitation answers. Be explicit about the boundaries: private
source PDF is not shipped, generic calibration is small, CPU generation can be
slow, model probe state is process-local, and poisoning/DoS/centralized log
controls are partial.

## Fast failure-attribution guide

For a golden case with expected evidence, compare Dense/BM25/RRF/reranker and
the selected evidence IDs. For an arbitrary query with no ground truth, say
“the trace shows the candidate journey and decision path; it does not prove
which layer is wrong.”

- absent from both branches: candidate retrieval is the first suspect;
- present in one branch but lost in RRF: fusion is the first suspect;
- present after RRF but removed by reranker: reranker is the first suspect;
- correct candidate but wrong final evidence: evidence selection/retrieval;
- correct evidence but gate rejects: answerability;
- gate passes but output is unsupported: prompt/generation/output validation.
