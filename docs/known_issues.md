# Known Issues

## Diagnostic boundary: selected evidence vs answerability rejection

Observed in the curated `old_support_limit` case (`Arşivdeki 2025 eğitim desteği üst sınırı neydi?`):

- BM25 retrieves the trusted chunk containing `25.000 TL`.
- Evidence Selection keeps that chunk.
- Answerability then rejects the run because the calibrated qualifier-coverage check fails.
- Generation is correctly skipped.
- The no-answer response currently exposes an empty `sources` list, so the diagnostic projection can incorrectly report `EVIDENCE_SELECTION_LOSS` and show `Evidence 0/1`.

Correct interpretation of the current run:

```text
BM25                 PASS
Evidence Selection   PASS · trusted fact selected
Answerability        FAIL · qualifier coverage
Generation           SKIPPED
```

This is a diagnostic/presentation-boundary issue, not evidence retrieval failure and not a reason to retune the frozen answerability policy. A follow-up correction should preserve the selected evidence for diagnostic display while keeping the no-answer and LLM-skip behavior unchanged. In BM25-only views, a displayed `RRF #1` should also be presented as `BM25 #1`; RRF is not applicable in that mode.
