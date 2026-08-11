# V11 prompt-packing ablation

The comparison uses the last real five-source deadline-style run as the input
evidence. Expected-answer values are used only after packing to measure required
fact retention; they are never passed into production prompt construction.

| Variant | Budget | Required date+time retained | Packed chars | Token estimate | Sources included |
|---|---:|---:|---:|---:|---:|
| V10 current baseline | 1,200 | no | 1,200 | 300 | 5 |
| V11 intent-aware | 1,200 | no | 1,200 | 300 | 5 |
| V11 intent-aware | 2,400 | yes | 2,400 | 600 | 5 |
| V11 bounded full-child-first | 3,600 | yes | 3,600 | 900 | 5 |

The selected production configuration is 2,400 characters: it is the smallest
measured bounded configuration that retained both the deadline date and
`23:59` in the actual included fragment text. The 3,600-character variant did
not add required-fact retention for this case and was therefore not selected.

Generation latency is measured separately against the same local Ollama model;
it is not inferred from prompt size. The local CPU/model limitation is recorded
in the final release manifest and runbook.
