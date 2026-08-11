# Dense / BM25 / Hybrid slice comparison

This report is generated from the committed frozen raw artifacts under
`projects/document_intelligence_service/eval/results/week2_stabilization_v1/`.
It does not change labels, threshold selection or the frozen 26-point
membership. The benchmark contains 44 cases (development 19, validation 11,
test 14); retrieval metrics below use the answerable cases in each slice.

## Slice metrics

| Query slice | Cases | Dense R@5 | BM25 R@5 | Hybrid R@5 | Dense MRR@10 | BM25 MRR@10 | Hybrid MRR@10 | Dense nDCG@10 | BM25 nDCG@10 | Hybrid nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct fact | 8 | 0.8750 | 0.8750 | **1.0000** | 0.8125 | 0.7292 | **0.8438** | 0.8750 | 0.8750 | **1.0000** |
| paraphrase | 6 | 1.0000 | 1.0000 | 1.0000 | 0.9167 | 0.9167 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| exact term/code | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| near miss | 6 | 1.0000 | 0.6667 | 1.0000 | 0.7083 | 0.5333 | 0.6806 | 1.0000 | 0.6667 | 1.0000 |
| multi-evidence | 4 | 0.5083 | 0.3833 | 0.4250 | 1.0000 | 0.7500 | 0.8750 | 0.7220 | 0.5320 | 0.6382 |
| no-answer | 6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

`no-answer` is evaluated by the answerability/security gate, not retrieval
recall, so retrieval ranking metrics are intentionally not fabricated there.

## Concrete branch wins and losses

At the per-case Recall@5 level, there were five branch disagreements among
answerable cases:

| Case | Slice | Dense | BM25 | Hybrid | Interpretation |
|---|---|---:|---:|---:|---|
| `direct_07` | direct fact | 0 | 1 | 1 | BM25-only branch win: exact presentation-duration wording |
| `direct_03` | direct fact | 1 | 0 | 1 | Dense semantic branch win |
| `near_miss_01` | near miss | 1 | 0 | 1 | Dense branch retained the relevant section |
| `near_miss_02` | near miss | 1 | 0 | 1 | Dense branch retained the relevant section |
| `multi_02` | multi-evidence | 1 | 0 | 1 | Dense branch retained one required evidence target |

This is the honest answer to “where did BM25 beat Dense?” in this frozen
measurement: `direct_07` is the observed BM25-only top-5 win. Hybrid preserved
the relevant target in all five disagreement cases, but its multi-evidence
macro score is not perfect because target coverage is graded across multiple
evidence items. The small slice sizes limit confidence.

## Reproduction

The raw artifacts and historical run manifest are the source of truth:

```bash
cd projects/document_intelligence_service
python -m eval.run_ablation_report \
  --help
```

The historical measurements were produced at git SHA
`90900ae35b167100f9eef9d6b759e3b4b38a2c38`; this correction release does not
rewrite them.
