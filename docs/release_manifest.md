# Week-2 standalone release manifest

This repository is released as the standalone Week-2 project. The first
complete source tree is anchored by base commit
`1522c51d87985716a337629074486be653c842c8`. The audit correction source
changes through the final CI correction are anchored by commit
`a998728ebc59696d401d04ee826b1830a0916a76` before this release metadata
update.

The validated product release identity is the Git tag `week2-final-v11`. The
V2 delivery release is anchored by a later clean source commit and tag
`week2-delivery-v2`. The package's `DELIVERY_SHA.txt` and
`results/final_delivery/` manifest are generated from that exact source SHA;
historical benchmark artifacts are preserved without rewriting their original
provenance. This release includes the Week-2 diagnostic and mentor-UI
corrections plus the V11 complementary-branch attribution, bounded
prompt-packing and delivery reproducibility corrections.
Historical evaluation artifacts retain the source revisions under
which their measured runs were produced.

Release invariants:

- frozen corpus snapshot: `c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25`;
- frozen point count: `26`;
- frozen collection: `document_chunks_week2_final_v1`;
- frozen mentor pipeline fingerprint:
  `132e52a3e8358e66906a7dd9bcfd0c8b57aa228dd3102e9b3d8f39ccfb4c41a4`;
- normal product upload profile: `auto` with `generic_v1` fallback;
- measured demo default: Hybrid RRF, reranker OFF, Gemma when runtime-ready.
- prompt packing: bounded 2,400-character production context; the V11 ablation
  retained the deadline date+time where the V10 1,200-character baseline did
  not, without passing expected answers or gold labels to the packer;
- blind human review: PENDING unless separately completed and recorded;
- Page 14 literal 5-gains/5-losses disagreement evidence: PARTIAL;
- CI/SBOM/vulnerability/Qdrant-CI gaps remain documented where not implemented.
