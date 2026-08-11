#!/usr/bin/env python3
"""Measure the bounded prompt-packing variants without leaking gold into packing.

The expected deadline fact is used only after packing as an evaluation
measurement. It is never passed to ``pack_prompt`` or the generation adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.document_intelligence_service.app.domain.retrieval import RetrievedChunk
from projects.document_intelligence_service.app.infrastructure.ollama.answer_generator import (
    OllamaAnswerGenerator,
)


def _chunks(snapshot: dict[str, object]) -> tuple[RetrievedChunk, ...]:
    result = snapshot.get("result")
    rows = result.get("sources", []) if isinstance(result, dict) else []
    chunks: list[RetrievedChunk] = []
    if not isinstance(rows, list):
        return ()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        chunks.append(
            RetrievedChunk(
                source_id=str(row.get("source_id", index)),
                document_id=str(row.get("document_id", "")),
                version_id="ablation",
                parent_id=str(row.get("parent_id", "")),
                title=str(row.get("title", "")),
                text=str(row.get("chunk_text", "")),
                page_start=int(row.get("page_start", 1)),
                page_end=int(row.get("page_end", 1)),
                score=0.0,
                rank=index,
                parent_text=str(row.get("parent_context", "")),
            )
        )
    return tuple(chunks)


def _measure(
    *,
    name: str,
    question: str,
    chunks: tuple[RetrievedChunk, ...],
    budget: int,
) -> dict[str, object]:
    generator = OllamaAnswerGenerator(
        base_url="http://127.0.0.1:11434",
        max_evidence_chars=budget,
    )
    started = perf_counter()
    packed = generator.pack_prompt(question=question, evidence=chunks)
    pack_ms = (perf_counter() - started) * 1000
    text = " ".join(fragment.included_text for fragment in packed.fragments)
    retained = all(value.casefold() in text.casefold() for value in ("10 Ağustos", "23:59"))
    return {
        "variant": name,
        "budget_chars": budget,
        "required_fact_retention": retained,
        "answer_completeness_proxy": "complete" if retained else "incomplete",
        "packed_chars": packed.total_evidence_chars,
        "packed_tokens_estimate": round(packed.total_evidence_chars / 4),
        "included_sources": len(packed.included_source_ids),
        "pack_latency_ms": round(pack_ms, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    question = "üniversite tercihleri için son tarih ne zaman"
    chunks = _chunks(snapshot)
    rows: list[dict[str, object]] = []
    previous = snapshot.get("result")
    previous_pack = previous.get("prompt_pack") if isinstance(previous, dict) else None
    previous_text = ""
    if isinstance(previous_pack, dict):
        previous_text = " ".join(
            str(fragment.get("included_text", ""))
            for fragment in previous_pack.get("fragments", [])
            if isinstance(fragment, dict)
        )
    rows.append(
        {
            "variant": "V10 current baseline",
            "budget_chars": 1200,
            "required_fact_retention": all(value.casefold() in previous_text.casefold() for value in ("10 Ağustos", "23:59")),
            "answer_completeness_proxy": "complete" if all(value.casefold() in previous_text.casefold() for value in ("10 Ağustos", "23:59")) else "incomplete",
            "packed_chars": previous_pack.get("total_evidence_chars") if isinstance(previous_pack, dict) else None,
            "packed_tokens_estimate": round((previous_pack.get("total_evidence_chars", 0) if isinstance(previous_pack, dict) else 0) / 4),
            "included_sources": previous_pack.get("included_count") if isinstance(previous_pack, dict) else None,
            "pack_latency_ms": None,
        }
    )
    for name, budget in (
        ("V11 intent-aware · 1200", 1200),
        ("V11 intent-aware · 2400", 2400),
        ("V11 bounded full-child-first · 3600", 3600),
    ):
        rows.append(_measure(name=name, question=question, chunks=chunks, budget=budget))
    payload = {
        "question": question,
        # Keep the committed metric artifact portable.  The real input snapshot
        # contains local document text and is intentionally not published.
        "input_snapshot": args.snapshot.name,
        "packing_does_not_receive_expected_answer": True,
        "rows": rows,
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
