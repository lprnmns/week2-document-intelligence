"""Run one bounded real local query and persist its output-validation trace."""

from argparse import ArgumentParser
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from ..app.domain.entities import RetrievalMode
from ..app.main import build_query_service
from ..app.settings import Settings


DEFAULT_QUESTION = "Yerel model karşılaştırmasında hangi değerler ölçülmelidir?"


def main() -> None:
    """Run a single low-memory Gemma smoke without starting the HTTP server."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=2, choices=range(1, 6))
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("DIS_QDRANT_URL", "http://127.0.0.1:6335"),
    )
    parser.add_argument(
        "--collection",
        default=os.environ.get(
            "DIS_QDRANT_COLLECTION", "document_chunks_week2_final_v1"
        ),
    )
    parser.add_argument(
        "--bm25-state-path",
        default=os.environ.get("DIS_BM25_STATE_PATH", "data/bm25_state.json"),
    )
    args = parser.parse_args()

    settings = Settings(
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection,
        bm25_state_path=args.bm25_state_path,
        section_marker_profile="mentor_program_v1",
        reranker_enabled=False,
        llm_model=args.model,
        llm_max_output_tokens=args.max_output_tokens,
        llm_timeout_seconds=args.timeout_seconds,
    )
    result = asyncio.run(
        build_query_service(settings).execute(
            question=args.question,
            mode=RetrievalMode.HYBRID,
            top_k=args.top_k,
        )
    )
    report = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "question": args.question,
        "model_requested": args.model,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
        "decision": result.decision.value,
        "answer": result.answer,
        "no_answer_reason": (
            result.no_answer_reason.value
            if result.no_answer_reason is not None
            else None
        ),
        "warnings": [
            {
                "code": warning.code.value,
                "message": warning.message,
                "values": list(warning.values),
            }
            for warning in result.warnings
        ],
        "sources": [
            {
                "source_id": candidate.source_id,
                "page_start": candidate.page_start,
                "page_end": candidate.page_end,
                "title": candidate.title,
                "score": candidate.score,
            }
            for candidate in result.sources
        ],
        "retrieval": {
            "mode": result.retrieval.mode,
            "dense_candidates": result.retrieval.dense_candidates,
            "sparse_candidates": result.retrieval.sparse_candidates,
            "rrf_candidates": result.retrieval.rrf_candidates,
            "reranked_candidates": result.retrieval.reranked_candidates,
        },
        "answerability": asdict(result.answerability),
        "model": {
            "provider": result.provider,
            "model": result.model,
        },
        "latency_ms": {
            "embedding": result.retrieval.embedding_ms,
            "search": result.retrieval.search_ms,
            "rerank": result.retrieval.rerank_ms,
            "llm": result.llm_ms,
            "total": result.total_ms,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
