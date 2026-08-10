"""Run one real Qdrant retrieval benchmark strategy and write raw JSON."""

from argparse import ArgumentParser
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess

from ..app.domain.entities import RetrievalMode
from ..app.main import build_pipeline_config, build_retrieval_service
from ..app.domain.ingestion import compute_pipeline_fingerprint
from ..app.settings import Settings
from .contracts import load_jsonl, validate_case_set
from .runner import run_retrieval_benchmark
from .reporting import build_run_manifest, slice_report, write_raw_artifacts

DEFAULT_DATASET = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")
DEFAULT_WARMUPS = (
    "Qdrant ne işe yarar?",
    "Embedding ne demektir?",
    "RAG akışı hangi adımlardan oluşur?",
)


def main() -> None:
    """Parse options, run retrieval without an LLM and persist raw results."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), required=True)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "development", "validation", "test"), default="all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--point-count", type=int, default=None)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
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
    parser.add_argument(
        "--section-profile",
        choices=(
            "auto",
            "generic_v1",
            "none",
            "mentor_program_v1",
            "mentor_program_week2_v1",
        ),
        default=os.environ.get("DIS_SECTION_MARKER_PROFILE", "mentor_program_v1"),
    )
    args = parser.parse_args()

    all_cases = validate_case_set(
        load_jsonl(args.dataset),
        minimum_count=44,
        expected_category_counts={
            "direct_fact": 8,
            "paraphrase": 6,
            "exact_term": 6,
            "near_miss": 6,
            "no_answer": 6,
            "multi_evidence": 4,
            "prompt_injection": 4,
            "leakage_acl": 4,
        },
    )
    ordered_cases = [
        case for case in all_cases if args.split == "all" or case.split == args.split
    ]
    random.Random(args.seed).shuffle(ordered_cases)
    cases = tuple(ordered_cases)
    settings = Settings(
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection,
        bm25_state_path=args.bm25_state_path,
        section_marker_profile=args.section_profile,
        reranker_enabled=args.reranker,
    )
    pipeline_config = build_pipeline_config(settings)
    service = build_retrieval_service(settings)
    run = run_retrieval_benchmark(
        retrieval_service=service,
        cases=cases,
        mode=RetrievalMode(args.mode),
        top_k=args.top_k,
        warmup_questions=DEFAULT_WARMUPS,
        reranker_enabled=args.reranker,
    )
    manifest = build_run_manifest(
        dataset_path=args.dataset,
        cases=cases,
        qdrant_collection=settings.qdrant_collection,
        point_count=args.point_count,
        pipeline_config={
            **pipeline_config.canonical_dict(),
            "pipeline_fingerprint": compute_pipeline_fingerprint(pipeline_config),
            "reranker_model": pipeline_config.reranker_model if args.reranker else None,
            "candidate_k": settings.retrieval_candidate_k,
            "fusion_k": settings.retrieval_fusion_k,
            "rerank_k": settings.retrieval_rerank_k,
            "rrf_k": settings.rrf_k,
            "llm_model": settings.llm_model,
            "prompt_version": "structured_prompt_v1",
            "metric_implementation_version": "retrieval_metrics_v1",
        },
        mode=args.mode,
        reranker_enabled=args.reranker,
        top_k=args.top_k,
        warmup_questions=DEFAULT_WARMUPS,
        query_order_seed=args.seed,
    )
    report = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset": str(args.dataset),
        "mode": args.mode,
        "reranker_enabled": args.reranker,
        "top_k": args.top_k,
        "warmup_questions": list(DEFAULT_WARMUPS),
        "qdrant_collection": settings.qdrant_collection,
        "embedding_model": pipeline_config.embedding_model,
        "sparse_encoder": pipeline_config.sparse_encoder,
        "reranker_model": pipeline_config.reranker_model if args.reranker else None,
        "section_marker_profile": settings.section_marker_profile,
        "bm25_state_path": settings.bm25_state_path,
        "llm_called": False,
        "run": asdict(run),
        "manifest": manifest,
        "query_order": [case.case_id for case in cases],
        "slices": slice_report(cases=cases, report={"run": asdict(run)}, seed=args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_raw_artifacts(
        output_dir=args.raw_output_dir or args.output.parent,
        strategy=args.output.stem,
        cases=cases,
        report=report,
    )
    metrics = run.metrics
    print(
        json.dumps(
            {
                "output": str(args.output),
                "strategy": run.strategy,
                "cases": run.cases_run,
                "recall_at_5": metrics.recall_at_5,
                "candidate_recall_at_20": metrics.candidate_recall_at_20,
                "mrr_at_10": metrics.mrr_at_10,
                "ndcg_at_10": metrics.ndcg_at_10,
                "total_p50_ms": run.total_latency.p50_ms,
                "total_p95_ms": run.total_latency.p95_ms,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
