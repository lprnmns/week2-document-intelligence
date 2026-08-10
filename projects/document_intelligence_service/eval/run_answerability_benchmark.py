"""Run the live answerability gate without calling Ollama."""

from argparse import ArgumentParser
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import random
import subprocess

from ..app.domain.answerability import AnswerabilityPolicy
from ..app.domain.entities import RetrievalMode
from ..app.main import build_pipeline_config, build_retrieval_service
from ..app.domain.ingestion import compute_pipeline_fingerprint
from ..app.settings import Settings
from .contracts import load_jsonl, validate_case_set
from .runner import run_answerability_benchmark
from .reporting import build_run_manifest

DEFAULT_DATASET = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")
DEFAULT_WARMUPS = (
    "Qdrant ne işe yarar?",
    "Embedding ne demektir?",
    "RAG akışı hangi adımlardan oluşur?",
)


def main() -> None:
    """Run the gate on all cases and persist its raw decisions."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "development", "validation", "test"), default="all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--point-count", type=int, default=None)
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
    policy = AnswerabilityPolicy(
        min_dense_score=settings.answerability_min_dense_score,
        min_sparse_score=settings.answerability_min_sparse_score,
        min_rerank_score=settings.answerability_min_rerank_score,
        min_margin=settings.answerability_min_margin,
        min_coverage=settings.answerability_min_coverage,
    )
    run = run_answerability_benchmark(
        retrieval_service=service,
        answerability=policy,
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
            "answerability_policy": {
                "min_dense_score": policy.min_dense_score,
                "min_sparse_score": policy.min_sparse_score,
                "min_rerank_score": policy.min_rerank_score,
                "min_margin": policy.min_margin,
                "min_coverage": policy.min_coverage,
            },
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
        "llm_called": False,
        "answerability_policy": {
            "min_dense_score": policy.min_dense_score,
            "min_sparse_score": policy.min_sparse_score,
            "min_rerank_score": policy.min_rerank_score,
            "min_margin": policy.min_margin,
            "min_coverage": policy.min_coverage,
        },
        "qdrant_collection": settings.qdrant_collection,
        "embedding_model": pipeline_config.embedding_model,
        "sparse_encoder": pipeline_config.sparse_encoder,
        "section_marker_profile": settings.section_marker_profile,
        "bm25_state_path": settings.bm25_state_path,
        "run": asdict(run),
        "manifest": manifest,
        "query_order": [case.case_id for case in cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    by_id = {case.case_id: case for case in cases}
    for observation in run.observations:
        case = by_id[observation.case_id]
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "split": case.split,
                "language": case.language,
                "expected_answerable": case.expected_answerable,
                "decision": observation.decision,
                "reason": observation.reason,
                "status": observation.status,
                "error_code": observation.error_code,
                "error_message": observation.error_message,
                "total_ms": observation.total_ms,
                "top_score": observation.top_score,
                "score_margin": observation.score_margin,
                "coverage_ratio": observation.coverage_ratio,
            }
        )
    raw_jsonl = args.output.with_name(f"{args.output.stem}_raw.jsonl")
    raw_csv = args.output.with_name(f"{args.output.stem}_raw.csv")
    with raw_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics = run.metrics
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": run.cases_run,
                "false_positive_no_answer": metrics.no_answer_false_positive_count,
                "false_negative_no_answer": metrics.no_answer_false_negative_count,
                "false_positive_rate": metrics.no_answer_false_positive_rate,
                "false_negative_rate": metrics.no_answer_false_negative_rate,
                "total_p50_ms": run.total_latency.p50_ms,
                "total_p95_ms": run.total_latency.p95_ms,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
