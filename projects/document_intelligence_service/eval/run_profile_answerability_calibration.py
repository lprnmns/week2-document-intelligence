"""Calibrate and report answerability for one non-frozen document profile."""

from argparse import ArgumentParser
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

from qdrant_client import QdrantClient, models

from ..app.domain.answerability import AnswerabilityPolicy
from ..app.domain.entities import RetrievalMode
from ..app.main import build_pipeline_config, build_retrieval_service
from ..app.settings import Settings
from .calibration import calibrate_threshold
from .contracts import GoldenCase, load_jsonl, validate_case_set
from .runner import AnswerabilityBenchmarkRun, run_answerability_benchmark


DEFAULT_DATASET = Path("data/evaluations/generic_document_answerability_v1.jsonl")


def main() -> None:
    """Use validation only to freeze a generic profile policy, then test it."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6335")
    parser.add_argument("--collection", default="document_chunks_week2_final_v1")
    parser.add_argument("--bm25-state-path", default="/tmp/week2_final_bm25_state.json")
    parser.add_argument("--profile", default="generic_v1")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = validate_case_set(load_jsonl(args.dataset), minimum_count=1)
    validation = tuple(case for case in cases if case.split == "validation")
    test = tuple(case for case in cases if case.split == "test")
    if not validation or not test:
        raise ValueError("generic calibration requires validation and test cases")

    settings = Settings(
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection,
        bm25_state_path=args.bm25_state_path,
        section_marker_profile=args.profile,
        preload_models=False,
        reranker_enabled=False,
    )
    retrieval = build_retrieval_service(settings)
    mode = RetrievalMode(args.mode)

    # A permissive policy is used only to capture score signals. It does not
    # produce answers and cannot influence threshold selection.
    signal_policy = AnswerabilityPolicy(
        min_dense_score=0.0,
        min_sparse_score=0.0,
        min_rerank_score=-100.0,
        min_margin=0.0,
        min_coverage=0.0,
    )
    validation_signals = run_answerability_benchmark(
        retrieval_service=retrieval,
        answerability=signal_policy,
        cases=validation,
        mode=mode,
        top_k=args.top_k,
        reranker_enabled=False,
    )
    scores, coverages = _signal_maps(validation, validation_signals)
    score_calibration = calibrate_threshold(
        validation,
        scores,
        score_kind="dense",
    )
    coverage_calibration = calibrate_threshold(
        validation,
        coverages,
        score_kind="coverage",
    )

    frozen_policy = AnswerabilityPolicy(
        min_dense_score=score_calibration.rounded_threshold,
        min_sparse_score=settings.answerability_min_sparse_score,
        min_rerank_score=settings.answerability_min_rerank_score,
        min_margin=settings.answerability_min_margin,
        min_coverage=coverage_calibration.rounded_threshold,
        profile_name=args.profile,
        calibration_id=args.dataset.stem,
    )
    test_run = run_answerability_benchmark(
        retrieval_service=retrieval,
        answerability=frozen_policy,
        cases=test,
        mode=mode,
        top_k=args.top_k,
        reranker_enabled=False,
    )

    pipeline = build_pipeline_config(
        settings,
        section_marker_profile=args.profile,
    )
    pipeline_fingerprint = _pipeline_fingerprint(pipeline)
    report = {
        "git_sha": _git_sha(),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "profile": args.profile,
        "mode": args.mode,
        "reranker_enabled": False,
        "top_k": args.top_k,
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "policy_scope": {
            "chunking_profile": args.profile,
            "retrieval_mode": args.mode,
            "reranker_enabled": False,
            "dense_model": pipeline.embedding_model,
            "sparse_encoder": pipeline.sparse_encoder,
            "pipeline_fingerprint": pipeline_fingerprint,
        },
        "calibration": {
            "dense_score": asdict(score_calibration),
            "coverage": asdict(coverage_calibration),
        },
        "frozen_runtime_policy": {
            "min_dense_score": frozen_policy.min_dense_score,
            "min_sparse_score": frozen_policy.min_sparse_score,
            "min_rerank_score": frozen_policy.min_rerank_score,
            "min_margin": frozen_policy.min_margin,
            "min_coverage": frozen_policy.min_coverage,
        },
        "validation_signal_observations": _observations(validation_signals),
        "test_observations": _observations(test_run),
        "test_metrics": asdict(test_run.metrics),
        "corpus": _corpus_summary(
            url=args.qdrant_url,
            collection=args.collection,
            cases=cases,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validation_cases": len(validation),
                "test_cases": len(test),
                "dense_threshold": frozen_policy.min_dense_score,
                "coverage_threshold": frozen_policy.min_coverage,
                "test_metrics": asdict(test_run.metrics),
            },
            ensure_ascii=False,
        )
    )


def _signal_maps(
    cases: tuple[GoldenCase, ...],
    run: AnswerabilityBenchmarkRun,
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract numeric signals and reject incomplete calibration input."""

    by_id = {observation.case_id: observation for observation in run.observations}
    scores: dict[str, float] = {}
    coverages: dict[str, float] = {}
    for case in cases:
        observation = by_id[case.case_id]
        if observation.top_score is None:
            raise ValueError(f"missing top score for {case.case_id}")
        scores[case.case_id] = observation.top_score
        coverages[case.case_id] = observation.coverage_ratio
    return scores, coverages


def _observations(run: AnswerabilityBenchmarkRun) -> list[dict[str, object]]:
    """Keep the raw signal/result rows needed to audit the small calibration."""

    return [asdict(observation) for observation in run.observations]


def _pipeline_fingerprint(pipeline: object) -> str:
    """Compute the same fingerprint as ingestion without importing adapters."""

    from ..app.domain.ingestion import compute_pipeline_fingerprint

    return compute_pipeline_fingerprint(pipeline)  # type: ignore[arg-type]


def _corpus_summary(
    *,
    url: str,
    collection: str,
    cases: tuple[GoldenCase, ...],
) -> dict[str, object]:
    """Report the real product document scope used by the dataset."""

    document_ids = sorted(
        {
            document_id
            for case in cases
            for document_id in case.relevant_document_ids
        }
    )
    try:
        client = QdrantClient(url=url)
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True),
            ),
            models.FieldCondition(
                key="chunking_profile_resolved",
                match=models.MatchValue(value="generic_v1"),
            ),
        ]
        if document_ids:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=document_ids),
                )
            )
        count = client.count(
            collection_name=collection,
            count_filter=models.Filter(must=conditions),
            exact=True,
        ).count
    except Exception:
        count = None
    return {
        "collection": collection,
        "document_ids": document_ids,
        "generic_active_point_count": count,
    }


def _git_sha() -> str:
    """Return the source revision used for the report."""

    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


if __name__ == "__main__":
    main()
