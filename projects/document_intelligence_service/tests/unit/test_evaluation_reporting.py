"""Tests for reproducible benchmark manifests and qualitative reports."""

from pathlib import Path
from typing import cast

from projects.document_intelligence_service.eval.contracts import GoldenCase
from projects.document_intelligence_service.eval.reporting import (
    bootstrap_mean_ci,
    build_ablation_matrix,
    build_run_manifest,
    qualitative_error_analysis,
    reranker_flips,
    slice_report,
)


def _case(case_id: str, title: str, *, language: str = "tr") -> GoldenCase:
    return GoldenCase(
        case_id=case_id,
        question=f"{title} nedir?",
        category="direct_fact",
        split="test",
        expected_answerable=True,
        relevant_sections=(title,),
        language=language,
    )


def _observation(case_id: str, titles: list[str]) -> dict[str, object]:
    return {
        "case_id": case_id,
        "status": "ok",
        "total_ms": 10.0,
        "embedding_ms": 3.0,
        "search_ms": 7.0,
        "rerank_ms": 0.0,
        "final_candidates": [
            {
                "source_id": f"{title}-{index}",
                "document_id": "doc-1",
                "title": title,
                "rank": index,
            }
            for index, title in enumerate(titles, start=1)
        ],
        "candidate_window": [],
    }


def test_bootstrap_ci_is_deterministic_and_bounded() -> None:
    first = bootstrap_mean_ci((0.0, 0.5, 1.0), seed=7, resamples=200)
    second = bootstrap_mean_ci((0.0, 0.5, 1.0), seed=7, resamples=200)

    assert first == second
    lower = cast(float, first["lower"])
    estimate = cast(float, first["estimate"])
    upper = cast(float, first["upper"])
    assert lower <= estimate <= upper


def test_manifest_records_dataset_host_and_pipeline_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text('{"id":"x"}\n', encoding="utf-8")
    cases = (_case("x", "rag"),)

    manifest = build_run_manifest(
        dataset_path=dataset,
        cases=cases,
        qdrant_collection="document_chunks_v1",
        point_count=3,
        pipeline_config={"chunker": "section_aware_v1"},
        mode="hybrid",
        reranker_enabled=False,
        top_k=5,
        warmup_questions=("warmup",),
        query_order_seed=17,
    )

    assert manifest["qdrant_point_count"] == 3
    assert manifest["warmup_count"] == 1
    assert manifest["pipeline"] == {"chunker": "section_aware_v1"}
    run_id = manifest["run_id"]
    assert isinstance(run_id, str)
    assert run_id.startswith("eval_hybrid_baseline_")
    assert manifest["retrieval"] == {
        "mode": "hybrid",
        "candidate_k": 30,
        "top_k": 5,
        "fusion_k": 20,
        "rerank_k": 5,
        "fusion_config": {"algorithm": "rrf", "k": 60},
        "reranker_enabled": False,
    }
    assert manifest["warmup_runs"] == 1
    assert manifest["metric_implementation_version"] == "retrieval_metrics_v1"
    assert len(cast(str, manifest["dataset_sha256"])) == 64


def test_slices_and_flips_produce_reviewable_evidence() -> None:
    cases = (_case("one", "rag"), _case("two", "embedding", language="en"))
    baseline = {
        "mode": "hybrid",
        "run": {
            "metrics": {"recall_at_5": 0.5, "mrr_at_10": 0.75, "ndcg_at_10": 0.8},
            "total_latency": {"p50_ms": 10.0, "p95_ms": 20.0},
            "observations": [
                _observation("one", ["embedding", "rag"]),
                _observation("two", ["embedding", "rag"]),
            ],
        },
    }
    reranked = {
        "mode": "hybrid",
        "reranker_enabled": True,
        "run": {
            "metrics": {"recall_at_5": 1.0, "mrr_at_10": 1.0, "ndcg_at_10": 1.0},
            "total_latency": {"p50_ms": 20.0, "p95_ms": 30.0},
            "observations": [
                _observation("one", ["rag", "embedding"]),
                _observation("two", ["rag", "embedding"]),
            ],
        },
    }

    slices = slice_report(cases=cases, report=baseline)
    flips = reranker_flips(cases=cases, baseline=baseline, reranked=reranked)
    matrix = build_ablation_matrix(
        cases=cases,
        reports={"hybrid": baseline, "hybrid_reranker": reranked},
    )
    qualitative = qualitative_error_analysis(flips)

    assert slices["language:en"]["case_count"] == 1
    assert {item["flip"] for item in flips} == {"positive", "negative"}
    assert matrix[1]["variant"] == "D_hybrid_reranker"
    assert qualitative["positive_flip_count"] == 1
    assert qualitative["negative_flip_count"] == 1
