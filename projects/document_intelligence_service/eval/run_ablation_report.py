"""Assemble A/B/C/D retrieval evidence into raw and reviewable artifacts."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path

from ..app.domain.ingestion import PipelineConfig
from .contracts import load_jsonl, validate_case_set
from .reporting import (
    build_ablation_matrix,
    build_run_manifest,
    qualitative_error_analysis,
    reranker_flips,
    slice_report,
    write_raw_artifacts,
)

DEFAULT_DATASET = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")
DEFAULT_RESULTS = Path("projects/document_intelligence_service/eval/results")


def main() -> None:
    """Build a versioned report from four same-corpus raw benchmark JSON files."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS / "week2_report_v2")
    parser.add_argument("--dense", type=Path, default=DEFAULT_RESULTS / "dense_baseline.json")
    parser.add_argument("--hybrid", type=Path, default=DEFAULT_RESULTS / "hybrid_baseline.json")
    parser.add_argument("--dense-reranker", type=Path, default=DEFAULT_RESULTS / "dense_reranker.json")
    parser.add_argument("--hybrid-reranker", type=Path, default=DEFAULT_RESULTS / "hybrid_reranker.json")
    parser.add_argument("--point-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cases = validate_case_set(
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
    report_paths = {
        "dense": args.dense,
        "hybrid": args.hybrid,
        "dense_reranker": args.dense_reranker,
        "hybrid_reranker": args.hybrid_reranker,
    }
    reports = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in report_paths.items()
        if path.exists()
    }
    if not reports:
        raise SystemExit("No raw benchmark report was found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        key: write_raw_artifacts(
            output_dir=args.output_dir,
            strategy=key,
            cases=cases,
            report=report,
        )
        for key, report in reports.items()
    }
    matrix = build_ablation_matrix(cases=cases, reports=reports)
    matrix_path = args.output_dir / "ablation_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        fields = tuple(matrix[0].keys()) if matrix else ("variant",)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix)

    flips: list[dict[str, object]] = []
    if "dense" in reports and "dense_reranker" in reports:
        flips.extend(
            {"comparison": "dense_to_dense_reranker", **row}
            for row in reranker_flips(
                cases=cases,
                baseline=reports["dense"],
                reranked=reports["dense_reranker"],
            )
        )
    if "hybrid" in reports and "hybrid_reranker" in reports:
        flips.extend(
            {"comparison": "hybrid_to_hybrid_reranker", **row}
            for row in reranker_flips(
                cases=cases,
                baseline=reports["hybrid"],
                reranked=reports["hybrid_reranker"],
            )
        )
    flips_path = args.output_dir / "reranker_flips.jsonl"
    with flips_path.open("w", encoding="utf-8") as handle:
        for row in flips:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    first = next(iter(reports.values()))
    source_manifest = first.get("manifest", {})
    pipeline = (
        dict(source_manifest.get("pipeline", {}))
        if isinstance(source_manifest, dict)
        else {}
    )
    if not pipeline:
        pipeline = {
            "parser": PipelineConfig().parser,
            "parser_version": PipelineConfig().parser_version,
            "normalizer": PipelineConfig().normalizer,
            "chunker": PipelineConfig().chunker,
            "chunker_version": PipelineConfig().chunker_version,
            "embedding_model": first.get("embedding_model"),
            "sparse_encoder": first.get("sparse_encoder"),
            "reranker_model": first.get("reranker_model"),
            "vector_schema_version": PipelineConfig().vector_schema_version,
        }
    # The first report is normally the dense/OFF baseline.  Preserve the
    # actual reranker identity from an ON variant in the combined manifest so
    # the four-way ablation remains reproducible from one artifact.
    for report in reports.values():
        candidate_pipeline = report.get("manifest", {}).get("pipeline", {})
        if isinstance(candidate_pipeline, dict) and candidate_pipeline.get(
            "reranker_model"
        ):
            pipeline["reranker_model"] = candidate_pipeline["reranker_model"]
            break
    manifest = build_run_manifest(
        dataset_path=args.dataset,
        cases=cases,
        qdrant_collection=str(first.get("qdrant_collection", "unknown")),
        point_count=args.point_count,
        pipeline_config=pipeline,
        mode="ablation",
        reranker_enabled=True,
        top_k=int(first.get("top_k", 5)),
        warmup_questions=tuple(first.get("warmup_questions", [])),
        query_order_seed=args.seed,
    )
    manifest["source_report_git_shas"] = {
        key: report.get("git_sha") for key, report in reports.items()
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "manifest": manifest,
        "variants": matrix,
        "slices": {
            key: slice_report(cases=cases, report=report, seed=args.seed)
            for key, report in reports.items()
        },
        "reranker_flips": flips,
        "qualitative_error_analysis": qualitative_error_analysis(flips),
        "raw_artifacts": {
            key: {"jsonl": str(paths[0]), "csv": str(paths[1])}
            for key, paths in raw_paths.items()
        },
    }
    summary_path = args.output_dir / "ablation_summary_v2.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "qualitative_error_analysis.json").write_text(
        json.dumps(summary["qualitative_error_analysis"], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "matrix": str(matrix_path), "flips": str(flips_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
