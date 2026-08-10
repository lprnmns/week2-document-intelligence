"""Measure cold and warm bounded reranker latency on the live local corpus."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient, models

from ..app.domain.entities import RetrievalMode
from ..app.domain.evaluation import compute_corpus_snapshot_id, load_corpus_snapshot
from ..app.domain.ingestion import compute_pipeline_fingerprint
from ..app.main import build_pipeline_config, build_retrieval_service
from ..app.settings import Settings


DEFAULT_QUESTION = "RAG akışında dokümandan cevaba kadar hangi adımlar bulunur?"


def main() -> None:
    """Run one cold request followed by bounded warm requests."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6335")
    parser.add_argument("--collection", default="document_chunks_week2_final_v1")
    parser.add_argument("--bm25-state-path", default="/tmp/week2_final_bm25_state.json")
    parser.add_argument("--section-profile", default="mentor_program_v1")
    parser.add_argument("--point-count", type=int, default=None)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warm_runs <= 0:
        raise SystemExit("--warm-runs must be positive")

    settings = Settings(
        qdrant_url=args.qdrant_url,
        qdrant_collection=args.collection,
        bm25_state_path=args.bm25_state_path,
        section_marker_profile=args.section_profile,
        reranker_enabled=True,
    )
    pipeline = build_pipeline_config(settings)
    service = build_retrieval_service(settings)
    point_count = args.point_count
    if point_count is None:
        point_count = QdrantClient(url=str(settings.qdrant_url)).count(
            collection_name=settings.qdrant_collection,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="active",
                        match=models.MatchValue(value=True),
                    )
                ]
            ),
            exact=True,
        ).count
    dataset = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    pipeline_fingerprint = compute_pipeline_fingerprint(pipeline)
    snapshot = None
    snapshot_path = Path("data/evaluations/week2_final_corpus_snapshot_v1.json")
    if snapshot_path.is_file():
        try:
            candidate = load_corpus_snapshot(snapshot_path)
            if (
                candidate.collection == settings.qdrant_collection
                and candidate.pipeline_fingerprint == pipeline_fingerprint
            ):
                snapshot = candidate
        except (OSError, ValueError, json.JSONDecodeError):
            snapshot = None

    def measure() -> dict[str, object]:
        started = perf_counter()
        result = service.search(
            question=args.question,
            mode=RetrievalMode.HYBRID,
            top_k=5,
            reranker_enabled=True,
        )
        wall_ms = (perf_counter() - started) * 1000
        return {
            "wall_ms": wall_ms,
            "embedding_ms": result.embedding_ms,
            "search_ms": result.search_ms,
            "rerank_ms": result.rerank_ms,
            "candidate_input": len(result.candidate_window),
            "final_output": result.reranked_candidates,
            "rerank_limit": result.rerank_limit,
        }

    cold = measure()
    warm = [measure() for _ in range(args.warm_runs)]
    rerank_warm = [_required_float(item["rerank_ms"]) for item in warm]
    wall_warm = [_required_float(item["wall_ms"]) for item in warm]
    report = {
        "measurement_version": "reranker_latency_v1",
        "question": args.question,
        "corpus_snapshot_id": (
            snapshot.snapshot_id
            if snapshot is not None
            else compute_corpus_snapshot_id(
                dataset_sha256=dataset_sha,
                qdrant_collection=settings.qdrant_collection,
                point_count=point_count,
                pipeline_fingerprint=pipeline_fingerprint,
            )
        ),
        "corpus_snapshot_basis": (
            "immutable_point_id_manifest"
            if snapshot is not None
            else "dataset_sha256+qdrant_collection+active_point_count+pipeline_fingerprint"
        ),
        "corpus_snapshot_point_count": snapshot.point_count if snapshot else None,
        "corpus_membership": (
            [
                {"document_id": document_id, "version_id": version_id}
                for document_id, version_id in snapshot.document_versions
            ]
            if snapshot is not None
            else []
        ),
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_point_count": point_count,
        "pipeline_fingerprint": pipeline_fingerprint,
        "reranker_model": settings.reranker_model,
        "candidate_bound": settings.retrieval_fusion_k,
        "final_output_bound": settings.retrieval_rerank_k,
        "initialization": "cold request includes lazy model initialization; warm requests reuse the same adapter/model",
        "cold": cold,
        "warm": {
            "runs": warm,
            "rerank_p50_ms": _percentile(rerank_warm, 0.50),
            "rerank_p95_ms": _percentile(rerank_warm, 0.95),
            "wall_p50_ms": _percentile(wall_warm, 0.50),
            "wall_p95_ms": _percentile(wall_warm, 0.95),
        },
        "conclusion": "Reranker is cached and bounded; measured CPU latency and quality trade-off keep it OFF by default.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def _percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic linear-interpolated percentile."""

    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _required_float(value: object) -> float:
    """Narrow a measured numeric field before percentile calculation."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError("latency measurement must be numeric")


if __name__ == "__main__":
    main()
