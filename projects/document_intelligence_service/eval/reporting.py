"""Reproducible reporting helpers for retrieval and reranker experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Any, Callable, Iterable, Mapping, cast

from ..app.domain.evaluation import (
    EvaluationCorpusSnapshot,
    compute_corpus_snapshot_id,
    load_corpus_snapshot,
)
from .contracts import EvidenceLike, GoldenCase
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


@dataclass(slots=True)
class RawEvidence:
    """Small adapter that lets stored JSON observations use metric functions."""

    source_id: str
    document_id: str
    title: str
    rank: int = 0


def dataset_sha256(path: Path) -> str:
    """Return the content hash of the exact evaluation dataset artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha(root: Path = Path(".")) -> str:
    """Return the source revision recorded beside raw benchmark output."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = 17,
    resamples: int = 1000,
) -> dict[str, float | int | None]:
    """Return deterministic percentile bootstrap CI for a macro metric."""

    materialized = tuple(float(value) for value in values)
    if not materialized:
        return {"estimate": None, "lower": None, "upper": None, "samples": 0}
    rng = random.Random(seed)
    means = [
        sum(rng.choice(materialized) for _ in materialized) / len(materialized)
        for _ in range(resamples)
    ]
    ordered = sorted(means)
    return {
        "estimate": sum(materialized) / len(materialized),
        "lower": _percentile(ordered, 0.025),
        "upper": _percentile(ordered, 0.975),
        "samples": len(materialized),
    }


def host_manifest() -> dict[str, object]:
    """Capture reproducibility-relevant host facts without extra dependencies."""

    memory_bytes: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "accelerator": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    }


def build_run_manifest(
    *,
    dataset_path: Path,
    cases: tuple[GoldenCase, ...],
    qdrant_collection: str,
    point_count: int | None,
    pipeline_config: Mapping[str, object],
    mode: str,
    reranker_enabled: bool,
    top_k: int,
    warmup_questions: tuple[str, ...],
    query_order_seed: int,
    root: Path = Path("."),
) -> dict[str, object]:
    """Build the manifest required to interpret one benchmark snapshot."""

    revision = git_sha(root)
    pipeline = dict(pipeline_config)
    host = host_manifest()
    memory_bytes = host.get("memory_bytes")
    ram_gb = (
        round(int(memory_bytes) / (1024**3), 2)
        if isinstance(memory_bytes, int)
        else None
    )
    candidate_k = _as_int(
        pipeline.get("candidate_k", pipeline.get("retrieval_candidate_k")),
        default=30,
    )
    fusion_k = _as_int(
        pipeline.get("fusion_k", pipeline.get("retrieval_fusion_k")),
        default=20,
    )
    rerank_k = _as_int(
        pipeline.get("rerank_k", pipeline.get("retrieval_rerank_k")),
        default=5,
    )
    rrf_k = _as_int(pipeline.get("rrf_k"), default=60)
    chunk_config = {
        key: pipeline.get(key)
        for key in (
            "chunker",
            "chunker_version",
            "chunk_size_sentences",
            "chunk_overlap_sentences",
            "section_marker_profile",
        )
    }
    chunk_config_hash = hashlib.sha256(
        json.dumps(chunk_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    pipeline_fingerprint = pipeline.get("pipeline_fingerprint")
    snapshot = _load_matching_corpus_snapshot(
        root=root,
        collection=qdrant_collection,
        pipeline_fingerprint=(
            pipeline_fingerprint if isinstance(pipeline_fingerprint, str) else None
        ),
    )
    corpus_snapshot_id = (
        snapshot.snapshot_id
        if snapshot is not None
        else compute_corpus_snapshot_id(
            dataset_sha256=dataset_sha256(dataset_path),
            qdrant_collection=qdrant_collection,
            point_count=point_count,
            pipeline_fingerprint=(
                pipeline_fingerprint
                if isinstance(pipeline_fingerprint, str)
                else None
            ),
        )
    )
    observed_point_count = (
        point_count if point_count is not None else snapshot.point_count if snapshot else None
    )
    return {
        "run_id": (
            f"eval_{mode}_{'reranker' if reranker_enabled else 'baseline'}_"
            f"{revision[:12]}_seed{query_order_seed}"
        ),
        "git_sha": revision,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "dataset_version": dataset_path.stem,
        "corpus_snapshot_id": corpus_snapshot_id,
        "corpus_snapshot_basis": (
            "immutable_point_id_manifest"
            if snapshot is not None
            else "dataset_sha256+qdrant_collection+active_point_count+pipeline_fingerprint"
        ),
        "qdrant_collection": qdrant_collection,
        "qdrant_point_count": observed_point_count,
        "corpus_snapshot_point_count": snapshot.point_count if snapshot else None,
        "corpus_snapshot_verified": (
            point_count == snapshot.point_count
            if snapshot is not None and point_count is not None
            else None
        ),
        "corpus_snapshot_manifest": (
            "data/evaluations/week2_final_corpus_snapshot_v1.json"
            if snapshot is not None
            else None
        ),
        "corpus_membership": (
            [
                {"document_id": document_id, "version_id": version_id}
                for document_id, version_id in snapshot.document_versions
            ]
            if snapshot is not None
            else []
        ),
        "chunk_config_hash": chunk_config_hash,
        "prompt_version": pipeline.get("prompt_version", "structured_prompt_v1"),
        "case_count": len(cases),
        "split_counts": _counts(cases, lambda case: case.split),
        "language_counts": _counts(cases, lambda case: case.language),
        "pipeline": pipeline,
        "pipeline_fingerprint": pipeline_fingerprint,
        "models": {
            "dense": pipeline.get("embedding_model"),
            "sparse": pipeline.get("sparse_encoder"),
            "reranker": pipeline.get("reranker_model"),
            "llm": pipeline.get("llm_model"),
        },
        "retrieval": {
            "mode": mode,
            "candidate_k": candidate_k,
            "top_k": top_k,
            "fusion_k": fusion_k,
            "rerank_k": rerank_k,
            "fusion_config": {"algorithm": "rrf", "k": rrf_k},
            "reranker_enabled": reranker_enabled,
        },
        "warmup_count": len(warmup_questions),
        "warmup_runs": len(warmup_questions),
        "warmup_questions": list(warmup_questions),
        "query_order_seed": query_order_seed,
        "host": {**host, "ram_gb": ram_gb},
        "metric_implementation_version": pipeline.get(
            "metric_implementation_version", "retrieval_metrics_v1"
        ),
        "llm_called": False,
    }


def _load_matching_corpus_snapshot(
    *,
    root: Path,
    collection: str,
    pipeline_fingerprint: str | None,
) -> EvaluationCorpusSnapshot | None:
    """Return the committed snapshot only when its config matches the run."""

    path = root / "data/evaluations/week2_final_corpus_snapshot_v1.json"
    if not path.is_file():
        return None
    try:
        snapshot = load_corpus_snapshot(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if snapshot.collection != collection:
        return None
    if pipeline_fingerprint != snapshot.pipeline_fingerprint:
        return None
    return snapshot


def _as_int(value: object, *, default: int) -> int:
    """Read an optional integer manifest field without trusting its type."""

    return value if isinstance(value, int) and not isinstance(value, bool) else default


def write_raw_artifacts(
    *,
    output_dir: Path,
    strategy: str,
    cases: tuple[GoldenCase, ...],
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write compact JSONL and CSV per-query artifacts for error analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    observations = report.get("run", {}).get("observations", [])
    by_id = {case.case_id: case for case in cases}
    rows: list[dict[str, object]] = []
    for observation in observations:
        case = by_id[observation["case_id"]]
        final_candidates = observation.get("final_candidates", [])
        candidate_window = observation.get("candidate_window", [])
        rows.append(
            {
                "strategy": strategy,
                "case_id": case.case_id,
                "category": case.category,
                "split": case.split,
                "language": case.language,
                "expected_answerable": case.expected_answerable,
                "status": observation.get("status", "ok"),
                "error_code": observation.get("error_code"),
                "error_message": observation.get("error_message"),
                "final_source_ids": [item.get("source_id") for item in final_candidates],
                "final_titles": [item.get("title") for item in final_candidates],
                "candidate_source_ids": [item.get("source_id") for item in candidate_window],
                "latency_ms": observation.get("total_ms", 0.0),
                "embedding_ms": observation.get("embedding_ms", 0.0),
                "search_ms": observation.get("search_ms", 0.0),
                "rerank_ms": observation.get("rerank_ms", 0.0),
            }
        )
    jsonl_path = output_dir / f"{strategy}_raw.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = output_dir / f"{strategy}_raw.csv"
    fields = tuple(rows[0].keys()) if rows else (
        "strategy",
        "case_id",
        "category",
        "split",
        "language",
        "expected_answerable",
        "status",
        "error_code",
        "error_message",
        "final_source_ids",
        "final_titles",
        "candidate_source_ids",
        "latency_ms",
        "embedding_ms",
        "search_ms",
        "rerank_ms",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                    for key, value in row.items()
                }
            )
    return jsonl_path, csv_path


def slice_report(
    *,
    cases: tuple[GoldenCase, ...],
    report: Mapping[str, Any],
    seed: int = 17,
) -> dict[str, dict[str, object]]:
    """Compute category, split and language slices with bootstrap intervals."""

    observations = {
        item["case_id"]: item
        for item in report.get("run", {}).get("observations", [])
    }
    groups: dict[str, tuple[GoldenCase, ...]] = {
        "all": cases,
    }
    for field in ("category", "split", "language"):
        values = sorted({_group_value(case, field) for case in cases})
        groups.update(
            {
                f"{field}:{value}": tuple(
                    case for case in cases if _group_value(case, field) == value
                )
                for value in values
            }
        )

    output: dict[str, dict[str, object]] = {}
    for name, group in groups.items():
        answerable = tuple(case for case in group if case.expected_answerable)
        per_case: list[tuple[float, float, float]] = []
        for case in answerable:
            observation = observations.get(case.case_id, {})
            final = _evidence_tuple(observation.get("final_candidates", []))
            per_case.append(
                (
                    recall_at_k(case, final, 5),
                    mrr_at_k(case, final, 10),
                    ndcg_at_k(case, final, 10),
                )
            )
        statuses = [observations.get(case.case_id, {}).get("status", "missing") for case in group]
        output[name] = {
            "case_count": len(group),
            "answerable_count": len(answerable),
            "failure_count": sum(status != "ok" for status in statuses),
            "failure_rate": (
                sum(status != "ok" for status in statuses) / len(group)
                if group
                else 0.0
            ),
            "recall_at_5": bootstrap_mean_ci(
                (item[0] for item in per_case), seed=seed
            ),
            "mrr_at_10": bootstrap_mean_ci(
                (item[1] for item in per_case), seed=seed + 1
            ),
            "ndcg_at_10": bootstrap_mean_ci(
                (item[2] for item in per_case), seed=seed + 2
            ),
        }
    return output


def build_ablation_matrix(
    *,
    cases: tuple[GoldenCase, ...],
    reports: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    """Return A/B/C/D comparison rows from four same-snapshot reports."""

    labels = {
        "A_dense": "dense",
        "B_hybrid": "hybrid",
        "C_dense_reranker": "dense_reranker",
        "D_hybrid_reranker": "hybrid_reranker",
    }
    rows: list[dict[str, object]] = []
    for label, key in labels.items():
        report = reports.get(key)
        if report is None:
            continue
        run = report.get("run", {})
        metrics = run.get("metrics", {})
        observations = run.get("observations", [])
        failures = sum(item.get("status", "ok") != "ok" for item in observations)
        rows.append(
            {
                "variant": label,
                "mode": report.get("mode"),
                "reranker_enabled": report.get("reranker_enabled", False),
                "case_count": len(cases),
                "failure_count": failures,
                "failure_rate": failures / len(cases) if cases else 0.0,
                "candidate_recall_at_20": metrics.get("candidate_recall_at_20"),
                "recall_at_5": metrics.get("recall_at_5"),
                "mrr_at_10": metrics.get("mrr_at_10"),
                "ndcg_at_10": metrics.get("ndcg_at_10"),
                "p50_ms": run.get("total_latency", {}).get("p50_ms"),
                "p95_ms": run.get("total_latency", {}).get("p95_ms"),
            }
        )
    return rows


def reranker_flips(
    *,
    cases: tuple[GoldenCase, ...],
    baseline: Mapping[str, Any],
    reranked: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Compare per-case top-k evidence and label positive/negative flips."""

    base_by_id = {item["case_id"]: item for item in baseline.get("run", {}).get("observations", [])}
    rerank_by_id = {item["case_id"]: item for item in reranked.get("run", {}).get("observations", [])}
    rows: list[dict[str, object]] = []
    for case in cases:
        baseline_observation = base_by_id.get(case.case_id, {})
        reranked_observation = rerank_by_id.get(case.case_id, {})
        before = _evidence_tuple(
            baseline_observation.get("final_candidates", [])
        )
        after = _evidence_tuple(
            reranked_observation.get("final_candidates", [])
        )
        before_recall = recall_at_k(case, before, 5)
        after_recall = recall_at_k(case, after, 5)
        before_mrr = mrr_at_k(case, before, 10)
        after_mrr = mrr_at_k(case, after, 10)
        delta = after_mrr - before_mrr
        if delta == 0:
            delta = after_recall - before_recall
        if delta == 0:
            continue
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "split": case.split,
                "language": case.language,
                "question": case.question,
                "targets": list(case.target_keys()),
                "baseline_recall_at_5": before_recall,
                "reranker_recall_at_5": after_recall,
                "baseline_mrr_at_10": before_mrr,
                "reranker_mrr_at_10": after_mrr,
                "quality_delta": delta,
                "delta": delta,
                "flip": "positive" if delta > 0 else "negative",
                "baseline_titles": [item.title for item in before],
                "reranker_titles": [item.title for item in after],
                "rank_movements": _rank_movements(
                    case=case,
                    baseline_observation=baseline_observation,
                    reranked_observation=reranked_observation,
                ),
                "rationale": (
                    "Positive flip: the reranker improved the best observed "
                    "gold relevance rank or coverage."
                    if delta > 0
                    else "Negative flip: the reranker moved the best observed "
                    "gold evidence down or out of the final top-k."
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -abs(cast(float, row["delta"])),
            str(row["case_id"]),
        ),
    )


def _rank_movements(
    *,
    case: GoldenCase,
    baseline_observation: Mapping[str, object],
    reranked_observation: Mapping[str, object],
) -> list[dict[str, object]]:
    """Describe real candidate movement with stable source metadata."""

    before_items = _candidate_index(baseline_observation)
    after_items = _candidate_index(reranked_observation)
    source_ids = sorted(set(before_items) | set(after_items))
    movements: list[dict[str, object]] = []
    for source_id in source_ids:
        before = before_items.get(source_id)
        after = after_items.get(source_id)
        representative = after or before or {}
        evidence = RawEvidence(
            source_id=source_id,
            document_id=str(representative.get("document_id", "")),
            title=str(representative.get("title", "")),
        )
        gold_key = case.match_key(evidence)
        before_rank = _rank_for(before, prefer_fusion=True)
        after_rank = _rank_for(after, prefer_fusion=False)
        if before_rank is None and after_rank is None:
            continue
        if before_rank is None:
            movement = "promoted_in"
        elif after_rank is None:
            movement = "demoted_out"
        elif after_rank < before_rank:
            movement = "promoted"
        elif after_rank > before_rank:
            movement = "demoted"
        else:
            movement = "unchanged"
        movements.append(
            {
                "source_id": source_id,
                "chunk_id": representative.get("chunk_id") or source_id,
                "parent_id": representative.get("parent_id"),
                "document_id": representative.get("document_id"),
                "title": representative.get("title"),
                "page_start": representative.get("page_start"),
                "page_end": representative.get("page_end"),
                "before_rank": before_rank,
                "after_rank": after_rank,
                "movement": movement,
                "gold_key": gold_key,
                "gold_relevance": case.grade_for(gold_key) if gold_key else 0,
            }
        )
    return sorted(
        movements,
        key=lambda item: (
            item["after_rank"] is None,
            item["after_rank"] if item["after_rank"] is not None else 999,
            item["before_rank"] if item["before_rank"] is not None else 999,
            str(item["source_id"]),
        ),
    )


def _candidate_index(observation: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Index a bounded union of pre-rerank and final candidates."""

    indexed: dict[str, Mapping[str, object]] = {}
    for field in ("candidate_window", "final_candidates"):
        raw = observation.get(field, [])
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            source_id = item.get("source_id")
            if isinstance(source_id, str) and source_id:
                indexed[source_id] = item
    return indexed


def _rank_for(item: Mapping[str, object] | None, *, prefer_fusion: bool) -> int | None:
    """Read the rank appropriate to the baseline or reranked view."""

    if item is None:
        return None
    keys = ("fusion_rank", "rank") if prefer_fusion else ("rank", "rerank_rank")
    for key in keys:
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def qualitative_error_analysis(flips: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Select up to five evidence-backed gains and losses without inventing cases."""

    materialized = list(flips)
    gains = [item for item in materialized if item.get("flip") == "positive"][:5]
    losses = [item for item in materialized if item.get("flip") == "negative"][:5]
    return {
        "positive_flip_count": sum(item.get("flip") == "positive" for item in materialized),
        "negative_flip_count": sum(item.get("flip") == "negative" for item in materialized),
        "five_positive_examples_available": len(gains) >= 5,
        "five_negative_examples_available": len(losses) >= 5,
        "positive_examples": gains,
        "negative_examples": losses,
        "interpretation_rule": "Only observed per-case Recall@5 changes are included; missing examples are reported, not fabricated.",
    }


def _evidence_tuple(raw: object) -> tuple[EvidenceLike, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        RawEvidence(
            source_id=str(item.get("source_id", "")),
            document_id=str(item.get("document_id", "")),
            title=str(item.get("title", "")),
            rank=int(item.get("rank", index)),
        )
        for index, item in enumerate(raw, start=1)
        if isinstance(item, dict)
    )


def _group_value(case: GoldenCase, field: str) -> str:
    """Return one of the explicitly supported slice dimensions."""

    if field == "category":
        return case.category
    if field == "split":
        return case.split
    if field == "language":
        return case.language
    raise ValueError(f"unsupported slice field: {field}")


def _counts(
    cases: tuple[GoldenCase, ...],
    getter: Callable[[GoldenCase], str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = getter(case)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight
