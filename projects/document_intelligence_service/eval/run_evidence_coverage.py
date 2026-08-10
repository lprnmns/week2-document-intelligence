"""Measure labeled phrase coverage in final retrieved evidence."""

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import load_jsonl, validate_case_set
from .evidence_coverage import build_evidence_coverage_report, load_benchmark_report

DEFAULT_DATASET = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")
DEFAULT_BENCHMARK = Path(
    "projects/document_intelligence_service/eval/results/hybrid_baseline.json"
)


def main() -> None:
    """Write an evidence coverage artifact without calling an LLM."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = validate_case_set(load_jsonl(args.dataset), minimum_count=44)
    benchmark = load_benchmark_report(args.benchmark)
    report = build_evidence_coverage_report(cases=cases, benchmark_report=benchmark)
    output = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "benchmark": str(args.benchmark),
        "benchmark_git_sha": benchmark.get("git_sha"),
        "retrieval_mode": benchmark.get("mode"),
        "llm_called": False,
        "coverage": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["coverage"], ensure_ascii=False))


if __name__ == "__main__":
    main()
