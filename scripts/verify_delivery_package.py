#!/usr/bin/env python3
"""Validate the mentor-facing delivery package without starting services."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

CANONICAL_FINGERPRINT = (
    "132e52a3e8358e66906a7dd9bcfd0c8b57aa228dd3102e9b3d8f39ccfb4c41a4"
)
SNAPSHOT_ID = "c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25"
DATASET_SHA = "5e822afa5d648656b18339b0d552c53a2c234c8e4e8213c5da782f51a53e369e"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_RE = re.compile(
    r"(?:ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,}|"
    r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|Authorization:\s*Bearer\s+\S+)"
)
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL · {message}")


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_root(value: Path) -> Path:
    root = value.resolve()
    if not root.is_dir():
        fail(f"package root is not a directory: {root}")
    return root


def validate_tree(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            fail(f"forbidden runtime/development path: {relative}")
        if path.is_file() and path.suffix in {".pyc", ".sqlite3", ".db"}:
            fail(f"generated/runtime file present: {relative}")
        if path.is_file() and path.name == ".env":
            fail(f"secret environment file present: {relative}")
        if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
            fail(f"unexpectedly large delivery file: {relative}")

    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".env.example":
            continue
        text = read(path)
        if SECRET_RE.search(text):
            fail(f"obvious secret pattern present: {path.relative_to(root)}")


def validate_identity(root: Path) -> str:
    delivery_sha_path = root / "DELIVERY_SHA.txt"
    if not delivery_sha_path.is_file():
        fail("DELIVERY_SHA.txt is missing")
    delivery_text = read(delivery_sha_path)
    match = re.search(
        r"^Delivery commit:\s*([0-9a-f]{40})\s*$",
        delivery_text,
        re.MULTILINE,
    )
    if not match:
        fail("DELIVERY_SHA.txt has no valid Delivery commit")
    source_sha = match.group(1)

    snapshot_path = root / "evaluation" / "week2_final_corpus_snapshot_v1.json"
    if not snapshot_path.is_file():
        fail("frozen snapshot manifest is missing")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("point_count") != 26 or len(snapshot.get("point_ids", [])) != 26:
        fail("frozen snapshot does not contain exactly 26 point IDs")
    if snapshot.get("corpus_snapshot_id") != SNAPSHOT_ID:
        fail("snapshot ID mismatch")
    if snapshot.get("pipeline_fingerprint") != CANONICAL_FINGERPRINT:
        fail("snapshot pipeline fingerprint mismatch")

    split_path = root / "evaluation" / "mentor_program_pdf_rag_split_v1.json"
    dataset_path = root / "evaluation" / "mentor_program_pdf_rag_golden_v1.jsonl"
    if not split_path.is_file() or not dataset_path.is_file():
        fail("dataset or split manifest is missing")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("total_cases") != 44:
        fail("golden dataset does not contain 44 cases")
    if split.get("split_counts") != {"development": 19, "validation": 11, "test": 14}:
        fail("golden split is not 19/11/14")
    if sha256(dataset_path) != DATASET_SHA or split.get("dataset_sha256") != DATASET_SHA:
        fail("golden dataset SHA mismatch")

    identity_files = [
        root / "README.md",
        delivery_sha_path,
        root / "verification" / "REPRODUCIBILITY_CHECK.md",
        root / "benchmark" / "BENCHMARK_REPORT.md",
    ]
    for path in identity_files:
        text = read(path)
        if CANONICAL_FINGERPRINT not in text or SNAPSHOT_ID not in text:
            fail(f"identity mismatch in {path.relative_to(root)}")
        fingerprints = set(re.findall(r"132e52a3[0-9a-f]+", text))
        invalid = fingerprints - {CANONICAL_FINGERPRINT}
        if invalid:
            fail(
                f"invalid fingerprint(s) in {path.relative_to(root)}: "
                f"{sorted(invalid)}"
            )

    for path in root.rglob("*.md"):
        text = read(path)
        if "ai-journey-ollama" in text:
            fail(f"machine-specific Ollama name remains in {path.relative_to(root)}")
    return source_sha


def validate_results(root: Path, source_sha: str) -> None:
    flips = root / "results" / "week2_stabilization_v1" / "reranker_flips.jsonl"
    if not flips.is_file():
        fail("reranker flip artifact is missing")
    counts = {"positive": 0, "negative": 0}
    for line in flips.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        flip = item.get("flip")
        if flip in counts:
            counts[flip] += 1
    if counts != {"positive": 8, "negative": 12}:
        fail(f"reranker flip counts mismatch: {counts}")

    final_dir = root / "results" / "final_delivery"
    manifests = sorted(final_dir.glob("*.json")) if final_dir.is_dir() else []
    if not manifests:
        fail("results/final_delivery has no machine-readable artifact")
    latest = manifests[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    if payload.get("git_sha") != source_sha:
        fail("latest final-delivery artifact is not tied to Delivery commit")
    for key, expected in {
        "pipeline_fingerprint": CANONICAL_FINGERPRINT,
        "corpus_snapshot_id": SNAPSHOT_ID,
        "dataset_sha256": DATASET_SHA,
    }.items():
        if payload.get(key) != expected:
            fail(f"final-delivery {key} mismatch")


def validate_links(root: Path) -> None:
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in root.rglob("*.md"):
        for target in pattern.findall(read(path)):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target_path = (path.parent / target.split("#", 1)[0]).resolve()
            if not target_path.is_file():
                fail(f"broken Markdown link in {path.relative_to(root)}: {target}")


def validate_zip(path: Path) -> None:
    if not path.is_file():
        fail(f"ZIP is missing: {path}")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name and not name.endswith("/")]
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            fail(f"ZIP must have exactly one root directory: {sorted(roots)}")
        if any("/.git/" in f"/{name}" or name.endswith("/.git") for name in names):
            fail("ZIP contains .git metadata")
        bad = [name for name in names if any(part in FORBIDDEN_PARTS for part in Path(name).parts)]
        if bad:
            fail(f"ZIP contains forbidden runtime paths: {bad[:3]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    root = package_root(args.package_root)
    validate_tree(root)
    source_sha = validate_identity(root)
    validate_results(root, source_sha)
    validate_links(root)
    if args.zip:
        validate_zip(args.zip.resolve())
    print(f"PASS · delivery package validated · source_sha={source_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
