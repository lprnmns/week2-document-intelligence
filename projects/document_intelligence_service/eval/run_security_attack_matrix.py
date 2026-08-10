"""Validate and render the repository security attack matrix."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date
import json
from pathlib import Path
import subprocess
from typing import Any

ALLOWED_STATUSES = frozenset({"pass", "partial", "not_ready"})
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATRIX = REPO_ROOT / "data/evaluations/security_attack_matrix_v1.json"


def load_matrix(path: Path) -> dict[str, Any]:
    """Load and validate the declarative matrix fixture."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("security attack matrix must be a JSON object")
    controls = raw.get("controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("security attack matrix needs a non-empty controls list")
    seen_ids: set[str] = set()
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError("each security control must be an object")
        control_id = _required_string(control, "id")
        if control_id in seen_ids:
            raise ValueError(f"duplicate security control id: {control_id}")
        seen_ids.add(control_id)
        status = _required_string(control, "status")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported security status: {status}")
        evidence = control.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{control_id} needs evidence")
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError(f"{control_id} evidence must be an object")
            _required_string(item, "path")
            _required_string(item, "claim")
    return raw


def build_report(matrix: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Add reproducibility metadata and check every evidence path."""

    controls = matrix["controls"]
    checked_controls: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    for raw_control in controls:
        control = dict(raw_control)
        checked_evidence: list[dict[str, Any]] = []
        for raw_item in raw_control["evidence"]:
            item = dict(raw_item)
            relative_path = item["path"]
            absolute_path = repo_root / relative_path
            exists = absolute_path.is_file()
            item["exists"] = exists
            if not exists:
                missing_paths.append(relative_path)
            checked_evidence.append(item)
        control["evidence"] = checked_evidence
        control["evidence_complete"] = all(item["exists"] for item in checked_evidence)
        checked_controls.append(control)

    status_counts = {
        status: sum(control["status"] == status for control in checked_controls)
        for status in sorted(ALLOWED_STATUSES)
    }
    report = dict(matrix)
    report.update(
        {
            "generated_on": date.today().isoformat(),
            "git_sha": _git_sha(repo_root),
            "controls": checked_controls,
            "summary": {
                "control_count": len(checked_controls),
                "status_counts": status_counts,
                "evidence_path_count": sum(
                    len(control["evidence"]) for control in checked_controls
                ),
                "missing_evidence_path_count": len(missing_paths),
                "missing_evidence_paths": sorted(set(missing_paths)),
                "release_ready": not missing_paths
                and status_counts["not_ready"] == 0
                and status_counts["partial"] == 0,
            },
        }
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render a concise human-readable report from the validated matrix."""

    summary = report["summary"]
    lines = [
        "# Security Attack Matrix v1",
        "",
        f"Generated: `{report['generated_on']}`",
        f"Git SHA: `{report['git_sha']}`",
        f"Scope: {report['scope']}",
        "",
        "Bu rapor tehdit sınıflarını mevcut kod ve test kanıtlarıyla eşleştirir. "
        "`pass` yalnız belirtilen dar kapsamın kanıtlandığını, genel güvenlik "
        "garantisi olmadığını ifade eder.",
        "",
        "## Özet",
        "",
        f"- Kontrol sayısı: `{summary['control_count']}`",
        f"- `pass`: `{summary['status_counts']['pass']}`",
        f"- `partial`: `{summary['status_counts']['partial']}`",
        f"- `not_ready`: `{summary['status_counts']['not_ready']}`",
        f"- Eksik kanıt yolu: `{summary['missing_evidence_path_count']}`",
        f"- Release-ready: `{summary['release_ready']}`",
        "",
        "## Matris",
        "",
        "| ID | Tehdit | Durum | Mevcut kontrol | Sonuç | Sonraki adım |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for control in report["controls"]:
        lines.append(
            "| {id} | {threat} | `{status}` | {control} | {result} | {next_action} |".format(
                id=control["id"],
                threat=control["threat"],
                status=control["status"],
                control=control["control"].replace("|", "\\|"),
                result=control["observed_result"].replace("|", "\\|"),
                next_action=control["next_action"].replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Okuma notu",
            "",
            "Özellikle `SEC-04` için `pass`, local MVP kapsamındaki tenant/ACL "
            "pre-filter ve source re-check izolasyonunun test edildiği anlamına "
            "gelir; authentication veya merkezi authorization sistemi anlamına "
            "gelmez. Filtre değerleri şu an istemci tarafından beyan edilir ve "
            "bir sonraki kapsamda authenticated request principal ile bağlanmalıdır.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Validate the matrix and write JSON/Markdown artifacts."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    report = build_report(matrix)
    if report["summary"]["missing_evidence_path_count"]:
        paths = ", ".join(report["summary"]["missing_evidence_paths"])
        raise SystemExit(f"security matrix has missing evidence paths: {paths}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def _required_string(raw: dict[str, Any], key: str) -> str:
    """Read one required non-empty string field."""

    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _git_sha(repo_root: Path) -> str:
    """Return the current commit for artifact reproducibility."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    main()
