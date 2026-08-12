"""Static regression checks for the mentor-demo engineering console."""

from pathlib import Path


UI_PATH = Path(__file__).parents[4] / "demo_ui" / "index.html"
SMOKE_PATH = Path(__file__).parents[4] / "scripts" / "compose_smoke.sh"


def test_demo_ui_keeps_diagnostic_hierarchy_and_safe_object_rendering() -> None:
    source = UI_PATH.read_text(encoding="utf-8")

    assert "RUN RESULT" in source
    assert "WAITING FOR RETRIEVAL" in source
    assert "CANDIDATES AVAILABLE" in source
    assert "FINAL EVIDENCE" in source
    assert "Generation failed · evidence remains inspectable" in source
    assert "function valueSummary(value)" in source
    assert "[object Object]" not in source
    assert "[truncated]" not in source
    assert "bounded details omitted" not in source
    assert "Semantic expected/actual similarity: not computed" not in source
    assert "SEARCHABLE DOCUMENTS" in source
    assert "Unavailable documents (" in source
    assert "No active searchable version" in source
    assert "Re-upload / Retry in Documents" in source
    assert "ASK" in source
    assert "DOCUMENTS" in source
    assert "BENCHMARKS" in source
    assert "function activateTab(tabName)" in source
    assert "PIPELINE EXPLORER" in source
    assert "stage-detail-panel" in source
    assert "Rank movement" in source
    assert "function movement(value)" in source
    assert "function movementClass(value)" in source
    assert "Actual PromptPackResult fragments" in source
    assert "Trusted expected evidence" in source
    assert "evidence-inspector" not in source
    assert "CANDIDATE JOURNEY" not in source
    assert "GOLD EVIDENCE JOURNEY" not in source
    assert "Prepare demo documents" in source
    assert "UNATTRIBUTED — GOLD EVIDENCE REQUIRED" not in source
    assert "Select trusted evidence (optional)" in source
    assert "/v1/demo/gold/evidence" in source
    assert "/trusted-evidence" in source
    assert "PromptPackResult" in source
    assert "fact_survival" in source
    assert "presentation?.claims" in source
    assert "reranker_movement" in source
    assert "bounded to budget" in source
    assert "LLM CALL SUCCEEDED" in source
    assert "Trusted chunk survival" in source
    assert "Technical attribution" in source
    assert "Show full rank table" in source
    assert "no second retrieval or LLM run" in source
    assert 'id="trusted-text"' in source
    assert 'addEventListener("input", scheduleTrustedEvidenceBrowse)' in source
    assert "trustedBrowseRequest" in source
    assert "Refresh results" in source


def test_compose_smoke_uses_content_scoped_idempotency_identity() -> None:
    source = SMOKE_PATH.read_text(encoding="utf-8")
    assert "sha256sum" in source
    assert "compose-smoke-upload-v2" not in source
    assert 'smoke_idempotency_key}"' in source
