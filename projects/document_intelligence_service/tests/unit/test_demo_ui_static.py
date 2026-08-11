"""Static regression checks for the mentor-demo engineering console."""

from pathlib import Path


UI_PATH = Path(__file__).parents[4] / "demo_ui" / "index.html"


def test_demo_ui_keeps_diagnostic_hierarchy_and_safe_object_rendering() -> None:
    source = UI_PATH.read_text(encoding="utf-8")

    assert "RUN RESULT" in source
    assert "WAITING FOR RETRIEVAL" in source
    assert "CANDIDATES AVAILABLE" in source
    assert "FINAL EVIDENCE" in source
    assert "Generation failed · evidence remains inspectable" in source
    assert "function valueSummary(value)" in source
    assert "[object Object]" not in source
    assert "SEARCHABLE DOCUMENTS" in source
    assert "Unavailable documents (" in source
    assert "No active searchable version" in source
    assert "Re-upload / Retry in Documents" in source
    assert "ASK" in source
    assert "DOCUMENTS" in source
    assert "BENCHMARKS" in source
    assert "function activateTab(tabName)" in source
    assert "GOLD EVIDENCE JOURNEY" in source
    assert "Prepare demo documents" in source
    assert "UNATTRIBUTED — GOLD EVIDENCE REQUIRED" not in source
