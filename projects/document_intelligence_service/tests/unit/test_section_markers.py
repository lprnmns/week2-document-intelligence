"""Tests for explicit document-family section-marker profiles."""

import pytest

from projects.document_intelligence_service.app.infrastructure.parsing.section_markers import (
    MENTOR_PROGRAM_V1_MARKERS,
    MENTOR_PROGRAM_WEEK2_V1_MARKERS,
    get_section_markers,
)


def test_none_profile_is_safe_for_unknown_pdf_families() -> None:
    assert get_section_markers("none") == ()


def test_mentor_profile_has_ordered_known_markers() -> None:
    markers = get_section_markers("mentor_program_v1")

    assert markers == MENTOR_PROGRAM_V1_MARKERS
    assert len(markers) == 7
    assert markers[0].title == "purpose"
    assert markers[-1].title == "deliverables"


def test_week2_profile_covers_all_28_pdf_headings() -> None:
    markers = get_section_markers("mentor_program_week2_v1")

    assert markers == MENTOR_PROGRAM_WEEK2_V1_MARKERS
    assert len(markers) == 28
    assert markers[0].title == "cover"
    assert markers[-1].title == "appendix"


def test_unknown_profile_fails_loudly() -> None:
    with pytest.raises(ValueError, match="Unknown section marker profile"):
        get_section_markers("made_up_profile")
