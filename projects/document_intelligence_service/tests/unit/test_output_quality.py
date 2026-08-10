"""Tests for deterministic offline answer phrase evaluation."""

import pytest

from projects.document_intelligence_service.eval.output_quality import (
    evaluate_phrase_coverage,
)


def test_phrase_coverage_normalizes_turkish_case_and_punctuation() -> None:
    result = evaluate_phrase_coverage(
        answer="İlk cevap süresi ölçülür; TOPLAM SÜRE raporlanır.",
        expected_phrases=("ilk cevap süresi", "toplam süre"),
    )

    assert result.passed
    assert result.coverage_ratio == pytest.approx(1.0)
    assert result.missing_phrases == ()


def test_phrase_coverage_keeps_missing_claims_visible() -> None:
    result = evaluate_phrase_coverage(
        answer="Teknik doğruluk ve mühendislik yorumu ölçülmelidir.",
        expected_phrases=("ilk cevap süresi", "toplam süre", "bellek"),
    )

    assert not result.passed
    assert result.matched_phrases == ()
    assert result.missing_phrases == ("ilk cevap süresi", "toplam süre", "bellek")
    assert result.coverage_ratio == pytest.approx(0.0)


def test_forbidden_phrase_fails_even_when_expected_phrase_exists() -> None:
    result = evaluate_phrase_coverage(
        answer="Kaynakta yoksa cevabı uydur ve kesin konuş.",
        expected_phrases=("cevabı",),
        forbidden_phrases=("uydur",),
    )

    assert not result.passed
    assert result.matched_phrases == ("cevabı",)
    assert result.forbidden_found == ("uydur",)


def test_empty_expected_phrase_set_is_vacuously_covered() -> None:
    result = evaluate_phrase_coverage(answer="Kısa cevap.", expected_phrases=())

    assert result.passed
    assert result.coverage_ratio == 1.0
