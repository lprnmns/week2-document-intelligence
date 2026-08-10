"""Contract checks for the versioned mentor golden dataset."""

from pathlib import Path

from projects.document_intelligence_service.eval.contracts import (
    load_jsonl,
    validate_case_set,
)


DATASET = Path("data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl")


def test_mentor_golden_dataset_has_expected_balance_and_splits() -> None:
    cases = validate_case_set(
        load_jsonl(DATASET),
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

    assert len(cases) == 44
    assert {case.split for case in cases} == {"development", "validation", "test"}
    assert sum(case.expected_answerable for case in cases) == 30
    assert sum(not case.expected_answerable for case in cases) == 14
