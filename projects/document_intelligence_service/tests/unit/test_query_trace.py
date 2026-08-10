"""Tests for privacy-conscious structured query traces."""

import hashlib

from projects.document_intelligence_service.app.domain.entities import (
    Decision,
    NoAnswerReason,
)
from projects.document_intelligence_service.app.domain.retrieval import RetrievalResult
from projects.document_intelligence_service.app.observability.query_trace import (
    QueryTraceEvent,
)


def test_trace_hashes_question_and_keeps_raw_text_out_of_payload() -> None:
    question = "Stajyer maaşı ne kadar?"
    event = QueryTraceEvent.from_query_result(
        question=question,
        decision=Decision.NO_ANSWER,
        no_answer_reason=NoAnswerReason.LOW_RELEVANCE,
        retrieval=RetrievalResult(
            mode="hybrid",
            candidates=(),
            dense_candidates=5,
            sparse_candidates=5,
            rrf_candidates=5,
            embedding_ms=12.0,
            search_ms=3.0,
        ),
        selected_evidence_count=0,
        top_score=0.2,
        score_margin=0.01,
        coverage_ratio=0.0,
        provider=None,
        model=None,
        warnings=(),
        llm_ms=0.0,
        total_ms=15.0,
    )

    payload = event.as_dict()

    assert payload["question_sha256"] == hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()
    assert question not in str(payload)
    assert payload["decision"] == "no_answer"
    assert payload["no_answer_reason"] == "LOW_RELEVANCE"
    assert payload["latency_ms"] == {
        "embedding": 12.0,
        "search": 3.0,
        "rerank": 0.0,
        "llm": 0.0,
        "total": 15.0,
    }
