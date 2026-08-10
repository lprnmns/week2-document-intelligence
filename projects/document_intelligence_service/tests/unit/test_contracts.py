"""Unit tests for answer/no-answer invariants."""

import pytest
from pydantic import ValidationError

from projects.document_intelligence_service.app.api.v1.contracts import (
    LatencyBreakdown,
    ModelInfo,
    QueryResponse,
    RetrievalInfo,
)
from projects.document_intelligence_service.app.domain.entities import (
    Decision,
    NoAnswerReason,
    RetrievalMode,
)


def _query_response(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "decision": Decision.NO_ANSWER,
        "answer": None,
        "no_answer_reason": NoAnswerReason.NO_EVIDENCE,
        "sources": [],
        "retrieval": {
            "mode": RetrievalMode.HYBRID,
            "dense_candidates": 0,
            "sparse_candidates": 0,
            "rrf_candidates": 0,
            "reranked_candidates": 0,
        },
        "model": {"provider": None, "model": None},
        "latency": {
            "embedding_ms": 0,
            "search_ms": 0,
            "rerank_ms": 0,
            "llm_ms": 0,
            "total_ms": 0,
        },
        "request_id": "contract-2",
    }
    payload.update(overrides)
    return payload


def test_no_answer_response_can_explicitly_skip_llm() -> None:
    response = QueryResponse.model_validate(_query_response())

    assert response.model.model is None
    assert response.latency.llm_ms == 0


def test_answered_response_requires_answer_and_no_reason() -> None:
    with pytest.raises(ValidationError):
        QueryResponse.model_validate(
            _query_response(
                decision=Decision.ANSWERED,
                answer=None,
                no_answer_reason=NoAnswerReason.NO_EVIDENCE,
            )
        )


def test_contract_helper_types_are_importable() -> None:
    assert ModelInfo(provider="ollama", model="gemma3:4b").model == "gemma3:4b"
    assert LatencyBreakdown(
        embedding_ms=1,
        search_ms=2,
        rerank_ms=3,
        llm_ms=4,
        total_ms=10,
    ).total_ms == 10
    assert RetrievalInfo(
        mode=RetrievalMode.DENSE,
        dense_candidates=1,
        sparse_candidates=0,
        rrf_candidates=0,
        reranked_candidates=1,
    ).mode is RetrievalMode.DENSE
