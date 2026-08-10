"""Unit tests for bounded lazy cross-encoder reranking."""

from collections.abc import Sequence

import pytest

from projects.document_intelligence_service.app.domain.retrieval import RetrievedChunk
from projects.document_intelligence_service.app.infrastructure.reranking.cross_encoder import (
    CrossEncoderReranker,
)


def make_candidate(source_id: str) -> RetrievedChunk:
    """Create one deterministic evidence candidate."""

    return RetrievedChunk(
        source_id=source_id,
        document_id="doc-1",
        version_id="ver-1",
        parent_id="parent-1",
        title="RAG",
        text=f"evidence-{source_id}",
        page_start=1,
        page_end=1,
        score=0.1,
        rank=1,
    )


class FakeCrossEncoder:
    """Return scores for only the received bounded pair list."""

    def __init__(self) -> None:
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Sequence[float]:
        del batch_size, show_progress_bar
        self.seen_pairs = sentences
        return tuple(float(index) for index in range(len(sentences)))


def test_cross_encoder_is_bounded_and_returns_rerank_scores() -> None:
    reranker = CrossEncoderReranker(max_candidates=2, batch_size=4)
    model = FakeCrossEncoder()
    reranker._model = model

    result = reranker.rerank(
        question="Qdrant ne işe yarar?",
        candidates=(
            make_candidate("first"),
            make_candidate("second"),
            make_candidate("third"),
        ),
        limit=1,
    )

    assert len(model.seen_pairs) == 2
    assert result[0].source_id == "second"
    assert result[0].rerank_score == 1.0
    assert result[0].rank == 1


def test_cross_encoder_rejects_a_limit_above_bounded_window() -> None:
    reranker = CrossEncoderReranker(max_candidates=2)

    with pytest.raises(ValueError, match="between 1 and 2"):
        reranker.rerank(
            question="question",
            candidates=(make_candidate("one"),),
            limit=3,
        )
