"""Lazy sentence-transformer cross-encoder reranker."""

from collections.abc import Sequence
from dataclasses import replace
import importlib
from typing import Any, Protocol, cast

from ...domain.retrieval import RetrievedChunk


class _CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        """Score question/evidence pairs."""


class CrossEncoderReranker:
    """Rerank only a bounded candidate window with a lazy cross-encoder."""

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        max_candidates: int = 20,
        batch_size: int = 8,
    ) -> None:
        if max_candidates <= 0 or batch_size <= 0:
            raise ValueError("reranker limits must be greater than zero")
        self._model_name = model_name
        self._max_candidates = min(max_candidates, 20)
        self._batch_size = batch_size
        self._model: _CrossEncoderModel | None = None

    def warmup(self) -> None:
        """Load the bounded reranker at an explicit lifecycle boundary."""

        self._load_model()

    def rerank(
        self,
        *,
        question: str,
        candidates: Sequence[RetrievedChunk],
        limit: int,
    ) -> tuple[RetrievedChunk, ...]:
        """Score at most 20 candidates and return the requested top-k."""

        if limit <= 0 or limit > self._max_candidates:
            raise ValueError(
                f"rerank limit must be between 1 and {self._max_candidates}"
            )
        bounded = tuple(candidates[: self._max_candidates])
        if not bounded:
            return ()

        pairs = [(question, candidate.text) for candidate in bounded]
        raw_scores = self._load_model().predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        scores = self._to_scores(raw_scores)
        if len(scores) != len(bounded):
            raise ValueError("reranker returned an unexpected batch size")

        ordered = sorted(
            zip(bounded, scores, strict=True),
            key=lambda item: (-item[1], item[0].source_id),
        )[:limit]
        return tuple(
            replace(
                candidate,
                score=score,
                rank=index,
                rerank_score=score,
            )
            for index, (candidate, score) in enumerate(ordered, start=1)
        )

    @staticmethod
    def _to_scores(raw_scores: Any) -> tuple[float, ...]:
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if isinstance(values, (int, float)):
            return (float(values),)
        return tuple(float(value) for value in cast(Sequence[Any], values))

    def _load_model(self) -> _CrossEncoderModel:
        if self._model is None:
            module = importlib.import_module("sentence_transformers")
            constructor = cast(Any, getattr(module, "CrossEncoder"))
            self._model = cast(_CrossEncoderModel, constructor(self._model_name))
        return self._model
