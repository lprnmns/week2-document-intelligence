"""Sentence-transformer dense embedding adapter."""

from collections.abc import Sequence
import importlib
from typing import Any, Protocol, cast


class _SentenceTransformer(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Encode a list of texts."""


class SentenceTransformerEmbedder:
    """Load the multilingual MiniLM model only on the first worker batch."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        expected_dimension: int = 384,
    ) -> None:
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero")
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._model: _SentenceTransformer | None = None

    @property
    def dimension(self) -> int:
        """Return the configured collection dimension."""

        return self._expected_dimension

    def warmup(self) -> None:
        """Load the model at an explicit lifecycle boundary."""

        self._load_model()

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode texts with normalized dense vectors."""

        if not texts:
            return ()
        model = self._load_model()
        encoded = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        rows = cast(list[list[float]], cast(Any, encoded).tolist())
        vectors = tuple(tuple(float(value) for value in row) for row in rows)
        if len(vectors) != len(texts):
            raise ValueError("dense embedder returned an unexpected batch size")
        if any(len(vector) != self._expected_dimension for vector in vectors):
            raise ValueError(
                "dense embedder returned a vector with an unexpected dimension"
            )
        return vectors

    def _load_model(self) -> _SentenceTransformer:
        if self._model is None:
            module = importlib.import_module("sentence_transformers")
            constructor = cast(Any, getattr(module, "SentenceTransformer"))
            self._model = cast(_SentenceTransformer, constructor(self._model_name))
        return self._model
