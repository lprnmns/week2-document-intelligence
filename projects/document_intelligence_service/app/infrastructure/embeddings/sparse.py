"""Deterministic lexical sparse encoder for the first hybrid spike."""

from collections import Counter
from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import re

from ...domain.vectors import SparseVector

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class HashingSparseEncoder:
    """Map normalized term frequencies to stable sparse feature IDs.

    Qdrant's IDF modifier supplies collection-level inverse document frequency
    during search. This adapter deliberately owns only deterministic tokenization
    and term-frequency encoding; a later spike will compare it with a fitted
    corpus-aware BM25 adapter.
    """

    def __init__(self, *, feature_count: int = 1_048_576) -> None:
        if feature_count <= 0:
            raise ValueError("feature_count must be greater than zero")
        self._feature_count = feature_count

    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        """Encode each text into sorted, unique sparse index/value pairs."""

        return tuple(self._encode_one(text) for text in texts)

    def _encode_one(self, text: str) -> SparseVector:
        counts = Counter(token.casefold() for token in _TOKEN_PATTERN.findall(text))
        pairs = sorted(
            (
                self._feature_index(token),
                1.0 + math.log(float(term_count)),
            )
            for token, term_count in counts.items()
        )
        # Hash collisions can produce the same feature index; merge them before
        # constructing the value object, which requires unique sorted indices.
        merged: dict[int, float] = {}
        for index, value in pairs:
            merged[index] = merged.get(index, 0.0) + value
        return SparseVector(
            indices=tuple(sorted(merged)),
            values=tuple(merged[index] for index in sorted(merged)),
        )

    def _feature_index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big") % self._feature_count


class BM25SparseEncoder:
    """Online BM25 document weights with an exact fitted vocabulary.

    Qdrant's IDF modifier supplies corpus-level IDF at query time. This
    adapter supplies the BM25 term-frequency saturation while its persisted
    vocabulary keeps feature IDs stable across worker restarts. ``b=0`` is a
    deliberate online-index choice: changing the corpus average document
    length would otherwise make already indexed vectors stale.
    """

    def __init__(
        self,
        *,
        state_path: str | Path | None = None,
        k1: float = 1.2,
        b: float = 0.0,
    ) -> None:
        if k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")
        self._state_path = Path(state_path) if state_path is not None else None
        self._k1 = k1
        self._b = b
        self._vocabulary: dict[str, int] = {}
        self._document_frequency: dict[str, int] = {}
        self._document_count = 0
        self._total_document_length = 0
        self._load_state()

    def fit_documents(self, texts: Sequence[str]) -> None:
        """Add one ingestion batch to the persistent corpus statistics."""

        tokenized = tuple(_tokens(text) for text in texts)
        for terms in tokenized:
            for token in terms:
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)
            self._document_count += 1
            self._total_document_length += len(terms)
            for token in set(terms):
                self._document_frequency[token] = (
                    self._document_frequency.get(token, 0) + 1
                )
        self._save_state()

    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        """Encode documents using BM25 TF saturation and fitted IDF."""

        if not texts:
            return ()
        if not self._vocabulary:
            self.fit_documents(texts)
        return tuple(self._encode_document(_tokens(text)) for text in texts)

    def embed_query(self, text: str) -> SparseVector:
        """Encode a query without mutating corpus statistics."""

        counts = Counter(token for token in _tokens(text) if token in self._vocabulary)
        pairs = sorted(
            (self._vocabulary[token], 1.0)
            for token in counts
        )
        return SparseVector(
            indices=tuple(index for index, _ in pairs),
            values=tuple(value for _, value in pairs),
        )

    def _encode_document(self, terms: tuple[str, ...]) -> SparseVector:
        counts = Counter(terms)
        average_length = (
            self._total_document_length / self._document_count
            if self._document_count
            else 1.0
        )
        document_length = len(terms)
        pairs: list[tuple[int, float]] = []
        for token, term_count in counts.items():
            denominator = self._k1 * (
                1.0 - self._b + self._b * document_length / average_length
            ) + term_count
            tf_weight = (self._k1 + 1.0) * term_count / denominator
            pairs.append((self._vocabulary[token], tf_weight))
        pairs.sort()
        return SparseVector(
            indices=tuple(index for index, _ in pairs),
            values=tuple(value for _, value in pairs),
        )

    def _load_state(self) -> None:
        if self._state_path is None or not self._state_path.is_file():
            return
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        self._vocabulary = {
            str(token): int(index) for token, index in raw.get("vocabulary", {}).items()
        }
        self._document_frequency = {
            str(token): int(value)
            for token, value in raw.get("document_frequency", {}).items()
        }
        self._document_count = int(raw.get("document_count", 0))
        self._total_document_length = int(raw.get("total_document_length", 0))

    def _save_state(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self._k1,
            "b": self._b,
            "vocabulary": self._vocabulary,
            "document_frequency": self._document_frequency,
            "document_count": self._document_count,
            "total_document_length": self._total_document_length,
        }
        self._state_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _tokens(text: str) -> tuple[str, ...]:
    """Tokenize Turkish/Unicode text consistently for documents and queries."""

    return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(text))
