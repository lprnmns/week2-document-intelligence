"""Unit tests for lazy dense and deterministic sparse embedding adapters."""

from pathlib import Path

from projects.document_intelligence_service.app.infrastructure.embeddings.sparse import (
    BM25SparseEncoder,
    HashingSparseEncoder,
)


def test_hashing_sparse_encoder_is_stable_and_sorted() -> None:
    encoder = HashingSparseEncoder(feature_count=128)

    first = encoder.embed_documents(("Qdrant Türkçe arama",))[0]
    second = encoder.embed_documents(("Qdrant Türkçe arama",))[0]

    assert first == second
    assert first.indices == tuple(sorted(first.indices))
    assert len(first.indices) == len(set(first.indices))
    assert all(value > 0 for value in first.values)


def test_hashing_sparse_encoder_counts_repeated_terms() -> None:
    encoder = HashingSparseEncoder(feature_count=128)

    repeated = encoder.embed_documents(("qdrant qdrant",))[0]
    single = encoder.embed_documents(("qdrant",))[0]

    assert repeated.indices == single.indices
    assert repeated.values[0] > single.values[0]


def test_bm25_encoder_fits_exact_vocabulary_and_survives_restart(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "bm25.json"
    encoder = BM25SparseEncoder(state_path=state_path)
    encoder.fit_documents(
        (
            "Qdrant belge araması",
            "Qdrant vector store",
            "Ollama model servisi",
        )
    )

    document = encoder.embed_documents(("Qdrant Qdrant belge araması",))[0]
    query = encoder.embed_query("Qdrant araması bilinmeyen")
    restored = BM25SparseEncoder(state_path=state_path)
    restored_query = restored.embed_query("Qdrant araması bilinmeyen")

    assert document.indices == tuple(sorted(document.indices))
    assert len(document.indices) == len(set(document.indices))
    assert query == restored_query
    assert len(query.indices) == 2
    assert all(value > 0 for value in document.values)


def test_bm25_query_does_not_mutate_corpus_state(tmp_path: Path) -> None:
    state_path = tmp_path / "bm25.json"
    encoder = BM25SparseEncoder(state_path=state_path)
    encoder.fit_documents(("qdrant belge",))
    before = state_path.read_text(encoding="utf-8")

    query = encoder.embed_query("yeni terim")

    assert query.indices == ()
    assert state_path.read_text(encoding="utf-8") == before
