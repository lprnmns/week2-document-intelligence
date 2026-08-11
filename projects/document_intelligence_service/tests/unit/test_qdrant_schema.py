"""Unit tests for named-vector Qdrant schema and point mapping."""

import pytest
from qdrant_client import QdrantClient, models
from unittest.mock import patch

from projects.document_intelligence_service.app.domain.chunks import ChildChunk
from projects.document_intelligence_service.app.domain.evaluation import (
    load_corpus_snapshot,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.chunk_store import (
    QdrantChunkStore,
    SparseEmbedding,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.retriever import (
    QdrantRetriever,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.schema import (
    QdrantSchema,
    QdrantSchemaError,
    QdrantSchemaManager,
)


def make_chunk(
    chunk_id: str = "chunk-1",
    *,
    document_id: str = "doc-1",
    version_id: str = "ver-1",
) -> ChildChunk:
    """Create a deterministic child chunk fixture."""

    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=f"{document_id}:{version_id}:parent:000",
        document_id=document_id,
        version_id=version_id,
        source="guide.pdf",
        title="RAG",
        text="Qdrant kanıt adaylarını saklar.",
        chunk_index=1,
        page_start=2,
        page_end=2,
        token_count_estimate=5,
        text_hash="a" * 64,
    )


def make_store(dense_size: int = 2) -> QdrantChunkStore:
    """Create an isolated in-memory Qdrant store."""

    schema = QdrantSchema(
        collection_name="test_named_vectors",
        dense_size=dense_size,
    )
    return QdrantChunkStore(QdrantClient(":memory:"), schema)


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
def test_schema_creates_named_dense_and_sparse_vectors() -> None:
    client = QdrantClient(":memory:")
    schema = QdrantSchema(collection_name="schema_test", dense_size=2)
    manager = QdrantSchemaManager(client, schema)

    manager.ensure_collection()
    info = client.get_collection("schema_test")

    assert isinstance(info.config.params.vectors, dict)
    assert "dense" in info.config.params.vectors
    assert isinstance(info.config.params.sparse_vectors, dict)
    assert "bm25" in info.config.params.sparse_vectors
    assert info.config.params.vectors["dense"].size == 2


def test_point_id_is_stable_for_same_version_and_chunk() -> None:
    first = QdrantChunkStore.point_id("ver-1", "chunk-1")
    second = QdrantChunkStore.point_id("ver-1", "chunk-1")
    different_version = QdrantChunkStore.point_id("ver-2", "chunk-1")

    assert first == second
    assert first != different_version


def test_upsert_stores_named_vectors_and_payload() -> None:
    store = make_store()
    chunk = make_chunk()
    sparse = SparseEmbedding(indices=(1, 4), values=(0.8, 0.2))

    store.upsert(
        chunks=[chunk],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-1",
        language="tr",
        content_hash="b" * 64,
        embedding_model="dense-v1",
        sparse_encoder="bm25-v1",
        parser_version="pypdf-1",
        chunker_version="chunker-1",
        is_active=False,
    )
    store.upsert(
        chunks=[chunk],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-1",
        language="tr",
        content_hash="b" * 64,
        embedding_model="dense-v1",
        sparse_encoder="bm25-v1",
        parser_version="pypdf-1",
        chunker_version="chunker-1",
        is_active=True,
    )

    client_info = store.client.count(store.collection_name, exact=True)
    point = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[store.point_id("ver-1", "chunk-1")],
        with_vectors=True,
    )[0]
    assert client_info.count == 1
    assert point.payload is not None
    assert point.payload["page_start"] == 2
    assert point.payload["is_active"] is True
    assert point.payload["content_hash"] == "b" * 64
    assert point.payload["embedding_model"] == "dense-v1"
    assert point.payload["sparse_encoder"] == "bm25-v1"
    assert point.payload["parser_version"] == "pypdf-1"
    assert point.payload["chunker_version"] == "chunker-1"
    assert isinstance(point.vector, dict)


def test_upsert_rejects_dense_dimension_mismatch() -> None:
    store = make_store(dense_size=3)

    with pytest.raises(ValueError, match="expected 3"):
        store.upsert(
            chunks=[make_chunk()],
            dense_vectors=[(1.0, 0.0)],
            sparse_vectors=[SparseEmbedding((1,), (1.0,))],
            pipeline_fingerprint="pipe-1",
        )


def test_upsert_rejects_misaligned_sparse_values() -> None:
    store = make_store()

    with pytest.raises(ValueError, match="equal lengths"):
        store.upsert(
            chunks=[make_chunk()],
            dense_vectors=[(1.0, 0.0)],
            sparse_vectors=[SparseEmbedding((1, 2), (1.0,))],
            pipeline_fingerprint="pipe-1",
        )


def test_existing_dense_dimension_mismatch_fails_startup_validation() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="mismatch",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    manager = QdrantSchemaManager(
        client,
        QdrantSchema(collection_name="mismatch", dense_size=3),
    )

    with pytest.raises(QdrantSchemaError, match="dimension"):
        manager.ensure_collection()


def test_existing_non_idf_sparse_collection_is_rejected_by_bm25_schema() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="idf_mismatch",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.NONE)
        },
    )
    manager = QdrantSchemaManager(
        client,
        QdrantSchema(collection_name="idf_mismatch", dense_size=2),
    )

    with pytest.raises(QdrantSchemaError, match="IDF"):
        manager.ensure_collection()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect in the local Qdrant")
def test_existing_compatible_collection_adds_missing_payload_indexes() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="legacy_payload_indexes",
        vectors_config={
            "dense": models.VectorParams(size=2, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    manager = QdrantSchemaManager(
        client,
        QdrantSchema(collection_name="legacy_payload_indexes", dense_size=2),
    )
    with patch.object(
        client,
        "create_payload_index",
        wraps=client.create_payload_index,
    ) as create_payload_index:
        manager.ensure_collection()

    indexed_fields = {
        call.kwargs["field_name"] for call in create_payload_index.call_args_list
    }
    assert {"content_hash", "embedding_model", "sparse_encoder"} <= indexed_fields


def test_stage_verify_activate_hides_previous_version() -> None:
    store = make_store()
    sparse = SparseEmbedding(indices=(1, 4), values=(0.8, 0.2))

    first = make_chunk(version_id="ver-1")
    store.stage_version(
        chunks=[first],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-1",
        language="tr",
    )
    first_verification = store.verify_version(
        document_id="doc-1",
        version_id="ver-1",
        expected_chunk_count=1,
    )
    assert first_verification.is_valid
    store.activate_version(
        document_id="doc-1",
        version_id="ver-1",
        verification=first_verification,
    )

    second = make_chunk(version_id="ver-2")
    store.stage_version(
        chunks=[second],
        dense_vectors=[(0.0, 1.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-2",
        language="tr",
    )
    second_verification = store.verify_version(
        document_id="doc-1",
        version_id="ver-2",
        expected_chunk_count=1,
    )
    assert second_verification.is_valid
    store.activate_version(
        document_id="doc-1",
        version_id="ver-2",
        verification=second_verification,
    )

    first_point = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[store.point_id("ver-1", "chunk-1")],
    )[0]
    second_point = store.client.retrieve(
        collection_name=store.collection_name,
        ids=[store.point_id("ver-2", "chunk-1")],
    )[0]
    assert first_point.payload is not None
    assert second_point.payload is not None
    assert first_point.payload["is_active"] is False
    assert second_point.payload["is_active"] is True


def test_retriever_searches_active_named_dense_and_sparse_vectors() -> None:
    schema = QdrantSchema(collection_name="retrieval_test", dense_size=2)
    store = QdrantChunkStore(QdrantClient(":memory:"), schema)
    chunk = make_chunk()
    sparse = SparseEmbedding(indices=(1, 4), values=(0.8, 0.2))
    store.stage_version(
        chunks=[chunk],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-1",
        language="tr",
    )
    verification = store.verify_version(
        document_id="doc-1",
        version_id="ver-1",
        expected_chunk_count=1,
    )
    store.activate_version(
        document_id="doc-1",
        version_id="ver-1",
        verification=verification,
    )
    retriever = QdrantRetriever(store.client, schema)

    dense_hits = retriever.search_dense(
        query_vector=(1.0, 0.0),
        limit=5,
        document_ids=("doc-1",),
    )
    sparse_hits = retriever.search_sparse(
        query_vector=sparse,
        limit=5,
        document_ids=("doc-1",),
    )

    assert len(dense_hits) == 1
    assert len(sparse_hits) == 1
    assert dense_hits[0].source_id == "chunk-1"
    assert sparse_hits[0].page_start == 2


def test_registry_scope_prevents_mixed_versions_in_dense_and_sparse_search() -> None:
    """An old physical payload cannot leak while cleanup is still pending."""

    schema = QdrantSchema(collection_name="authoritative_version_scope", dense_size=2)
    store = QdrantChunkStore(QdrantClient(":memory:"), schema)
    sparse = SparseEmbedding(indices=(1,), values=(1.0,))
    for version_id, dense_vector in (
        ("ver-1", (1.0, 0.0)),
        ("ver-2", (0.0, 1.0)),
    ):
        chunk = make_chunk(version_id=version_id)
        store.stage_version(
            chunks=[chunk],
            dense_vectors=[dense_vector],
            sparse_vectors=[sparse],
            pipeline_fingerprint=f"pipe-{version_id}",
            language="tr",
        )
        verification = store.verify_version(
            document_id="doc-1",
            version_id=version_id,
            expected_chunk_count=1,
        )
        store.activate_version(
            document_id="doc-1",
            version_id=version_id,
            verification=verification,
            cleanup_previous=False,
        )

    retriever = QdrantRetriever(
        store.client,
        schema,
        active_version_ids_provider=lambda document_ids, tenant_id: ("ver-2",),
    )
    dense_hits = retriever.search_dense(
        query_vector=(1.0, 0.0),
        limit=5,
        document_ids=("doc-1",),
    )
    sparse_hits = retriever.search_sparse(
        query_vector=sparse,
        limit=5,
        document_ids=("doc-1",),
    )

    assert [hit.version_id for hit in dense_hits] == ["ver-2"]
    assert [hit.version_id for hit in sparse_hits] == ["ver-2"]


def test_retriever_enforces_tenant_and_acl_filters_before_returning_sources() -> None:
    schema = QdrantSchema(collection_name="acl_retrieval_test", dense_size=2)
    store = QdrantChunkStore(QdrantClient(":memory:"), schema)
    sparse = SparseEmbedding(indices=(1,), values=(1.0,))
    tenant_a_chunk = make_chunk(document_id="tenant-a-doc", version_id="a-v1")
    tenant_b_chunk = make_chunk(document_id="tenant-b-doc", version_id="b-v1")

    store.stage_version(
        chunks=[tenant_a_chunk],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-a",
        tenant_id="tenant_a",
        acl_tags=("finance",),
    )
    verification_a = store.verify_version(
        document_id="tenant-a-doc",
        version_id="a-v1",
        expected_chunk_count=1,
    )
    store.activate_version(
        document_id="tenant-a-doc",
        version_id="a-v1",
        verification=verification_a,
    )
    store.stage_version(
        chunks=[tenant_b_chunk],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="pipe-b",
        tenant_id="tenant_b",
        acl_tags=("finance",),
    )
    verification_b = store.verify_version(
        document_id="tenant-b-doc",
        version_id="b-v1",
        expected_chunk_count=1,
    )
    store.activate_version(
        document_id="tenant-b-doc",
        version_id="b-v1",
        verification=verification_b,
    )
    retriever = QdrantRetriever(store.client, schema)

    tenant_a_hits = retriever.search_dense(
        query_vector=(1.0, 0.0),
        limit=5,
        document_ids=(),
        tenant_id="tenant_a",
        acl_tags=("finance",),
    )
    tenant_b_without_tag = retriever.search_dense(
        query_vector=(1.0, 0.0),
        limit=5,
        document_ids=(),
        tenant_id="tenant_b",
        acl_tags=("hr",),
    )

    assert [item.document_id for item in tenant_a_hits] == ["tenant-a-doc"]
    assert tenant_b_without_tag == ()


def test_frozen_snapshot_membership_excludes_same_fingerprint_points() -> None:
    """A matching pipeline is metadata; point IDs define frozen membership."""

    schema = QdrantSchema(collection_name="snapshot_membership_test", dense_size=2)
    store = QdrantChunkStore(QdrantClient(":memory:"), schema)
    sparse = SparseEmbedding(indices=(1,), values=(1.0,))
    frozen = make_chunk(
        chunk_id="frozen-child",
        document_id="frozen-doc",
        version_id="frozen-version",
    )
    later = make_chunk(
        chunk_id="later-child",
        document_id="later-doc",
        version_id="later-version",
    )

    for chunk in (frozen, later):
        store.stage_version(
            chunks=[chunk],
            dense_vectors=[(1.0, 0.0)],
            sparse_vectors=[sparse],
            pipeline_fingerprint="same-pipeline",
            language="tr",
        )
        verification = store.verify_version(
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            expected_chunk_count=1,
        )
        store.activate_version(
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            verification=verification,
        )

    retriever = QdrantRetriever(
        store.client,
        schema,
        pipeline_fingerprint="same-pipeline",
        corpus_point_ids=(store.point_id(frozen.version_id, frozen.chunk_id),),
    )
    hits = retriever.search_dense(
        query_vector=(1.0, 0.0),
        limit=10,
        document_ids=(),
    )

    assert [hit.source_id for hit in hits] == ["frozen-child"]
    assert [hit.document_id for hit in hits] == ["frozen-doc"]


def test_frozen_snapshot_manifest_stays_fixed_after_same_pipeline_ingestion() -> None:
    """A later same-fingerprint version cannot expand the frozen snapshot."""

    committed_snapshot = load_corpus_snapshot(
        "data/evaluations/week2_final_corpus_snapshot_v1.json"
    )
    assert committed_snapshot.point_count == 26
    assert len(committed_snapshot.point_ids) == 26

    schema = QdrantSchema(collection_name="snapshot_manifest_regression", dense_size=2)
    store = QdrantChunkStore(QdrantClient(":memory:"), schema)
    sparse = SparseEmbedding(indices=(1,), values=(1.0,))
    frozen_chunks = tuple(
        make_chunk(
            chunk_id=f"frozen-child-{index:02d}",
            document_id="frozen-doc",
            version_id="frozen-version",
        )
        for index in range(26)
    )
    store.stage_version(
        chunks=frozen_chunks,
        dense_vectors=[(1.0, 0.0)] * len(frozen_chunks),
        sparse_vectors=[sparse] * len(frozen_chunks),
        pipeline_fingerprint="same-pipeline",
        language="tr",
    )
    frozen_verification = store.verify_version(
        document_id="frozen-doc",
        version_id="frozen-version",
        expected_chunk_count=26,
    )
    store.activate_version(
        document_id="frozen-doc",
        version_id="frozen-version",
        verification=frozen_verification,
    )

    later = make_chunk(
        chunk_id="later-child",
        document_id="later-doc",
        version_id="later-version",
    )
    store.stage_version(
        chunks=[later],
        dense_vectors=[(1.0, 0.0)],
        sparse_vectors=[sparse],
        pipeline_fingerprint="same-pipeline",
        language="tr",
    )
    later_verification = store.verify_version(
        document_id="later-doc",
        version_id="later-version",
        expected_chunk_count=1,
    )
    store.activate_version(
        document_id="later-doc",
        version_id="later-version",
        verification=later_verification,
    )

    frozen_point_ids = tuple(
        store.point_id(chunk.version_id, chunk.chunk_id) for chunk in frozen_chunks
    )
    retriever = QdrantRetriever(
        store.client,
        schema,
        pipeline_fingerprint="same-pipeline",
        corpus_point_ids=frozen_point_ids,
    )
    hits = retriever.search_dense(
        query_vector=(1.0, 0.0),
        limit=50,
        document_ids=(),
    )

    assert len(hits) == 26
    assert {hit.document_id for hit in hits} == {"frozen-doc"}
    assert {hit.source_id for hit in hits} == {
        chunk.chunk_id for chunk in frozen_chunks
    }


def test_committed_week2_snapshot_has_exact_frozen_membership() -> None:
    snapshot = load_corpus_snapshot(
        "data/evaluations/week2_final_corpus_snapshot_v1.json"
    )

    assert snapshot.snapshot_id == (
        "c5e87f7e063769adef368866854d8e45f7b7f9856f905abe9cebe31783262b25"
    )
    assert snapshot.point_count == 26
    assert len(snapshot.point_ids) == 26
    assert snapshot.document_versions == (
        (
            "doc_b20e5ee9255db127f8394092773d5e1c17b5a9e258849e82db27273d44fe9898",
            "ver_04dc4c638fc7ea19267b6cc7524d925bae6d93efcec388a19285c7befef98d61",
        ),
    )
