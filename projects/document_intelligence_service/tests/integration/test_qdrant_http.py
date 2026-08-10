"""Ephemeral Qdrant integration coverage for the named-vector boundary."""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import QdrantClient

from projects.document_intelligence_service.app.domain.chunks import ChildChunk
from projects.document_intelligence_service.app.domain.vectors import SparseVector
from projects.document_intelligence_service.app.infrastructure.qdrant.chunk_store import (
    QdrantChunkStore,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.retriever import (
    QdrantRetriever,
)
from projects.document_intelligence_service.app.infrastructure.qdrant.schema import (
    QdrantSchema,
)


@pytest.mark.integration
def test_http_qdrant_stage_verify_activate_and_filter() -> None:
    """Prove persistence, named vectors and tenant/ACL filters against HTTP Qdrant."""

    url = os.environ.get("QDRANT_INTEGRATION_URL")
    if not url:
        pytest.skip("QDRANT_INTEGRATION_URL is not configured")

    collection = f"ci_{uuid.uuid4().hex}"
    schema = QdrantSchema(collection_name=collection, dense_size=2)
    client = QdrantClient(url=url)
    store = QdrantChunkStore(client, schema)
    chunk = ChildChunk(
        chunk_id="doc-a:child:001",
        parent_id="doc-a:parent:000",
        document_id="doc-a",
        version_id="ver-a",
        source="sample.pdf",
        title="Sample",
        text="Qdrant hybrid search",
        chunk_index=1,
        page_start=1,
        page_end=1,
        token_count_estimate=3,
        text_hash="text-hash",
        parent_text="Qdrant hybrid search",
    )
    try:
        store.stage_version(
            chunks=(chunk,),
            dense_vectors=((1.0, 0.0),),
            sparse_vectors=(SparseVector(indices=(1, 2), values=(1.0, 1.0)),),
            pipeline_fingerprint="pipeline-a",
            tenant_id="tenant-a",
            acl_tags=("finance",),
        )
        verification = store.verify_version(
            document_id="doc-a",
            version_id="ver-a",
            expected_chunk_count=1,
        )
        assert verification.is_valid
        store.activate_version(
            document_id="doc-a",
            version_id="ver-a",
            verification=verification,
        )

        retriever = QdrantRetriever(client, schema)
        allowed = retriever.search_dense(
            query_vector=(1.0, 0.0),
            limit=5,
            document_ids=(),
            tenant_id="tenant-a",
            acl_tags=("finance",),
        )
        denied = retriever.search_dense(
            query_vector=(1.0, 0.0),
            limit=5,
            document_ids=(),
            tenant_id="tenant-b",
            acl_tags=("finance",),
        )
        assert [item.document_id for item in allowed] == ["doc-a"]
        assert denied == ()
    finally:
        client.delete_collection(collection)
