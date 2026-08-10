"""Document catalog use cases shared by HTTP and future worker adapters."""

import asyncio
import logging

from ..domain.errors import ErrorCode, ServiceError
from ..domain.ingestion import (
    DocumentPage,
    DocumentSnapshot,
    normalize_tenant_id,
)
from .ports import ChunkVectorStore, IngestionRegistry
from ..observability.audit import emit_audit


class DocumentService:
    """Coordinate document metadata lifecycle with optional vector cleanup."""

    def __init__(
        self,
        *,
        registry: IngestionRegistry,
        vector_store: ChunkVectorStore | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._registry = registry
        self._vector_store = vector_store
        self._logger = logger or logging.getLogger(
            "document_intelligence_service.documents"
        )

    async def list_documents(
        self,
        limit: int,
        cursor: str | None,
        tenant_id: str = "default",
    ) -> DocumentPage:
        """Return a bounded page of logical document metadata."""

        return await self._registry.list_documents(
            limit,
            cursor,
            normalize_tenant_id(tenant_id),
        )

    async def get_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> DocumentSnapshot:
        """Return a document or a stable not-found error."""

        normalized_tenant = normalize_tenant_id(tenant_id)
        document = await self._registry.get_document(
            document_id,
            normalized_tenant,
        )
        if document is None:
            raise ServiceError(
                code=ErrorCode.DOCUMENT_NOT_FOUND,
                message="Document was not found",
            )
        return document

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> None:
        """Remove vector points and mark metadata deleted after busy checks."""

        normalized_tenant = normalize_tenant_id(tenant_id)
        document = await self.get_document(document_id, normalized_tenant)
        if document.status.value == "indexing":
            raise ServiceError(
                code=ErrorCode.DOCUMENT_BUSY,
                message="Document has an ingestion job in progress",
            )
        if self._vector_store is not None:
            try:
                await asyncio.to_thread(
                    self._vector_store.delete_document,
                    document.document_id,
                )
            except Exception as error:
                raise ServiceError(
                    code=ErrorCode.DEPENDENCY_UNAVAILABLE,
                    message="Vector store is unavailable for document deletion",
                ) from error
        await self._registry.delete_document(document.document_id, normalized_tenant)
        emit_audit(
            action="document.deleted",
            result="success",
            document_id=document.document_id,
            tenant_id=normalized_tenant,
            metadata={"status_before_delete": document.status.value},
            logger=self._logger,
        )
