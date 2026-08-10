"""Document resource contract routes."""

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Header,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from ..errors import openapi_error_responses
from ...application.ingestion_service import IngestionService
from ...domain.ingestion import DocumentSnapshot
from ...observability.request_id import get_request_id
from .contracts import (
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_202_ACCEPTED: {"model": DocumentUploadResponse},
        **openapi_error_responses(),
    },
)
async def create_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF document")],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    acl_tags_header: Annotated[str | None, Header(alias="X-ACL-Tags")] = None,
) -> DocumentUploadResponse:
    """Validate and accept a PDF for asynchronous ingestion."""

    ingestion_service: IngestionService = request.app.state.ingestion_service
    content = await file.read(ingestion_service.max_upload_bytes + 1)
    receipt = await ingestion_service.accept_receipt(
        content=content,
        filename=file.filename or "upload.pdf",
        content_type=file.content_type,
        idempotency_key=idempotency_key,
        tenant_id=tenant_id,
        acl_tags=tuple(
            tag.strip()
            for tag in (acl_tags_header or "").split(",")
            if tag.strip()
        ),
    )
    worker = getattr(request.app.state, "ingestion_worker", None)
    if worker is not None:
        background_tasks.add_task(worker.run_job, receipt.job_id)
    return DocumentUploadResponse(
        document_id=receipt.document_id,
        version_id=receipt.version_id,
        job_id=receipt.job_id,
        status=receipt.status,
        request_id=get_request_id(),
        idempotent_hit=receipt.idempotent_hit,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={**openapi_error_responses()},
)
async def list_documents(
    request: Request,
    tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
) -> DocumentListResponse:
    """List documents with bounded cursor pagination."""

    page = await request.app.state.document_service.list_documents(
        limit,
        cursor,
        tenant_id or "default",
    )
    return DocumentListResponse(
        items=[_summary(document) for document in page.items],
        next_cursor=page.next_cursor,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    responses={**openapi_error_responses()},
)
async def get_document(
    request: Request,
    document_id: Annotated[str, Path(min_length=1, max_length=128)],
    tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
) -> DocumentDetailResponse:
    """Return one document and its available versions."""

    document = await request.app.state.document_service.get_document(
        document_id,
        tenant_id or "default",
    )
    return DocumentDetailResponse(
        **_summary(document).model_dump(),
        available_version_ids=list(document.available_version_ids),
    )


@router.delete(
    "/{document_id}",
    response_model=None,
    status_code=status.HTTP_204_NO_CONTENT,
    responses={**openapi_error_responses()},
)
async def delete_document(
    request: Request,
    document_id: Annotated[str, Path(min_length=1, max_length=128)],
    tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
) -> Response:
    """Delete a document unless an active ingestion job makes it busy."""

    await request.app.state.document_service.delete_document(
        document_id,
        tenant_id or "default",
    )
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"X-Request-ID": get_request_id()},
    )


def _summary(document: DocumentSnapshot) -> DocumentSummary:
    """Map the domain read model without exposing adapter details."""

    return DocumentSummary(
        document_id=document.document_id,
        title=document.title,
        content_hash=document.content_hash,
        active_version_id=document.active_version_id,
        status=document.status,
        created_at=document.created_at,
        tenant_id=document.tenant_id,
    )
