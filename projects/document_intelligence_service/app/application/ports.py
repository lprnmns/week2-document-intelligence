"""Ports implemented by infrastructure adapters."""

from typing import Protocol
from collections.abc import Awaitable, Callable
from collections.abc import Sequence

from ..domain.entities import DocumentStatus
from ..domain.health import DependencyHealth
from ..domain.ingestion import (
    DocumentPage,
    DocumentSnapshot,
    IngestionReceipt,
    JobSnapshot,
    PdfInspection,
    PreparedIngestion,
    ChunkingResolution,
    StageEvent,
    VersionVerification,
)
from ..domain.evaluation import EvaluationRunSnapshot
from ..domain.generation import GeneratedAnswer
from ..domain.chunks import ChildChunk, PageText
from ..domain.retrieval import RetrievedChunk
from ..domain.vectors import SparseVector
from ..domain.model_profile import ModelMetadata, RuntimeStatus
from ..domain.system_profile import SystemProfile


class HealthProbe(Protocol):
    """Contract for checking one required dependency."""

    async def check(self) -> DependencyHealth:
        """Return the dependency's current health without raising."""

        ...


class HostProfilePort(Protocol):
    """Infrastructure boundary for sanitized host inspection."""

    def detect(self) -> SystemProfile:
        """Return hardware facts without user paths or environment values."""

        ...


class ModelRuntimePort(Protocol):
    """Local runtime boundary; no shell commands cross this interface."""

    async def check_runtime(self) -> RuntimeStatus:
        """Check runtime reachability."""

        ...

    async def list_installed_models(self) -> tuple[ModelMetadata, ...]:
        """Return installed runtime models."""

        ...

    async def pull_model(
        self,
        model_id: str,
        on_progress: Callable[[dict[str, object]], Awaitable[None]] | None = None,
    ) -> None:
        """Pull one validated model through the runtime API."""

        ...


class PdfInspector(Protocol):
    """Port for page-aware PDF structure inspection."""

    def inspect(self, content: bytes, max_pages: int) -> PdfInspection:
        """Validate PDF structure and return bounded page metadata."""

        ...


class PageTextExtractor(Protocol):
    """Port for page-preserving selectable text extraction."""

    def extract(self, content: bytes) -> tuple[PageText, ...]:
        """Return normalized text while retaining page boundaries."""

        ...


class ChunkingProfileResolver(Protocol):
    """Resolve a requested profile before the ingestion identity is stored."""

    def resolve(
        self,
        content: bytes,
        requested_profile: str,
    ) -> ChunkingResolution:
        """Return the effective profile and safe detection metadata."""

        ...


class DenseEmbedder(Protocol):
    """Port for a dense embedding model loaded outside request handling."""

    @property
    def dimension(self) -> int:
        """Return the fixed output dimension of the embedding model."""

        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode a bounded batch of texts into dense vectors."""

        ...


class SparseEmbedder(Protocol):
    """Port for a deterministic lexical/sparse encoder."""

    def embed_documents(self, texts: Sequence[str]) -> tuple[SparseVector, ...]:
        """Encode a bounded batch of texts into sparse vectors."""

        ...


class ChunkVectorStore(Protocol):
    """Port for staging and activating versioned retrieval chunks."""

    def stage_version(
        self,
        *,
        chunks: Sequence[ChildChunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector],
        pipeline_fingerprint: str,
        language: str,
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
        content_hash: str | None = None,
        embedding_model: str | None = None,
        sparse_encoder: str | None = None,
        parser_version: str | None = None,
        chunker_version: str | None = None,
        chunk_config_hash: str | None = None,
        chunking_profile_requested: str | None = None,
        chunking_profile_resolved: str | None = None,
        structure_detection_method: str | None = None,
        structure_confidence: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        """Write a version as inactive points."""

        ...


    def verify_version(
        self,
        *,
        document_id: str,
        version_id: str,
        expected_chunk_count: int,
    ) -> VersionVerification:
        """Validate schema, point count and staged metadata."""

        ...

    def activate_version(
        self,
        *,
        document_id: str,
        version_id: str,
        verification: VersionVerification,
    ) -> None:
        """Make the verified version visible to retrieval."""

        ...

    def delete_document(self, document_id: str) -> None:
        """Delete all vector points belonging to a logical document."""

        ...

    def discard_version(self, document_id: str, version_id: str) -> None:
        """Remove an unpublishable staged version during failure cleanup."""

        ...


class ChunkRetriever(Protocol):
    """Port for active-version dense and sparse evidence search."""

    def search_dense(
        self,
        *,
        query_vector: Sequence[float],
        limit: int,
        document_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> tuple[RetrievedChunk, ...]:
        """Return dense candidates from active points only."""

        ...

    def search_sparse(
        self,
        *,
        query_vector: SparseVector,
        limit: int,
        document_ids: Sequence[str],
        tenant_id: str = "default",
        acl_tags: Sequence[str] = ("public",),
    ) -> tuple[RetrievedChunk, ...]:
        """Return sparse candidates from active points only."""

        ...


class Reranker(Protocol):
    """Port for bounded question/evidence cross-encoder reranking."""

    def rerank(
        self,
        *,
        question: str,
        candidates: Sequence[RetrievedChunk],
        limit: int,
    ) -> tuple[RetrievedChunk, ...]:
        """Return the highest-scoring bounded evidence candidates."""

        ...


class AnswerGenerator(Protocol):
    """Port for grounded answer generation after the answerability gate."""

    async def generate(
        self,
        *,
        question: str,
        evidence: Sequence[RetrievedChunk],
    ) -> GeneratedAnswer:
        """Generate an answer using only the supplied evidence."""

        ...


class IngestionRegistry(Protocol):
    """Port for idempotent document and job state."""

    async def accept(
        self,
        prepared: PreparedIngestion,
        idempotency_key: str | None,
    ) -> IngestionReceipt:
        """Persist an accepted ingestion identity and return its receipt."""

        ...


    async def get_job(self, job_id: str) -> JobSnapshot | None:
        """Return one job snapshot, if it exists."""

        ...

    async def claim_job(
        self,
        job_id: str,
        stale_after_seconds: float = 300.0,
    ) -> JobSnapshot | None:
        """Atomically claim one queued or recoverable stale job."""

        ...

    async def list_recoverable_jobs(
        self,
        limit: int = 10,
        stale_after_seconds: float = 300.0,
    ) -> tuple[str, ...]:
        """Return bounded queued/retryable/stale-running job IDs."""

        ...

    async def get_staged_content(self, job_id: str) -> bytes | None:
        """Return staged bytes for a future ingestion worker."""

        ...

    async def get_staged_ingestion(self, job_id: str) -> PreparedIngestion | None:
        """Return the complete staged ingestion identity for a worker."""

        ...

    async def list_documents(
        self,
        limit: int,
        cursor: str | None,
        tenant_id: str = "default",
    ) -> DocumentPage:
        """Return a bounded page of logical documents."""

        ...

    async def get_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> DocumentSnapshot | None:
        """Return one logical document read model, if known."""

        ...

    async def delete_document(
        self,
        document_id: str,
        tenant_id: str = "default",
    ) -> None:
        """Mark all versions deleted unless an ingestion is still running."""

        ...

    async def update_job(self, snapshot: JobSnapshot) -> None:
        """Persist a worker progress transition."""

        ...

    async def record_stage_event(self, job_id: str, event: StageEvent) -> None:
        """Persist a stage transition and expose its latest snapshot."""

        ...

    async def set_document_status(
        self,
        *,
        document_id: str,
        version_id: str,
        status: DocumentStatus,
    ) -> None:
        """Persist the lifecycle of one version independently from its job."""

        ...


class EvaluationRegistry(Protocol):
    """Port for bounded evaluation run metadata."""

    async def create(self, snapshot: EvaluationRunSnapshot) -> None:
        """Persist a newly queued run."""

        ...

    async def get(self, run_id: str) -> EvaluationRunSnapshot | None:
        """Return one evaluation run, if known."""

        ...

    async def update(self, snapshot: EvaluationRunSnapshot) -> None:
        """Replace one run state transition."""

        ...

    async def list(self, limit: int) -> tuple[EvaluationRunSnapshot, ...]:
        """Return the newest bounded run snapshots."""

        ...
