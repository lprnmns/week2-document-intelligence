"""Version 1 request and response contracts.

These models define the external API before application implementations are
connected. They intentionally contain no Qdrant, embedding or Ollama types.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...domain.entities import (
    Decision,
    DocumentStatus,
    EvaluationRunStatus,
    JobStatus,
    NoAnswerReason,
    RetrievalMode,
    StageStatus,
)
from ...domain.evidence_validation import EvidenceWarningCode


class PageQuery(BaseModel):
    """Bounded cursor pagination shared by list endpoints."""

    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=256)


class DocumentUploadResponse(BaseModel):
    """Accepted asynchronous document ingestion response."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "document_id": "doc_123",
                    "version_id": "ver_001",
                    "job_id": "job_456",
                    "status": "indexing",
                    "request_id": "req_demo",
                }
            ]
        }
    )

    document_id: str
    version_id: str
    job_id: str
    status: DocumentStatus = DocumentStatus.INDEXING
    request_id: str
    idempotent_hit: bool = False


class DocumentSummary(BaseModel):
    """Safe document metadata for list responses."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str
    title: str
    content_hash: str
    active_version_id: str | None
    status: DocumentStatus
    created_at: datetime
    tenant_id: str = "default"


class DocumentListResponse(BaseModel):
    """Bounded document listing response."""

    items: list[DocumentSummary]
    next_cursor: str | None


class DocumentDetailResponse(DocumentSummary):
    """Document detail response with version information."""

    available_version_ids: list[str]


class DeleteDocumentResponse(BaseModel):
    """Accepted document deletion response."""

    document_id: str
    status: str = "deleted"
    request_id: str


class JobResponse(BaseModel):
    """Asynchronous ingestion job status."""

    job_id: str
    document_id: str
    status: JobStatus
    progress_percent: int = Field(ge=0, le=100)
    error_code: str | None
    request_id: str
    current_stage: str | None = None
    stages: list["StageEventResponse"] = Field(default_factory=list)
    page_count: int | None = Field(default=None, ge=1)
    point_count: int | None = Field(default=None, ge=0)
    error_message: str | None = None
    failed_stage: str | None = None
    attempt_count: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    next_attempt_at: datetime | None = None
    last_attempt_at: datetime | None = None


class StageEventResponse(BaseModel):
    """Public, non-sensitive ingestion timeline entry."""

    name: str
    status: StageStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    inputs: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    outputs: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    decision: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


class EvaluationRunRequest(BaseModel):
    """Bounded configuration for one offline golden-set evaluation."""

    evaluation_type: Literal["retrieval", "answerability"] = "retrieval"
    dataset: Literal["mentor_program_pdf_rag_golden_v1"] = (
        "mentor_program_pdf_rag_golden_v1"
    )
    split: Literal["all", "development", "validation", "test"] = "all"
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    reranker_enabled: bool = False


class EvaluationRunResponse(BaseModel):
    """Observable state and metrics of one evaluation run."""

    run_id: str
    status: EvaluationRunStatus
    evaluation_type: str
    dataset: str
    split: str
    mode: RetrievalMode
    top_k: int
    reranker_enabled: bool
    requested_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    case_count: int | None = Field(default=None, ge=0)
    metrics: dict[str, int | float | str | bool | None] | None = None
    artifact_path: str | None = None
    git_sha: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    configuration: dict[str, object] = Field(default_factory=dict)


class EvaluationRunListResponse(BaseModel):
    """Bounded evaluation run listing."""

    items: list[EvaluationRunResponse]


class SourceResponse(BaseModel):
    """Evidence source returned to a caller."""

    source_id: str
    document_id: str
    version_id: str
    chunk_id: str
    parent_id: str
    page: int | None = Field(default=None, ge=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    title: str | None
    snippet: str
    excerpt: str
    score: float | None
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    fusion_rank: int | None = Field(default=None, ge=1)
    rerank_rank: int | None = Field(default=None, ge=1)
    selected_as_evidence: bool = False
    rank_delta: int | None = None


class NoAnswerInfo(BaseModel):
    """Stable machine-readable explanation for an intentionally skipped answer."""

    reason_code: NoAnswerReason
    message: str
    searched_document_ids: list[str] = Field(default_factory=list, max_length=100)


class RetrievalInfo(BaseModel):
    """Debug-safe retrieval counts and selected strategy."""

    mode: RetrievalMode
    dense_candidates: int = Field(ge=0)
    sparse_candidates: int = Field(ge=0)
    rrf_candidates: int = Field(ge=0)
    reranked_candidates: int = Field(ge=0)
    candidate_limit: int = Field(default=0, ge=0)
    fusion_limit: int = Field(default=0, ge=0)
    rerank_limit: int = Field(default=0, ge=0)
    reranker_enabled: bool = False
    reranker_skipped_reason: str | None = None
    dense_distribution: list["DocumentCandidateDistributionResponse"] = Field(
        default_factory=list
    )
    sparse_distribution: list["DocumentCandidateDistributionResponse"] = Field(
        default_factory=list
    )
    dense_model: str | None = None
    sparse_model: str | None = None
    reranker_model: str | None = None


class DocumentCandidateDistributionResponse(BaseModel):
    """Candidate count by document for one retrieval stage."""

    document_id: str
    title: str
    count: int = Field(ge=0)


class RetrievalDebugCandidateResponse(BaseModel):
    """Safe candidate diagnostics shown only when debug is requested."""

    source_id: str
    retrieval_rank: int | None = None
    rerank_rank: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    matched_terms: list[str] = Field(default_factory=list)
    document_id: str = ""
    title: str = ""
    page_start: int | None = None
    page_end: int | None = None
    excerpt: str = ""
    fusion_rank: int | None = None
    selected_as_evidence: bool = False
    rank_delta: int | None = None


class RetrievalDebugResponse(BaseModel):
    """Bounded retrieval diagnostics for the UI and error analysis."""

    candidates: list[RetrievalDebugCandidateResponse]


class LatencyBreakdown(BaseModel):
    """Stage-level latency measurements in milliseconds."""

    embedding_ms: float = Field(ge=0)
    search_ms: float = Field(ge=0)
    rerank_ms: float = Field(ge=0)
    llm_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class ModelInfo(BaseModel):
    """Model metadata; null means the LLM stage was skipped."""

    provider: str | None
    model: str | None


class OutputWarningResponse(BaseModel):
    """Structured output/evidence concern for human or policy review."""

    code: EvidenceWarningCode
    message: str
    values: list[str]


class QueryRequest(BaseModel):
    """Question and bounded retrieval controls."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "Yerel model karşılaştırmasında hangi değerler ölçülmelidir?",
                    "retrieval_mode": "hybrid",
                    "top_k": 5,
                    "include_debug": False,
                }
            ]
        }
    )

    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=20)
    include_debug: bool = False
    reranker_enabled: bool | None = None
    tenant_id: str | None = Field(default=None, max_length=128)
    acl_tags: list[str] = Field(default_factory=list, max_length=50)


class QueryResponse(BaseModel):
    """Answer or explicit no-answer response contract."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "decision": "no_answer",
                    "answer": None,
                    "no_answer": {
                        "reason_code": "NO_EVIDENCE",
                        "message": "Sufficient evidence was not found.",
                        "searched_document_ids": [],
                    },
                    "no_answer_reason": "NO_EVIDENCE",
                    "sources": [],
                    "retrieval": {
                        "mode": "hybrid",
                        "dense_candidates": 30,
                        "sparse_candidates": 30,
                        "rrf_candidates": 20,
                        "reranked_candidates": 5,
                    },
                    "model": {"provider": None, "model": None},
                    "warnings": [],
                    "latency": {
                        "embedding_ms": 12.4,
                        "search_ms": 18.1,
                        "rerank_ms": 38.2,
                        "llm_ms": 0,
                        "total_ms": 70.1,
                    },
                    "request_id": "req_demo",
                }
            ]
        }
    )

    decision: Decision
    answer: str | None
    no_answer_reason: NoAnswerReason | None
    no_answer: NoAnswerInfo | None = None
    sources: list[SourceResponse]
    retrieval: RetrievalInfo
    model: ModelInfo
    warnings: list[OutputWarningResponse] = Field(default_factory=list)
    latency: LatencyBreakdown
    debug: RetrievalDebugResponse | None = None
    request_id: str

    @model_validator(mode="after")
    def validate_decision_fields(self) -> "QueryResponse":
        """Keep answer and no-answer fields mutually consistent."""

        if self.decision is Decision.ANSWERED:
            if not self.answer or self.no_answer_reason is not None or self.no_answer is not None:
                raise ValueError("answered responses require answer and no reason")
        if self.decision is Decision.NO_ANSWER and self.no_answer_reason is not None and self.no_answer is None:
            self.no_answer = NoAnswerInfo(
                reason_code=self.no_answer_reason,
                message="Sufficient evidence was not found; the LLM was skipped.",
            )
        if self.decision is Decision.NO_ANSWER and (
            self.answer is not None
            or self.no_answer_reason is None
            or self.no_answer is None
        ):
            raise ValueError("no-answer responses require a reason and no answer")
        if (
            self.decision is Decision.NO_ANSWER
            and self.no_answer is not None
            and self.no_answer_reason is not self.no_answer.reason_code
        ):
            raise ValueError("no-answer reason fields must agree")
        return self


class SearchRequest(BaseModel):
    """Evidence-only search request for retrieval debugging."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "Qdrant ne işe yarar?",
                    "retrieval_mode": "hybrid",
                    "top_k": 10,
                }
            ]
        }
    )

    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=50)
    include_debug: bool = False
    reranker_enabled: bool | None = None
    tenant_id: str | None = Field(default=None, max_length=128)
    acl_tags: list[str] = Field(default_factory=list, max_length=50)


class SearchResponse(BaseModel):
    """Retrieval candidates without LLM generation."""

    sources: list[SourceResponse]
    retrieval: RetrievalInfo
    latency: LatencyBreakdown
    debug: RetrievalDebugResponse | None = None
    request_id: str
