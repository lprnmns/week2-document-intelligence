"""Environment-backed service settings."""

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Validated runtime configuration loaded from DIS_* variables."""

    model_config = SettingsConfigDict(
        env_prefix="DIS_",
        env_file=".env",
        extra="ignore",
    )

    environment: str = "development"
    service_name: str = "document-intelligence-service"
    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:6333")
    # Final Week-2 demo/evaluation snapshot. Legacy collections remain
    # addressable only through an explicit DIS_QDRANT_COLLECTION override.
    qdrant_collection: str = "document_chunks_week2_final_v1"
    ollama_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")
    dependency_timeout_seconds: float = Field(default=1.0, gt=0, le=10)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    ingestion_registry_backend: Literal["memory", "sqlite"] = "memory"
    ingestion_database_path: str = "data/ingestions.sqlite3"
    embedded_worker: bool = True
    # Once the real adapters are wired, model loading belongs to the
    # application lifecycle rather than the first user request. Tests can
    # still disable this explicitly when they inject fakes.
    preload_models: bool = True
    worker_poll_interval_seconds: float = Field(default=1.0, gt=0, le=60)
    worker_stale_after_seconds: float = Field(default=300.0, gt=0, le=3600)
    worker_health_enabled: bool = False
    worker_heartbeat_path: str = "data/worker_heartbeat.json"
    worker_heartbeat_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    bm25_state_path: str = "data/bm25_state.json"
    evaluation_artifact_dir: str = "projects/document_intelligence_service/eval/results/api_runs"
    section_marker_profile: Literal[
        "auto",
        "generic_v1",
        "none",
        "mentor_program_v1",
        "mentor_program_week2_v1",
    ] = "auto"
    evaluation_section_marker_profile: Literal[
        "mentor_program_v1",
        "mentor_program_week2_v1",
    ] = "mentor_program_v1"
    dense_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    sparse_model: str = "bm25_qdrant_idf_v2"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    chunk_size_sentences: int = Field(default=3, ge=1, le=20)
    chunk_overlap_sentences: int = Field(default=1, ge=0, le=19)
    generic_parent_max_chars: int = Field(default=4000, ge=256, le=100_000)
    retrieval_candidate_k: int = Field(default=30, ge=1, le=50)
    retrieval_fusion_k: int = Field(default=20, ge=1, le=50)
    retrieval_rerank_k: int = Field(default=5, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    log_level: str = "INFO"
    log_query_text: bool = False
    reranker_enabled: bool = False
    llm_model: str = "gemma3:4b"
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    # 2,400 chars is the smallest measured bounded context that retained the
    # deadline date+time across the real five-source selected evidence set.
    llm_max_evidence_chars: int = Field(default=2_400, gt=0, le=32_000)
    llm_max_output_tokens: int = Field(default=64, gt=0, le=1024)
    answerability_min_dense_score: float = Field(default=0.338, ge=0, le=1)
    answerability_generic_min_dense_score: float = Field(default=0.247, ge=0, le=1)
    answerability_min_sparse_score: float = Field(default=0.1, ge=0)
    answerability_min_rerank_score: float = -5.0
    answerability_min_margin: float = 0.0
    answerability_min_coverage: float = Field(default=0.0, ge=0, le=1)
    answerability_generic_min_coverage: float = Field(default=0.367, ge=0, le=1)
    answerability_generic_calibration_id: str = "generic_document_answerability_v1"
    demo_trace_enabled: bool = True
    demo_trace_ttl_seconds: float = Field(default=900.0, gt=0, le=86_400)
    demo_trace_max_runs: int = Field(default=32, ge=1, le=500)
    system_profile_enabled: bool = True
    local_model_management_enabled: bool = False
    ollama_model_catalog: str = "gemma3:4b,qwen3:4b"
    model_context_length: int = Field(default=4096, ge=256, le=131_072)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "Settings":
        """Reject an impossible overlapping sentence window early."""

        if self.chunk_overlap_sentences >= self.chunk_size_sentences:
            raise ValueError(
                "chunk_overlap_sentences must be smaller than chunk_size_sentences"
            )
        return self
