"""FastAPI composition root for the document intelligence service."""

from collections.abc import AsyncIterator
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from .api.errors import service_error_handler, validation_error_handler
from .api.v1.health import router as health_router
from .api.v1.documents import router as documents_router
from .api.v1.evaluations import router as evaluations_router
from .api.v1.jobs import router as jobs_router
from .api.v1.metrics import router as metrics_router
from .api.v1.queries import legacy_router as legacy_queries_router
from .api.v1.queries import router as queries_router
from .api.v1.search import router as search_router
from .api.v1.demo import router as demo_router
from .api.v1.system import router as system_router
from .application.health_service import HealthService
from .application.chunking_service import DocumentChunkingService
from .application.document_service import DocumentService
from .application.evaluation_service import (
    EvaluationService,
    OfflineEvaluationExecutor,
)
from .application.gold_diagnostic import (
    GoldDiagnosticService,
    GoldEvidenceResolver,
)
from .application.ingestion_service import (
    IngestionPreparationService,
    IngestionService,
)
from .application.ingestion_worker import IngestionWorker
from .application.query_service import QueryService
from .application.model_service import ModelCompatibilityEstimator, ModelService
from .application.retrieval_service import RetrievalService
from .application.ports import HealthProbe, IngestionRegistry
from .domain.errors import ServiceError
from .domain.answerability import AnswerabilityPolicy, AnswerabilityPolicySet
from .domain.evaluation import (
    EvaluationCorpusSnapshot,
    compute_corpus_snapshot_id,
    load_corpus_snapshot,
)
from .domain.ingestion import IngestionLimits, PipelineConfig
from .domain.ingestion import compute_pipeline_fingerprint
from .infrastructure.health_checks import (
    HeartbeatFileProbe,
    HttpHealthProbe,
    LocalModelHealthProbe,
)
from .infrastructure.embeddings.dense import SentenceTransformerEmbedder
from .infrastructure.embeddings.sparse import BM25SparseEncoder
from .infrastructure.parsing.pdf_inspector import PypdfInspector
from .infrastructure.parsing.pdf_text import PypdfTextExtractor
from .infrastructure.parsing.section_markers import (
    KnownSectionMarkerProfileResolver,
    get_section_markers,
)
from .infrastructure.qdrant.chunk_store import QdrantChunkStore
from .infrastructure.qdrant.gold_lookup import QdrantGoldEvidenceLookup
from .infrastructure.qdrant.retriever import QdrantRetriever
from .infrastructure.reranking.cross_encoder import CrossEncoderReranker
from .infrastructure.ollama.answer_generator import OllamaAnswerGenerator
from .infrastructure.ollama.model_runtime import OllamaModelRuntimeAdapter
from .infrastructure.system.host_profile import HostProfileAdapter
from .infrastructure.qdrant.schema import QdrantSchema
from .infrastructure.storage.in_memory_registry import InMemoryIngestionRegistry
from .infrastructure.storage.sqlite_registry import SqliteIngestionRegistry
from .infrastructure.storage.in_memory_evaluation_registry import (
    InMemoryEvaluationRegistry,
)
from qdrant_client import QdrantClient, models
from .observability.request_id import RequestIdMiddleware
from .observability.metrics import MetricsRegistry
from .observability.query_trace import LiveQueryTraceStore
from .settings import Settings

LOGGER = logging.getLogger("document_intelligence_service.lifecycle")


def build_health_service(settings: Settings) -> HealthService:
    """Wire concrete dependency probes into the application service."""

    timeout = settings.dependency_timeout_seconds
    model_catalog = tuple(
        item.strip()
        for item in settings.ollama_model_catalog.split(",")
        if item.strip()
    )
    model_runtime = OllamaModelRuntimeAdapter(
        base_url=str(settings.ollama_url),
        timeout_seconds=timeout,
        allowed_model_ids=tuple(dict.fromkeys((settings.llm_model, *model_catalog))),
    )
    probes: list[HealthProbe] = [
        HttpHealthProbe(
            name="qdrant",
            url=f"{str(settings.qdrant_url).rstrip('/')}/readyz",
            timeout_seconds=timeout,
        ),
        HttpHealthProbe(
            name="ollama",
            url=f"{str(settings.ollama_url).rstrip('/')}/api/tags",
            timeout_seconds=timeout,
        ),
        LocalModelHealthProbe(
            name="llm",
            runtime=model_runtime,
            model_id=settings.llm_model,
        ),
    ]
    if settings.worker_health_enabled:
        probes.append(
            HeartbeatFileProbe(
                name="worker",
                path=settings.worker_heartbeat_path,
                stale_after_seconds=settings.worker_heartbeat_interval_seconds * 4,
            )
        )
    return HealthService(probes=tuple(probes))


def build_model_service(settings: Settings) -> ModelService:
    """Wire host inspection and Ollama through application ports."""

    catalog = tuple(
        item.strip()
        for item in settings.ollama_model_catalog.split(",")
        if item.strip()
    )
    schema = QdrantSchema(collection_name=settings.qdrant_collection)
    return ModelService(
        host_profile=HostProfileAdapter(),
        runtime=OllamaModelRuntimeAdapter(
            base_url=str(settings.ollama_url),
            timeout_seconds=settings.dependency_timeout_seconds,
            allowed_model_ids=tuple(dict.fromkeys((settings.llm_model, *catalog))),
        ),
        generation_model=settings.llm_model,
        embedding_model=settings.dense_model,
        sparse_model=settings.sparse_model,
        reranker_model=settings.reranker_model,
        embedding_dimension=schema.dense_size,
        qdrant_collection=schema.collection_name,
        ollama_catalog=catalog,
        compatibility=ModelCompatibilityEstimator(
            context_length=settings.model_context_length,
        ),
    )


def build_ingestion_registry(settings: Settings) -> IngestionRegistry:
    """Choose the registry implementation without changing application code."""

    if settings.ingestion_registry_backend == "sqlite":
        return SqliteIngestionRegistry(settings.ingestion_database_path)
    return InMemoryIngestionRegistry()


def build_document_service(
    settings: Settings,
    *,
    registry: IngestionRegistry | None = None,
) -> DocumentService:
    """Wire document metadata and vector cleanup to the same registry."""

    return DocumentService(
        registry=registry if registry is not None else build_ingestion_registry(settings),
        vector_store=QdrantChunkStore(
            QdrantClient(url=str(settings.qdrant_url)),
            QdrantSchema(collection_name=settings.qdrant_collection),
        ),
    )


def build_evaluation_service(
    settings: Settings,
    *,
    retrieval_service: RetrievalService | None = None,
) -> EvaluationService:
    """Wire golden-set evaluation to the optional live retrieval adapter."""

    root = _repository_root()
    return EvaluationService(
        registry=InMemoryEvaluationRegistry(),
        executor=OfflineEvaluationExecutor(
            retrieval_service=retrieval_service,
            answerability=_mentor_answerability_policy(settings),
            repo_root=root,
        ),
        artifact_dir=root / settings.evaluation_artifact_dir,
        repo_root=root,
        default_configuration=_evaluation_configuration(settings, root),
    )


def _evaluation_configuration(settings: Settings, root: Path) -> dict[str, object]:
    """Capture the model, retrieval and sanitized host identity for API runs."""

    pipeline_config = build_pipeline_config(
        settings,
        section_marker_profile=settings.evaluation_section_marker_profile,
    )
    pipeline_fingerprint = compute_pipeline_fingerprint(pipeline_config)
    corpus_snapshot = _load_evaluation_corpus_snapshot(
        root,
        settings.qdrant_collection,
    )
    dataset_path = root / "data/evaluations/mentor_program_pdf_rag_golden_v1.jsonl"
    dataset_sha256 = (
        hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if dataset_path.is_file()
        else None
    )
    point_count: int | None = None
    try:
        qdrant = QdrantClient(url=str(settings.qdrant_url))
        count_must: list[models.Condition] = [
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True),
            ),
        ]
        if corpus_snapshot is not None:
            count_must.append(
                models.HasIdCondition(has_id=list(corpus_snapshot.point_ids))
            )
        else:
            count_must.append(
                models.FieldCondition(
                    key="pipeline_fingerprint",
                    match=models.MatchValue(value=pipeline_fingerprint),
                )
            )
        point_count = qdrant.count(
            collection_name=settings.qdrant_collection,
            count_filter=models.Filter(must=count_must),
            exact=True,
        ).count
    except Exception:
        # Health/readiness owns dependency failure reporting.  Evaluation
        # configuration remains inspectable even while Qdrant is restarting.
        point_count = None
    corpus_snapshot_id = (
        corpus_snapshot.snapshot_id
        if corpus_snapshot is not None
        else compute_corpus_snapshot_id(
            dataset_sha256=dataset_sha256,
            qdrant_collection=settings.qdrant_collection,
            point_count=point_count,
            pipeline_fingerprint=pipeline_fingerprint,
        )
    )
    snapshot_expected_count = (
        corpus_snapshot.point_count if corpus_snapshot is not None else None
    )
    return {
        "dataset_version": "mentor_program_pdf_rag_golden_v1",
        "dataset_sha256": dataset_sha256,
        "corpus_snapshot_id": corpus_snapshot_id,
        "corpus_snapshot_basis": (
            "immutable_point_id_manifest"
            if corpus_snapshot is not None
            else "dataset_sha256+qdrant_collection+active_point_count+pipeline_fingerprint"
        ),
        "qdrant_point_count": point_count,
        "corpus_snapshot_point_count": snapshot_expected_count,
        "corpus_snapshot_verified": (
            point_count == snapshot_expected_count
            if corpus_snapshot is not None and point_count is not None
            else None
        ),
        "corpus_snapshot_manifest": (
            "data/evaluations/week2_final_corpus_snapshot_v1.json"
            if corpus_snapshot is not None
            else None
        ),
        "corpus_membership": (
            [
                {"document_id": document_id, "version_id": version_id}
                for document_id, version_id in corpus_snapshot.document_versions
            ]
            if corpus_snapshot is not None
            else []
        ),
        "pipeline_fingerprint": pipeline_fingerprint,
        "retrieval": {
            "candidate_k": settings.retrieval_candidate_k,
            "fusion_k": settings.retrieval_fusion_k,
            "rerank_k": settings.retrieval_rerank_k,
            "rrf_k": settings.rrf_k,
            "fusion_algorithm": "rrf",
        },
        "models": {
            "dense": settings.dense_model,
            "sparse": settings.sparse_model,
            "reranker": settings.reranker_model,
            "llm": settings.llm_model,
        },
        "machine": HostProfileAdapter().detect().as_dict(),
        "qdrant_collection": settings.qdrant_collection,
        "benchmark_recommendation": _load_benchmark_recommendation(root),
    }


def _load_benchmark_recommendation(root: Path) -> dict[str, object]:
    """Load the small committed benchmark summary used by the demo UI."""

    path = root / "data/evaluations/week2_stabilization_summary_v1.json"
    if not path.is_file():
        return {"status": "unavailable"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unavailable"}
    return payload if isinstance(payload, dict) else {"status": "unavailable"}


def _load_evaluation_corpus_snapshot(
    root: Path,
    collection: str,
) -> EvaluationCorpusSnapshot | None:
    """Load the immutable benchmark membership only for its collection."""

    path = root / "data/evaluations/week2_final_corpus_snapshot_v1.json"
    if not path.is_file():
        return None
    try:
        snapshot = load_corpus_snapshot(path)
    except (OSError, ValueError, json.JSONDecodeError):
        LOGGER.warning("evaluation corpus snapshot manifest is invalid: %s", path)
        return None
    return snapshot if snapshot.collection == collection else None


def _repository_root() -> Path:
    """Resolve the repository root in both source and container layouts."""

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "data").is_dir() and (parent / "projects").is_dir():
            return parent
    # The production image has /app/app without the monorepo's sibling folders.
    # Evaluation execution will then fail explicitly if its dataset is absent,
    # but importing and serving the core API remains valid.
    return module_path.parents[1]


def build_pipeline_config(
    settings: Settings,
    *,
    section_marker_profile: str | None = None,
) -> PipelineConfig:
    """Build one shared fingerprint configuration for ingestion stages."""

    return PipelineConfig(
        section_marker_profile=(
            section_marker_profile
            if section_marker_profile is not None
            else settings.section_marker_profile
        ),
        embedding_model=settings.dense_model,
        sparse_encoder=settings.sparse_model,
        reranker_model=settings.reranker_model,
        chunk_size_sentences=settings.chunk_size_sentences,
        chunk_overlap_sentences=settings.chunk_overlap_sentences,
        generic_parent_max_chars=settings.generic_parent_max_chars,
    )


def build_ingestion_service(
    settings: Settings,
    *,
    registry: IngestionRegistry | None = None,
) -> IngestionService:
    """Wire the preparation use case to a selectable persistence adapter."""

    pipeline_config = build_pipeline_config(settings)
    text_extractor = PypdfTextExtractor()
    preparation = IngestionPreparationService(
        limits=IngestionLimits(
            max_upload_bytes=settings.max_upload_bytes,
            max_pdf_pages=settings.max_pdf_pages,
        ),
        pipeline_config=pipeline_config,
        pdf_inspector=PypdfInspector(),
        profile_resolver=KnownSectionMarkerProfileResolver(text_extractor),
    )
    return IngestionService(
        preparation=preparation,
        registry=registry
        if registry is not None
        else build_ingestion_registry(settings),
        max_upload_bytes=settings.max_upload_bytes,
    )


def build_ingestion_worker(
    settings: Settings,
    *,
    registry: IngestionRegistry,
    metrics: MetricsRegistry | None = None,
) -> IngestionWorker:
    """Wire the lazy embedding, parser and Qdrant worker adapters."""

    pipeline_config = build_pipeline_config(settings)
    schema = QdrantSchema(collection_name=settings.qdrant_collection)
    return IngestionWorker(
        registry=registry,
        chunker=DocumentChunkingService(
            extractor=PypdfTextExtractor(),
            pipeline_config=pipeline_config,
        ),
        dense_embedder=SentenceTransformerEmbedder(
            model_name=pipeline_config.embedding_model,
            expected_dimension=schema.dense_size,
        ),
        sparse_embedder=BM25SparseEncoder(state_path=settings.bm25_state_path),
        vector_store=QdrantChunkStore(
            QdrantClient(url=str(settings.qdrant_url)),
            schema,
        ),
        section_markers=get_section_markers(settings.section_marker_profile),
        section_markers_by_profile={
            profile: get_section_markers(profile)
            for profile in (
                "auto",
                "generic_v1",
                "none",
                "mentor_program_v1",
                "mentor_program_week2_v1",
            )
        },
        embedding_model=pipeline_config.embedding_model,
        sparse_encoder=pipeline_config.sparse_encoder,
        parser_version=pipeline_config.parser_version,
        chunker_version=pipeline_config.chunker_version,
        metrics=metrics,
    )


def build_retrieval_service(
    settings: Settings,
    *,
    section_marker_profile: str | None = None,
    registry: IngestionRegistry | None = None,
) -> RetrievalService:
    """Wire lazy query embedders to the active-version Qdrant retriever."""

    effective_profile = (
        section_marker_profile
        if section_marker_profile is not None
        else settings.section_marker_profile
    )
    pipeline_config = build_pipeline_config(
        settings,
        section_marker_profile=effective_profile,
    )
    evaluation_snapshot = _load_evaluation_corpus_snapshot(
        _repository_root(),
        settings.qdrant_collection,
    )
    corpus_point_ids = (
        evaluation_snapshot.point_ids
        if evaluation_snapshot is not None
        and compute_pipeline_fingerprint(pipeline_config)
        == evaluation_snapshot.pipeline_fingerprint
        else ()
    )
    schema = QdrantSchema(collection_name=settings.qdrant_collection)
    return RetrievalService(
        dense_embedder=SentenceTransformerEmbedder(
            model_name=pipeline_config.embedding_model,
            expected_dimension=schema.dense_size,
        ),
        sparse_embedder=BM25SparseEncoder(state_path=settings.bm25_state_path),
        retriever=QdrantRetriever(
            QdrantClient(url=str(settings.qdrant_url)),
            schema,
            pipeline_fingerprint=(
                compute_pipeline_fingerprint(pipeline_config)
                if effective_profile
                not in {"auto", "generic_v1", "none"}
                else None
            ),
            corpus_point_ids=corpus_point_ids,
            active_version_ids_provider=(
                registry.active_version_ids
                if registry is not None and not corpus_point_ids
                else None
            ),
        ),
        reranker=CrossEncoderReranker(model_name=pipeline_config.reranker_model),
        candidate_limit=settings.retrieval_candidate_k,
        rrf_k=settings.rrf_k,
        fusion_limit=settings.retrieval_fusion_k,
        rerank_limit=settings.retrieval_rerank_k,
        reranker_default_enabled=settings.reranker_enabled,
        dense_model=settings.dense_model,
        sparse_model=settings.sparse_model,
        reranker_model=settings.reranker_model,
    )


def build_query_service(
    settings: Settings,
    *,
    retrieval_service: RetrievalService | None = None,
    metrics: MetricsRegistry | None = None,
) -> QueryService:
    """Wire answerability policy and the host-local Ollama boundary."""

    return QueryService(
        retrieval_service=retrieval_service
        if retrieval_service is not None
        else build_retrieval_service(settings),
        answerability=_product_answerability_policies(settings),
        answer_generator=OllamaAnswerGenerator(
            base_url=str(settings.ollama_url),
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_evidence_chars=settings.llm_max_evidence_chars,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
        metrics=metrics,
    )


def build_gold_diagnostic_service(
    settings: Settings,
    *,
    document_service: DocumentService,
    ingestion_service: IngestionService,
    query_service: QueryService,
    retrieval_service: RetrievalService | None,
) -> GoldDiagnosticService:
    """Wire the curated Demo Lab without changing the normal query path."""

    root = _repository_root()
    manifest_path = root / "data/evaluations/atlas_orion_demo/atlas_orion_diagnostic_cases.json"
    lookup = QdrantGoldEvidenceLookup(
        QdrantClient(url=str(settings.qdrant_url)),
        QdrantSchema(collection_name=settings.qdrant_collection),
    )
    resolver = GoldEvidenceResolver(
        manifest_path=manifest_path,
        document_service=document_service,
        lookup=lookup,
    )
    return GoldDiagnosticService(
        manifest_path=manifest_path,
        asset_dir=manifest_path.parent,
        document_service=document_service,
        ingestion_service=ingestion_service,
        query_service=query_service,
        resolver=resolver,
        retrieval_service=retrieval_service,
    )


def _mentor_answerability_policy(settings: Settings) -> AnswerabilityPolicy:
    """Return the unchanged Week-2 mentor calibration policy."""

    return AnswerabilityPolicy(
        min_dense_score=settings.answerability_min_dense_score,
        min_sparse_score=settings.answerability_min_sparse_score,
        min_rerank_score=settings.answerability_min_rerank_score,
        min_margin=settings.answerability_min_margin,
        min_coverage=settings.answerability_min_coverage,
        profile_name="mentor_program_v1",
        calibration_id="week2_stabilization_v1",
    )


def _product_answerability_policies(settings: Settings) -> AnswerabilityPolicySet:
    """Keep mentor/default behavior while selecting the generic calibration."""

    generic = AnswerabilityPolicy(
        min_dense_score=settings.answerability_generic_min_dense_score,
        min_sparse_score=settings.answerability_min_sparse_score,
        min_rerank_score=settings.answerability_min_rerank_score,
        min_margin=settings.answerability_min_margin,
        min_coverage=settings.answerability_generic_min_coverage,
        profile_name="generic_v1",
        calibration_id=settings.answerability_generic_calibration_id,
    )
    return AnswerabilityPolicySet(
        default=_mentor_answerability_policy(settings),
        by_chunking_profile={"generic_v1": generic},
    )


def create_app(
    *,
    settings: Settings | None = None,
    health_service: HealthService | None = None,
    ingestion_service: IngestionService | None = None,
    ingestion_worker: IngestionWorker | None = None,
    document_service: DocumentService | None = None,
    evaluation_service: EvaluationService | None = None,
    retrieval_service: RetrievalService | None = None,
    query_service: QueryService | None = None,
    metrics_registry: MetricsRegistry | None = None,
    demo_trace_store: LiveQueryTraceStore | None = None,
    model_service: ModelService | None = None,
    gold_diagnostic_service: GoldDiagnosticService | None = None,
) -> FastAPI:
    """Create an application with replaceable dependencies for testing."""

    resolved_settings = settings or Settings()
    resolved_metrics = metrics_registry or MetricsRegistry()
    resolved_health_service = health_service or build_health_service(resolved_settings)
    resolved_ingestion_worker = ingestion_worker
    resolved_document_service = document_service
    resolved_evaluation_service = evaluation_service
    resolved_retrieval_service = retrieval_service
    resolved_query_service = query_service
    # Product queries and the frozen benchmark intentionally use different
    # retrieval scopes.  An injected retrieval service is a test/application
    # override and must be reused by evaluation; the production Compose wiring
    # creates a separate, explicit benchmark-profile adapter below.
    evaluation_retrieval_service = (
        resolved_retrieval_service if retrieval_service is not None else None
    )
    resolved_demo_trace_store = demo_trace_store or LiveQueryTraceStore(
        ttl_seconds=resolved_settings.demo_trace_ttl_seconds,
        max_runs=resolved_settings.demo_trace_max_runs,
    )
    resolved_model_service = model_service or build_model_service(resolved_settings)
    if ingestion_service is None:
        registry = build_ingestion_registry(resolved_settings)
        resolved_ingestion_service = build_ingestion_service(
            resolved_settings,
            registry=registry,
        )
        if resolved_document_service is None:
            resolved_document_service = build_document_service(
                resolved_settings,
                registry=registry,
            )
        if (
            resolved_ingestion_worker is None
            and resolved_settings.ingestion_registry_backend == "sqlite"
            and resolved_settings.embedded_worker
        ):
            resolved_ingestion_worker = build_ingestion_worker(
                resolved_settings,
                registry=registry,
                metrics=resolved_metrics,
            )
        if (
            resolved_retrieval_service is None
            and resolved_settings.ingestion_registry_backend == "sqlite"
        ):
            resolved_retrieval_service = build_retrieval_service(
                resolved_settings,
                registry=registry,
            )
        if (
            resolved_query_service is None
            and resolved_retrieval_service is not None
        ):
            resolved_query_service = build_query_service(
                resolved_settings,
                retrieval_service=resolved_retrieval_service,
                metrics=resolved_metrics,
            )
        if resolved_evaluation_service is None:
            if (
                evaluation_retrieval_service is None
                and resolved_settings.ingestion_registry_backend == "sqlite"
            ):
                evaluation_retrieval_service = build_retrieval_service(
                    resolved_settings,
                    section_marker_profile=(
                        resolved_settings.evaluation_section_marker_profile
                    ),
                )
            elif evaluation_retrieval_service is None:
                evaluation_retrieval_service = resolved_retrieval_service
            resolved_evaluation_service = build_evaluation_service(
                resolved_settings,
                retrieval_service=evaluation_retrieval_service,
            )
    else:
        resolved_ingestion_service = ingestion_service
        if (
            resolved_retrieval_service is None
            and resolved_settings.ingestion_registry_backend == "sqlite"
        ):
            resolved_retrieval_service = build_retrieval_service(
                resolved_settings,
                registry=ingestion_service.registry,
            )
        if resolved_document_service is None:
            resolved_document_service = DocumentService(
                registry=ingestion_service.registry,
            )
        if resolved_evaluation_service is None:
            if (
                evaluation_retrieval_service is None
                and resolved_settings.ingestion_registry_backend == "sqlite"
            ):
                evaluation_retrieval_service = build_retrieval_service(
                    resolved_settings,
                    section_marker_profile=(
                        resolved_settings.evaluation_section_marker_profile
                    ),
                )
            elif evaluation_retrieval_service is None:
                evaluation_retrieval_service = resolved_retrieval_service
            resolved_evaluation_service = build_evaluation_service(
                resolved_settings,
                retrieval_service=evaluation_retrieval_service,
            )
        if (
            resolved_query_service is None
            and resolved_retrieval_service is not None
        ):
            resolved_query_service = build_query_service(
                resolved_settings,
                retrieval_service=resolved_retrieval_service,
                metrics=resolved_metrics,
            )

    resolved_gold_diagnostic_service = gold_diagnostic_service
    if (
        resolved_gold_diagnostic_service is None
        and resolved_document_service is not None
        and resolved_ingestion_service is not None
        and resolved_query_service is not None
    ):
        resolved_gold_diagnostic_service = build_gold_diagnostic_service(
            resolved_settings,
            document_service=resolved_document_service,
            ingestion_service=resolved_ingestion_service,
            query_service=resolved_query_service,
            retrieval_service=resolved_retrieval_service,
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.model_service = resolved_model_service
        application.state.gold_diagnostic_service = resolved_gold_diagnostic_service
        application.state.model_pull_store = getattr(
            application.state,
            "model_pull_store",
            {},
        )
        application.state.health_service = resolved_health_service
        application.state.ingestion_service = resolved_ingestion_service
        application.state.ingestion_worker = resolved_ingestion_worker
        application.state.document_service = resolved_document_service
        application.state.evaluation_service = resolved_evaluation_service
        application.state.retrieval_service = resolved_retrieval_service
        application.state.query_service = resolved_query_service
        application.state.metrics = resolved_metrics
        try:
            if resolved_settings.preload_models:
                for adapter in (resolved_retrieval_service, resolved_ingestion_worker):
                    warmup = getattr(adapter, "warmup", None)
                    if callable(warmup):
                        LOGGER.info("preloading model adapters")
                        await asyncio.to_thread(warmup)
            resolved_health_service.mark_started()
            yield
        finally:
            resolved_health_service.mark_stopped()

    application = FastAPI(
        title="Document Intelligence Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(RequestIdMiddleware)
    application.add_exception_handler(ServiceError, service_error_handler)
    application.add_exception_handler(RequestValidationError, validation_error_handler)
    application.state.health_service = resolved_health_service
    application.state.ingestion_service = resolved_ingestion_service
    application.state.ingestion_worker = resolved_ingestion_worker
    application.state.document_service = resolved_document_service
    application.state.evaluation_service = resolved_evaluation_service
    application.state.retrieval_service = resolved_retrieval_service
    application.state.query_service = resolved_query_service
    application.state.metrics = resolved_metrics
    application.state.settings = resolved_settings
    application.state.demo_trace_store = resolved_demo_trace_store
    application.state.model_service = resolved_model_service
    application.state.gold_diagnostic_service = resolved_gold_diagnostic_service
    application.state.model_pull_store = {}
    application.include_router(health_router, prefix="/v1")
    application.include_router(documents_router, prefix="/v1")
    application.include_router(evaluations_router, prefix="/v1")
    application.include_router(jobs_router, prefix="/v1")
    application.include_router(queries_router, prefix="/v1")
    application.include_router(legacy_queries_router, prefix="/v1")
    application.include_router(search_router, prefix="/v1")
    application.include_router(metrics_router, prefix="/v1")
    application.include_router(demo_router, prefix="/v1")
    application.include_router(system_router, prefix="/v1")
    return application


app = create_app()
