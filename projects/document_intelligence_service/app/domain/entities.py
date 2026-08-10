"""Domain enumerations shared by application use cases and API contracts."""

from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle state of a document version."""

    INDEXING = "indexing"
    ACTIVE = "active"
    FAILED = "failed"
    DELETED = "deleted"


class JobStatus(StrEnum):
    """Lifecycle state of an asynchronous ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvaluationRunStatus(StrEnum):
    """Lifecycle state of an offline evaluation run."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StageStatus(StrEnum):
    """Lifecycle state of one ingestion pipeline stage."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class Decision(StrEnum):
    """Final answerability decision for a query."""

    ANSWERED = "answered"
    NO_ANSWER = "no_answer"


class RetrievalMode(StrEnum):
    """Supported evidence retrieval strategies."""

    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class NoAnswerReason(StrEnum):
    """Machine-readable reasons for declining to answer."""

    NO_EVIDENCE = "NO_EVIDENCE"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    SECURITY_POLICY = "SECURITY_POLICY"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
