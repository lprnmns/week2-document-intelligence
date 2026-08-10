"""Tests for retrieval benchmark orchestration with fake infrastructure."""

from dataclasses import replace
from collections.abc import Sequence

from projects.document_intelligence_service.app.domain.entities import RetrievalMode
from projects.document_intelligence_service.app.domain.retrieval import RetrievedChunk
from projects.document_intelligence_service.app.domain.retrieval import RetrievalResult
from projects.document_intelligence_service.eval.contracts import GoldenCase
from projects.document_intelligence_service.eval.runner import run_retrieval_benchmark


class FakeRetrievalService:
    """Return a reranked final result and a larger candidate window."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    def search(
        self,
        *,
        question: str,
        mode: RetrievalMode,
        top_k: int,
        document_ids: Sequence[str] = (),
        reranker_enabled: bool | None = None,
    ) -> RetrievalResult:
        del mode, top_k, document_ids, reranker_enabled
        self.questions.append(question)
        candidate_window = (
            _chunk("embedding", rank=1),
            _chunk("rag", rank=2),
        )
        return RetrievalResult(
            mode="hybrid",
            candidates=(replace(candidate_window[1], rank=1),),
            dense_candidates=2,
            sparse_candidates=2,
            rrf_candidates=2,
            embedding_ms=10.0,
            search_ms=5.0,
            candidate_window=candidate_window,
        )


def _chunk(title: str, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        source_id=f"chunk-{title}",
        document_id="mentor-week1",
        version_id="version-1",
        parent_id=f"parent-{title}",
        title=title,
        text=title,
        page_start=1,
        page_end=1,
        score=0.5,
        rank=rank,
    )


def test_runner_excludes_warmup_from_case_count_and_keeps_candidate_window() -> None:
    service = FakeRetrievalService()
    case = GoldenCase(
        case_id="case-1",
        question="RAG nasıl çalışır?",
        category="direct_fact",
        split="development",
        expected_answerable=True,
        relevant_sections=("rag",),
    )

    run = run_retrieval_benchmark(
        retrieval_service=service,
        cases=(case,),
        mode=RetrievalMode.HYBRID,
        warmup_questions=("warmup",),
    )

    assert run.cases_run == 1
    assert run.warmup_count == 1
    assert service.questions == ["warmup", "RAG nasıl çalışır?"]
    assert run.metrics.recall_at_1 == 1.0
    assert run.metrics.candidate_recall_at_20 == 1.0
    assert run.observations[0].candidate_window[0].title == "embedding"
