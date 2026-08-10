"""Unit tests for the bounded Ollama generation adapter."""

import asyncio

import pytest

from projects.document_intelligence_service.app.domain.generation import (
    AnswerGenerationError,
)
from projects.document_intelligence_service.app.domain.retrieval import RetrievedChunk
from projects.document_intelligence_service.app.infrastructure.ollama.answer_generator import (
    OllamaAnswerGenerator,
)


def evidence(source_id: str, text: str, parent_text: str | None = None) -> RetrievedChunk:
    """Create one compact evidence fixture."""

    return RetrievedChunk(
        source_id=source_id,
        document_id="doc-1",
        version_id="ver-1",
        parent_id="parent-1",
        title="Guide",
        text=text,
        page_start=1,
        page_end=1,
        score=0.8,
        rank=1,
        parent_text=parent_text,
    )


class FakeResponse:
    """Minimal successful Ollama response."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"response": "Kanıta dayalı cevap."}


class FakeAsyncClient:
    """Capture one request without contacting a local model runtime."""

    last_payload: dict[str, object] | None = None

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
    ) -> FakeResponse:
        assert url.endswith("/api/generate")
        FakeAsyncClient.last_payload = json
        return FakeResponse()


def test_adapter_bounds_prompt_and_returns_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projects.document_intelligence_service.app.infrastructure.ollama.answer_generator.httpx.AsyncClient",
        FakeAsyncClient,
    )
    generator = OllamaAnswerGenerator(
        base_url="http://127.0.0.1:11434/",
        max_evidence_chars=20,
        max_output_tokens=32,
    )

    async def scenario() -> None:
        result = await generator.generate(
            question="Soru",
            evidence=(
                evidence("source-1", "A" * 100),
                evidence("source-2", "B" * 100),
            ),
        )
        assert result.answer == "Kanıta dayalı cevap."
        assert result.provider == "ollama"
        assert result.model == "gemma3:4b"

    asyncio.run(scenario())
    assert FakeAsyncClient.last_payload is not None
    assert FakeAsyncClient.last_payload["stream"] is False
    options = FakeAsyncClient.last_payload["options"]
    assert isinstance(options, dict)
    assert options["num_predict"] == 32
    prompt = FakeAsyncClient.last_payload["prompt"]
    assert isinstance(prompt, str)
    assert "source=source-2" not in prompt
    assert "BEGIN_USER_QUESTION" in prompt
    assert "BEGIN_UNTRUSTED_EVIDENCE" in prompt


def test_adapter_rejects_empty_evidence_before_http() -> None:
    generator = OllamaAnswerGenerator(base_url="http://127.0.0.1:11434")

    async def scenario() -> None:
        with pytest.raises(AnswerGenerationError, match="without evidence"):
            await generator.generate(question="Soru", evidence=())

    asyncio.run(scenario())


def test_adapter_uses_parent_context_for_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "projects.document_intelligence_service.app.infrastructure.ollama.answer_generator.httpx.AsyncClient",
        FakeAsyncClient,
    )
    generator = OllamaAnswerGenerator(base_url="http://127.0.0.1:11434")

    async def scenario() -> None:
        await generator.generate(
            question="Soru",
            evidence=(
                evidence(
                    "source-1",
                    "child evidence",
                    parent_text="parent evidence with surrounding context",
                ),
            ),
        )

    asyncio.run(scenario())
    assert FakeAsyncClient.last_payload is not None
    prompt = FakeAsyncClient.last_payload["prompt"]
    assert isinstance(prompt, str)
    assert "parent evidence with surrounding context" in prompt
    assert "child evidence" not in prompt
