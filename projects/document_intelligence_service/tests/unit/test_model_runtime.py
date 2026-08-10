"""Tests for strict local-runtime model identifier handling."""

import asyncio

import pytest

from projects.document_intelligence_service.app.infrastructure.ollama.model_runtime import (
    OllamaModelRuntimeAdapter,
)


def test_model_pull_rejects_shell_like_identifier_before_network_call() -> None:
    adapter = OllamaModelRuntimeAdapter(
        base_url="http://127.0.0.1:11434",
        allowed_model_ids=("qwen3:4b",),
    )
    with pytest.raises(ValueError):
        asyncio.run(adapter.pull_model("qwen3:4b; touch /tmp/pwned"))


def test_model_pull_rejects_non_allowlisted_identifier() -> None:
    adapter = OllamaModelRuntimeAdapter(
        base_url="http://127.0.0.1:11434",
        allowed_model_ids=("qwen3:4b",),
    )
    with pytest.raises(ValueError):
        asyncio.run(adapter.pull_model("gemma3:4b"))
