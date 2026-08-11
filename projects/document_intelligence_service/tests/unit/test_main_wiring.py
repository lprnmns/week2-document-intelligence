"""Composition-root regression tests for product and benchmark separation."""

from types import SimpleNamespace
from typing import cast

import pytest

from projects.document_intelligence_service.app import main
from projects.document_intelligence_service.app.application.health_service import (
    HealthService,
)
from projects.document_intelligence_service.app.application.document_service import (
    DocumentService,
)
from projects.document_intelligence_service.app.application.model_service import (
    ModelService,
)
from projects.document_intelligence_service.app.infrastructure.storage.in_memory_registry import (
    InMemoryIngestionRegistry,
)
from projects.document_intelligence_service.app.settings import Settings


def test_compose_keeps_product_and_frozen_evaluation_profiles_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The product uses AUTO while evaluation is pinned to the mentor profile."""

    profiles: list[str | None] = []
    retrievals: list[object] = []
    evaluation_retrieval: list[object] = []
    registry = InMemoryIngestionRegistry()

    def fake_build_retrieval_service(
        settings: Settings,
        *,
        section_marker_profile: str | None = None,
        registry: InMemoryIngestionRegistry | None = None,
    ) -> object:
        del settings
        del registry
        profiles.append(section_marker_profile)
        service = object()
        retrievals.append(service)
        return service

    def fake_build_evaluation_service(
        settings: Settings,
        *,
        retrieval_service: object | None = None,
    ) -> object:
        del settings
        assert retrieval_service is not None
        evaluation_retrieval.append(retrieval_service)
        return object()

    monkeypatch.setattr(main, "build_ingestion_registry", lambda settings: registry)
    monkeypatch.setattr(
        main,
        "build_ingestion_service",
        lambda settings, *, registry: SimpleNamespace(registry=registry),
    )
    monkeypatch.setattr(main, "build_retrieval_service", fake_build_retrieval_service)
    monkeypatch.setattr(main, "build_query_service", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "build_evaluation_service", fake_build_evaluation_service)

    settings = Settings(
        ingestion_registry_backend="sqlite",
        embedded_worker=False,
        preload_models=False,
    )
    application = main.create_app(
        settings=settings,
        health_service=HealthService(()),
        document_service=cast(DocumentService, object()),
        model_service=cast(ModelService, object()),
    )

    assert profiles == [None, "mentor_program_v1"]
    assert len(retrievals) == 2
    assert evaluation_retrieval == [retrievals[1]]
    assert application.state.retrieval_service is retrievals[0]
