"""Named section-marker profiles for known document families."""

from typing import Literal

from ...domain.chunks import SectionMarker
from ...domain.errors import ServiceError
from ...domain.ingestion import ChunkingResolution
from ...application.ports import PageTextExtractor

SectionMarkerProfile = Literal[
    "auto",
    "generic_v1",
    "none",
    "mentor_program_v1",
    "mentor_program_week2_v1",
]

GENERIC_PROFILE = "generic_v1"


MENTOR_PROGRAM_V1_MARKERS: tuple[SectionMarker, ...] = (
    SectionMarker("purpose", "Programın Amacı"),
    SectionMarker("model_fundamentals", "01 Modelin nasıl düşündüğünü anla"),
    SectionMarker("embedding", "02 Embedding ve anlamsal aramayı somutlaştır"),
    SectionMarker("rag", "03 RAG akışının tamamını kur"),
    SectionMarker("local_model", "04 Yerel modeli ayağa kaldır ve karşılaştır"),
    SectionMarker("corporate_problem", "05 Gerçek bir kurumsal problem seç"),
    SectionMarker("deliverables", "Teslim Paketi"),
)


def _week2_marker(
    page_number: int,
    section_id: str,
    heading: str,
) -> SectionMarker:
    """Anchor a Week 2 heading at the extracted page boundary."""

    return SectionMarker(
        section_id,
        f"ALPEREN MANAS / AI ENGINEERING PROGRAM Sayfa {page_number} {heading}",
    )


MENTOR_PROGRAM_WEEK2_V1_MARKERS: tuple[SectionMarker, ...] = (
    _week2_marker(1, "cover", "YAPAY ZEKA MÜHENDİSLİĞİ"),
    _week2_marker(2, "week2_start", "02 Alperen, İkinci Haftaya Başlarken"),
    _week2_marker(3, "engineering_approach", "03 Çalışmanın Amacı ve Mühendislik Yaklaşımı"),
    _week2_marker(4, "system_architecture", "04 Hedef Sistem Mimarisi"),
    _week2_marker(5, "query_sequence", "05 Sorgu Akışı - Sequence Diagram"),
    _week2_marker(6, "project_structure", "06 Proje Yapısı ve Bağımlılık Yönü"),
    _week2_marker(7, "fastapi_skeleton", "07 FastAPI Uygulama İskeleti"),
    _week2_marker(8, "rest_api", "08 REST API Tasarımı"),
    _week2_marker(9, "response_models", "09 Response Modelleri ve Hata Taksonomisi"),
    _week2_marker(10, "indexing_pipeline", "10 İndeksleme Pipeline'ı ve Sürümleme"),
    _week2_marker(11, "qdrant_schema", "11 Qdrant Collection ve Point Şeması"),
    _week2_marker(12, "metadata_filters", "12 Metadata ve Filtre Stratejisi"),
    _week2_marker(13, "hybrid_search", "13 Hybrid Search Tasarımı"),
    _week2_marker(14, "retrieval_comparison", "14 BM25 ve Dense Retrieval Karşılaştırması"),
    _week2_marker(15, "reranker_evaluation", "15 Reranker Değerlendirme Metodolojisi"),
    _week2_marker(16, "benchmark_reproducibility", "16 Benchmark Tasarımı ve Yeniden Üretilebilirlik"),
    _week2_marker(17, "evaluation_dataset", "17 Evaluation Dataset ve Etiketleme Rehberi"),
    _week2_marker(18, "answerability", "18 Answerability, No-Answer ve Kaynaklı Üretim"),
    _week2_marker(19, "prompt_security", "19 Prompt Güvenliği ve Doküman Tehdit Modeli"),
    _week2_marker(20, "observability", "20 Observability, Structured Logging ve Trace Tasarımı"),
    _week2_marker(21, "compose", "21 Docker Compose Mimarisi"),
    _week2_marker(22, "ci_cd", "22 CI/CD Önerileri ve Kodlama Standartları"),
    _week2_marker(23, "schedule", "23 5 Günlük Uygulama Takvimi"),
    _week2_marker(24, "deliverables", "24 Teslim Paketi ve Kabul Kriterleri"),
    _week2_marker(25, "review_demo", "25 Teknik Review Checklist ve Demo Senaryosu"),
    _week2_marker(26, "rubric", "26 Mentor Değerlendirme Rubriği"),
    _week2_marker(27, "interview", "27 Mentor Teknik Görüşme Soruları"),
    _week2_marker(28, "appendix", "28 Ekler - Konfigürasyon, ADR ve Kaynaklar"),
)


def get_section_markers(
    profile: SectionMarkerProfile | str,
) -> tuple[SectionMarker, ...]:
    """Return immutable markers for one explicit profile."""

    if profile in {"auto", "generic_v1", "none"}:
        return ()
    if profile == "mentor_program_v1":
        return MENTOR_PROGRAM_V1_MARKERS
    if profile == "mentor_program_week2_v1":
        return MENTOR_PROGRAM_WEEK2_V1_MARKERS
    raise ValueError(f"Unknown section marker profile: {profile}")


class KnownSectionMarkerProfileResolver:
    """Detect only explicit, reliable known headings and otherwise fall back."""

    _AUTO_CANDIDATES = (
        "mentor_program_week2_v1",
        "mentor_program_v1",
    )

    def __init__(self, extractor: PageTextExtractor) -> None:
        self._extractor = extractor

    def resolve(
        self,
        content: bytes,
        requested_profile: str,
    ) -> ChunkingResolution:
        """Resolve AUTO without making arbitrary PDFs conform to a mentor profile."""

        if requested_profile in {"none", "generic_v1"}:
            return ChunkingResolution(
                requested_profile=requested_profile,
                resolved_profile=GENERIC_PROFILE,
                detection_method="explicit_generic_profile",
            )
        if requested_profile in {
            "mentor_program_v1",
            "mentor_program_week2_v1",
        }:
            # Strict validation remains in sectionize_pages during the actual
            # Chunk stage.  Resolving identity here must not silently weaken
            # an explicitly requested reproducibility profile.
            return ChunkingResolution(
                requested_profile=requested_profile,
                resolved_profile=requested_profile,
                detection_method="explicit_profile",
            )
        if requested_profile != "auto":
            raise ValueError(f"Unknown section marker profile: {requested_profile}")

        try:
            pages = self._extractor.extract(content)
        except ServiceError:
            # The real parse stage remains the owner of parse errors.  AUTO
            # detection is advisory and must not turn a valid upload into a
            # rejection before its normal pipeline timeline exists.
            return ChunkingResolution(
                requested_profile="auto",
                resolved_profile=GENERIC_PROFILE,
                detection_method="known_section_markers_v1",
                confidence="unknown",
                fallback_reason=(
                    "Structure detection unavailable; generic_v1 fallback selected."
                ),
            )

        joined_text = "\n".join(page.text for page in pages if page.text.strip())
        for candidate in self._AUTO_CANDIDATES:
            markers = get_section_markers(candidate)
            if _markers_are_present_in_order(joined_text, markers):
                return ChunkingResolution(
                    requested_profile="auto",
                    resolved_profile=candidate,
                    detection_method="known_section_markers_v1",
                    confidence="high",
                )

        return ChunkingResolution(
            requested_profile="auto",
            resolved_profile=GENERIC_PROFILE,
            detection_method="known_section_markers_v1",
            confidence="low",
            fallback_reason=(
                "No reliable structured section markers detected; "
                "generic_v1 fallback selected."
            ),
        )


def _markers_are_present_in_order(
    text: str,
    markers: tuple[SectionMarker, ...],
) -> bool:
    """Return true only when the complete known marker contract is present."""

    if not text or not markers:
        return False
    positions: list[int] = []
    for marker in markers:
        position = text.find(marker.marker)
        if position < 0:
            return False
        positions.append(position)
    return positions == sorted(positions)
