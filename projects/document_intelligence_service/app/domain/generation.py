"""Framework-independent answer generation boundary objects."""

from dataclasses import dataclass


class AnswerGenerationError(RuntimeError):
    """Raised when the configured local generation dependency cannot answer."""

    def __init__(self, message: str, *, reason_code: str = "GENERATION_FAILED") -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """One answer returned by an external/local generator adapter."""

    answer: str
    provider: str
    model: str
    latency_ms: float
