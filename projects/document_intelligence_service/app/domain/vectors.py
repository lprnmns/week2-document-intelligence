"""Framework-independent vector value objects."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse vector represented by sorted non-zero index/value pairs."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse indices and values must have equal lengths")
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse indices must be non-negative")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("sparse values must be finite")
        if tuple(sorted(self.indices)) != self.indices:
            raise ValueError("sparse indices must be sorted")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("sparse indices must be unique")
