"""Pipeline events, consumed the same way by the TUI and the console mode."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dataset_generator.estimate import Estimate
from dataset_generator.kinds import Example


@dataclass(slots=True, frozen=True)
class FileStarted:
    path: Path


@dataclass(slots=True, frozen=True)
class TextExtracted:
    path: Path
    chars: int
    total_chunks: int
    language: str
    detected: bool  # the language was detected automatically, not set manually
    references_removed: int  # how many reference list entries were removed


@dataclass(slots=True, frozen=True)
class AnalysisStarted:
    path: Path


@dataclass(slots=True, frozen=True)
class KindChosen:
    path: Path
    kind: str
    title: str
    # user: chosen by the user; auto: picked by the model; resumed: taken from a previous
    # auto analysis of this file; fallback: the analysis failed, the default type is used
    how: Literal["user", "auto", "resumed", "fallback"]
    genre: str = ""
    reason: str = ""
    cost: float = 0.0  # cost of the analysis, $


@dataclass(slots=True, frozen=True)
class FileLoaded:
    path: Path
    output: str  # dataset path, or a path pattern for "Auto — per fragment"
    total_chunks: int
    already_done: int


@dataclass(slots=True, frozen=True)
class EstimateReady:
    path: Path
    estimate: Estimate
    parallel: int = 1
    cost: float | None = None  # expected cost of the remaining fragments, $


@dataclass(slots=True, frozen=True)
class ChunkStarted:
    index: int
    total: int
    think: bool


@dataclass(slots=True, frozen=True)
class ThinkingDelta:
    text: str
    index: int = -1  # fragment index; -1 is the source analysis


@dataclass(slots=True, frozen=True)
class ContentDelta:
    text: str
    index: int = -1


@dataclass(slots=True, frozen=True)
class ChunkRetry:
    index: int
    attempt: int
    reason: str
    cost: float = 0.0  # a failed attempt is billed too


@dataclass(slots=True, frozen=True)
class ChunkFinished:
    index: int
    total: int
    examples: list[Example]
    written: int
    tokens: int
    seconds: float  # the last (successful) attempt
    elapsed: float  # all attempts together, used to refine the remaining time
    cost: float = 0.0  # $, the last attempt


@dataclass(slots=True, frozen=True)
class ChunkFailed:
    index: int
    reason: str
    elapsed: float
    cost: float = 0.0


@dataclass(slots=True, frozen=True)
class FileFinished:
    path: Path
    output: str
    records: dict[str, int]  # dataset type → number of examples
    failed_chunks: int
    cost: float = 0.0  # $ for this run


@dataclass(slots=True, frozen=True)
class FileFailed:
    path: Path
    reason: str


Event = (
    FileStarted
    | TextExtracted
    | AnalysisStarted
    | KindChosen
    | FileLoaded
    | EstimateReady
    | ChunkStarted
    | ThinkingDelta
    | ContentDelta
    | ChunkRetry
    | ChunkFinished
    | ChunkFailed
    | FileFinished
    | FileFailed
)
