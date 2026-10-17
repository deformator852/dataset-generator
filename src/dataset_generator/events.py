"""События конвейера — их одинаково потребляют TUI и консольный режим."""

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
    detected: bool  # язык определён автоматически, а не задан вручную
    references_removed: int  # сколько записей списков литературы вырезано


@dataclass(slots=True, frozen=True)
class AnalysisStarted:
    path: Path


@dataclass(slots=True, frozen=True)
class KindChosen:
    path: Path
    kind: str
    title: str
    # user — выбран пользователем, auto — определён моделью, resumed — взят из прошлого
    # авто-анализа этого файла, fallback — анализ не удался, взят тип по умолчанию
    how: Literal["user", "auto", "resumed", "fallback"]
    genre: str = ""
    reason: str = ""
    cost: float = 0.0  # стоимость анализа, $


@dataclass(slots=True, frozen=True)
class FileLoaded:
    path: Path
    output: str  # путь к датасету или шаблон путей для режима «по фрагментам»
    total_chunks: int
    already_done: int


@dataclass(slots=True, frozen=True)
class EstimateReady:
    path: Path
    estimate: Estimate
    parallel: int = 1
    cost: float | None = None  # ожидаемая стоимость оставшихся фрагментов, $


@dataclass(slots=True, frozen=True)
class ChunkStarted:
    index: int
    total: int
    think: bool


@dataclass(slots=True, frozen=True)
class ThinkingDelta:
    text: str
    index: int = -1  # номер фрагмента; -1 — анализ источника


@dataclass(slots=True, frozen=True)
class ContentDelta:
    text: str
    index: int = -1


@dataclass(slots=True, frozen=True)
class ChunkRetry:
    index: int
    attempt: int
    reason: str
    cost: float = 0.0  # неудачная попытка тоже оплачена


@dataclass(slots=True, frozen=True)
class ChunkFinished:
    index: int
    total: int
    examples: list[Example]
    written: int
    tokens: int
    seconds: float  # последняя (удачная) попытка
    elapsed: float  # все попытки вместе — для уточнения оставшегося времени
    cost: float = 0.0  # $, последняя попытка


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
    records: dict[str, int]  # тип датасета → число примеров
    failed_chunks: int
    cost: float = 0.0  # $ за этот запуск


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
