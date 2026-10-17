"""Конвейер обработки файла: текст → язык и тип датасета → фрагменты → модель → датасет."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from ollama import AsyncClient

from dataset_generator.chunking import split_into_chunks, strip_references
from dataset_generator.config import Settings
from dataset_generator.dataset import DatasetWriter, base_fingerprint, file_sha256, find_auto_kind
from dataset_generator.estimate import estimate_chunks, measure_speed, speed_cached, system_info
from dataset_generator.events import (
    AnalysisStarted,
    ChunkFailed,
    ChunkFinished,
    ChunkRetry,
    ChunkStarted,
    EstimateReady,
    Event,
    FileFailed,
    FileFinished,
    FileLoaded,
    FileStarted,
    KindChosen,
    SpeedMeasured,
    SpeedMeasureStarted,
    TextExtracted,
)
from dataset_generator.extract import ExtractionError, extract_text, is_supported
from dataset_generator.kinds import AUTO, expected_output_chars, max_items_for, run_title
from dataset_generator.language import Language, detect_language, language_from_code
from dataset_generator.llm import (
    Generation,
    GenerationError,
    SourceAnalysis,
    Structured,
    analyze_source,
    generate_examples,
    generation_request,
)

_FALLBACK_KIND = "qa"
_ANALYSIS_SAMPLES = 4
_ANALYSIS_SAMPLE_CHARS = 1500


@dataclass(slots=True, frozen=True)
class Job:
    path: Path
    kind: str  # ключ из KINDS, AUTO (тип на весь документ) или MIXED (типы по фрагментам)
    dataset_format: str


def resolve_input(raw: str, settings: Settings) -> list[Path]:
    """Превращает введённое имя/путь в список файлов.

    Ищет сначала относительно текущей папки, затем в папке входных файлов.
    Для папки возвращает все поддерживаемые файлы внутри неё.
    """
    name = raw.strip().strip("'\"")
    if not name:
        return []
    candidate = Path(name).expanduser()
    for path in (candidate, settings.input_dir / candidate):
        if path.is_file():
            return [path]
        if path.is_dir():
            return sorted(p for p in path.rglob("*") if p.is_file() and is_supported(p))
    return []


async def process_file(job: Job, settings: Settings, client: AsyncClient) -> AsyncIterator[Event]:
    path = job.path
    yield FileStarted(path)
    try:
        text = await asyncio.to_thread(extract_text, path)
        sha256 = await asyncio.to_thread(file_sha256, path)
    except (ExtractionError, OSError) as exc:
        yield FileFailed(path, str(exc))
        return

    text, references = strip_references(text)
    chunks = split_into_chunks(text, settings.chunk_size, settings.chunk_overlap)
    total = len(chunks)
    detected = settings.language == "auto"
    language = (
        await asyncio.to_thread(detect_language, text)
        if detected
        else language_from_code(settings.language)
    )
    yield TextExtracted(path, len(text), total, language.label, detected, references)
    fingerprint = base_fingerprint(sha256, settings, job.dataset_format, language)

    run = job.kind
    if job.kind != AUTO:
        yield KindChosen(path, run, run_title(run), "user")
    elif previous := find_auto_kind(path, settings, fingerprint):
        run = previous
        yield KindChosen(path, run, run_title(run), "resumed")
    else:
        yield AnalysisStarted(path)
        analysis: SourceAnalysis | None = None
        error = ""
        for think in _attempt_modes(settings):
            try:
                async for event in analyze_source(
                    client,
                    settings,
                    source=path.stem,
                    samples=_samples(chunks),
                    think=think,
                ):
                    if isinstance(event, Structured):
                        analysis = event.value  # type: ignore[assignment]
                    else:
                        yield event
                break
            except GenerationError as exc:
                error = str(exc)
            except ConnectionError as exc:
                yield FileFailed(path, f"lost connection to Ollama: {exc}")
                return
        if analysis:
            run = analysis.kind
            yield KindChosen(path, run, run_title(run), "auto", analysis.genre, analysis.reason)
        else:
            run = _FALLBACK_KIND
            yield KindChosen(path, run, run_title(run), "fallback", reason=error)

    writer = await asyncio.to_thread(
        DatasetWriter,
        path,
        settings,
        run=run,
        auto=job.kind == AUTO,
        dataset_format=job.dataset_format,
        language=language,
        fingerprint=fingerprint,
    )
    yield FileLoaded(path, writer.output_label, total, len(writer.done))

    remaining = [index for index in range(total) if index not in writer.done]
    if remaining:
        try:
            async for event in _estimate(path, run, language, chunks, remaining, settings, client):
                yield event
        except ConnectionError as exc:
            yield FileFailed(path, f"lost connection to Ollama: {exc}")
            return

    failed = 0
    for index, chunk in enumerate(chunks):
        if index in writer.done:
            continue
        generation = None
        started = time.monotonic()
        modes = _attempt_modes(settings)
        for attempt, think in enumerate(modes, 1):
            yield ChunkStarted(index, total, think)
            try:
                async for event in generate_examples(
                    client,
                    settings,
                    run,
                    language,
                    source=path.stem,
                    chunk=chunk,
                    index=index,
                    total=total,
                    think=think,
                ):
                    if isinstance(event, Generation):
                        generation = event
                    else:
                        yield event
                break
            except GenerationError as exc:
                if attempt < len(modes):
                    yield ChunkRetry(index, attempt, str(exc))
                else:
                    yield ChunkFailed(index, str(exc), time.monotonic() - started)
            except ConnectionError as exc:
                yield FileFailed(path, f"lost connection to Ollama: {exc}")
                return

        if generation is None:
            failed += 1
            continue
        examples = generation.examples
        written = await asyncio.to_thread(writer.add, index, examples)
        yield ChunkFinished(
            index,
            total,
            examples,
            written,
            generation.tokens,
            generation.seconds,
            time.monotonic() - started,
        )

    yield FileFinished(path, writer.output_label, dict(writer.records), failed)


async def _estimate(
    path: Path,
    run: str,
    language: Language,
    chunks: list[str],
    remaining: list[int],
    settings: Settings,
    client: AsyncClient,
) -> AsyncIterator[Event]:
    """Замер скорости модели (раз за сеанс) и прогноз времени для оставшихся фрагментов."""
    max_items = max_items_for(run, settings.max_items)
    # Замер — это запрос первого оставшегося фрагмента: тот же промпт и та же схема.
    first = remaining[0]
    messages, schema = generation_request(
        run,
        language,
        source=path.stem,
        chunk=chunks[first],
        index=first,
        total=len(chunks),
        max_items=max_items,
    )
    fresh = not speed_cached(settings)
    if fresh:
        yield SpeedMeasureStarted(path)
    speed = await measure_speed(client, settings, messages, schema)
    if fresh:
        yield SpeedMeasured(speed, system_info())
    estimate = estimate_chunks(
        [chunks[index] for index in remaining],
        speed,
        output_chars=expected_output_chars(run, max_items),
        think=settings.think,
        system_chars=len(messages[0]["content"]),
    )
    yield EstimateReady(path, estimate)


def _attempt_modes(settings: Settings) -> list[bool]:
    """Режим мышления для каждой попытки.

    Последнюю попытку делаем без мышления: чаще всего срыв — это слишком длинные
    размышления, а без них модель отвечает сразу. Вызывается заново для каждого
    фрагмента, чтобы переключение мышления в TUI применялось со следующего.
    """
    attempts = settings.retries + 1
    return [settings.think and (n < attempts or attempts == 1) for n in range(1, attempts + 1)]


def _samples(chunks: list[str]) -> list[str]:
    """Фрагменты из начала, середины и конца — для определения жанра источника.

    У короткого документа (статьи на пару фрагментов) берётся весь текст.
    """
    if len(chunks) <= _ANALYSIS_SAMPLES:
        picked = chunks
    else:
        step = (len(chunks) - 1) / (_ANALYSIS_SAMPLES - 1)
        picked = [chunks[round(i * step)] for i in range(_ANALYSIS_SAMPLES)]
    return [chunk[:_ANALYSIS_SAMPLE_CHARS] for chunk in picked]
