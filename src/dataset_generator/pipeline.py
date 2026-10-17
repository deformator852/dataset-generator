"""Конвейер обработки файла: текст → язык и тип датасета → фрагменты → модель → датасет."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from dataset_generator.backends import (
    Backend,
    FatalBackendError,
    GenerationError,
    OllamaBackend,
    Structured,
)
from dataset_generator.chunking import split_into_chunks, strip_references
from dataset_generator.config import Settings
from dataset_generator.dataset import DatasetWriter, base_fingerprint, file_sha256, find_auto_kind
from dataset_generator.estimate import (
    estimate_api,
    estimate_chunks,
    measure_speed,
    speed_cached,
    system_info,
)
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
    SourceAnalysis,
    analyze_source,
    generate_examples,
    generation_request,
)
from dataset_generator.pricing import Usage, cost, price_for

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


async def process_file(job: Job, settings: Settings, backend: Backend) -> AsyncIterator[Event]:
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
        spent = 0.0
        for think in _attempt_modes(settings):
            try:
                async for event in analyze_source(
                    backend, source=path.stem, samples=_samples(chunks), think=think
                ):
                    if isinstance(event, Structured):
                        analysis = event.value  # type: ignore[assignment]
                        spent += _cost(event.usage, settings)
                    else:
                        yield event
                break
            except GenerationError as exc:
                error = str(exc)
                spent += _cost(exc.usage, settings)
            except FatalBackendError as exc:
                yield FileFailed(path, str(exc))
                return
        if analysis:
            run = analysis.kind
            yield KindChosen(
                path, run, run_title(run), "auto", analysis.genre, analysis.reason, spent
            )
        else:
            run = _FALLBACK_KIND
            yield KindChosen(path, run, run_title(run), "fallback", reason=error, cost=spent)

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
            async for event in _estimate(path, run, language, chunks, remaining, settings, backend):
                yield event
        except FatalBackendError as exc:
            yield FileFailed(path, str(exc))
            return

    async for event in _process_chunks(
        path, run, language, chunks, remaining, writer, settings, backend
    ):
        yield event


async def _process_chunks(
    path: Path,
    run: str,
    language: Language,
    chunks: list[str],
    remaining: list[int],
    writer: DatasetWriter,
    settings: Settings,
    backend: Backend,
) -> AsyncIterator[Event]:
    """Обрабатывает фрагменты, до `parallel` одновременно; события идут через очередь.

    Каждый готовый фрагмент сразу записывается в датасет и отмечается в состоянии,
    поэтому прерванную обработку можно продолжить с того же места.
    """
    total = len(chunks)
    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    semaphore = asyncio.Semaphore(_parallel(settings))
    write_lock = asyncio.Lock()
    failed = 0
    spent = 0.0

    async def process(index: int) -> None:
        nonlocal failed, spent
        started = time.monotonic()
        generation = None
        modes = _attempt_modes(settings)
        for attempt, think in enumerate(modes, 1):
            await queue.put(ChunkStarted(index, total, think))
            try:
                async for event in generate_examples(
                    backend,
                    settings,
                    run,
                    language,
                    source=path.stem,
                    chunk=chunks[index],
                    index=index,
                    total=total,
                    think=think,
                ):
                    if isinstance(event, Generation):
                        generation = event
                    else:
                        await queue.put(event)
                break
            except GenerationError as exc:
                lost = _cost(exc.usage, settings)
                spent += lost
                if attempt < len(modes):
                    await queue.put(ChunkRetry(index, attempt, str(exc), lost))
                else:
                    await queue.put(ChunkFailed(index, str(exc), time.monotonic() - started, lost))

        if generation is None:
            failed += 1
            return
        price = _cost(generation.usage, settings)
        spent += price
        async with write_lock:
            written = await asyncio.to_thread(writer.add, index, generation.examples)
        await queue.put(
            ChunkFinished(
                index,
                total,
                generation.examples,
                written,
                generation.tokens,
                generation.seconds,
                time.monotonic() - started,
                price,
            )
        )

    async def worker(index: int) -> None:
        nonlocal failed
        async with semaphore:
            try:
                await process(index)
            except FatalBackendError as exc:
                await queue.put(FileFailed(path, str(exc)))
            except Exception as exc:  # ошибка одного фрагмента не должна ронять остальные
                failed += 1
                await queue.put(ChunkFailed(index, f"unexpected error: {exc!r}", 0.0))

    tasks = [asyncio.create_task(worker(index)) for index in remaining]

    async def close_queue() -> None:
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.put(None)

    closer = asyncio.create_task(close_queue())
    try:
        while (event := await queue.get()) is not None:
            yield event
            if isinstance(event, FileFailed):
                return
    finally:
        for task in tasks:
            task.cancel()
        closer.cancel()

    yield FileFinished(path, writer.output_label, dict(writer.records), failed, spent)


async def _estimate(
    path: Path,
    run: str,
    language: Language,
    chunks: list[str],
    remaining: list[int],
    settings: Settings,
    backend: Backend,
) -> AsyncIterator[Event]:
    """Прогноз времени (и стоимости для API) для оставшихся фрагментов.

    Локально — по замеру скорости модели (раз за сеанс), для API — по тарифам и
    типичной скорости; в обоих случаях ETA потом уточняется по факту.
    """
    max_items = max_items_for(run, settings.max_items)
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
    left = [chunks[index] for index in remaining]
    output_chars = expected_output_chars(run, max_items)

    if isinstance(backend, OllamaBackend):
        # Замер — это запрос первого оставшегося фрагмента: тот же промпт и та же схема.
        fresh = not speed_cached(settings)
        if fresh:
            yield SpeedMeasureStarted(path)
        speed = await measure_speed(backend.client, settings, messages, schema)
        if fresh:
            yield SpeedMeasured(speed, system_info())
        estimate = estimate_chunks(
            left,
            speed,
            output_chars=output_chars,
            think=settings.think,
            system_chars=len(messages[0]["content"]),
        )
        yield EstimateReady(path, estimate)
        return

    schema_chars = len(json.dumps(schema.model_json_schema(), separators=(",", ":")))
    estimate, price = estimate_api(
        left,
        output_chars=output_chars,
        think=settings.think,
        effort=settings.effort,
        system_chars=len(messages[0]["content"]) + schema_chars,
        parallel=_parallel(settings),
        model=settings.model,
    )
    known = price_for(settings.model) is not None
    yield EstimateReady(path, estimate, _parallel(settings), price if known else None)


def _parallel(settings: Settings) -> int:
    # Локальная Ollama считает запросы по очереди — параллельность ей не помогает.
    return 1 if settings.local else max(settings.parallel, 1)


def _cost(usage: Usage | None, settings: Settings) -> float:
    return cost(usage, settings.model) if usage else 0.0


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
