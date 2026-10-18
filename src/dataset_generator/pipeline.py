"""File processing pipeline: text → language and dataset type → fragments → model → dataset."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from dataset_generator.backends import Backend, FatalBackendError, GenerationError, Structured
from dataset_generator.chunking import split_into_chunks, strip_references
from dataset_generator.config import Settings
from dataset_generator.dataset import DatasetWriter, base_fingerprint, file_sha256
from dataset_generator.estimate import estimate_api
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
    SourceAnalysed,
    TextExtracted,
)
from dataset_generator.extract import ExtractionError, extract_text, is_supported
from dataset_generator.kinds import AUTO, KINDS, expected_output_chars, max_items_for, run_title
from dataset_generator.language import Language, detect_language, language_from_code
from dataset_generator.llm import (
    Generation,
    SourceAnalysis,
    analyze_source,
    generate_examples,
    generation_request,
)
from dataset_generator.pricing import Usage, cost, price_for
from dataset_generator.quality import match_quote
from dataset_generator.source_card import (
    card_path,
    load_card,
    new_card,
    read_card,
    record_datasets,
    save_card,
    verified_share,
)

_FALLBACK_KIND = "qa"
_ANALYSIS_SAMPLES = 6
_ANALYSIS_SAMPLE_CHARS = 2500


@dataclass(slots=True, frozen=True)
class Job:
    path: Path
    kind: str  # a KINDS key, AUTO (one type per document) or MIXED (types per fragment)
    dataset_format: str


def resolve_input(raw: str, settings: Settings) -> list[Path]:
    """Turns an entered name or path into a list of files.

    Looks relative to the current folder first, then in the input folder.
    For a folder, returns all supported files inside it.
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

    analysed: SourceAnalysed | None = None
    async for event in _analyse_source(path, sha256, language, text, chunks, settings, backend):
        if isinstance(event, FileFailed):
            yield event
            return
        if isinstance(event, SourceAnalysed):
            analysed = event
        yield event

    run = job.kind
    if job.kind != AUTO:
        yield KindChosen(path, run, run_title(run), "user")
    elif analysed and analysed.kind:
        run = analysed.kind
        how = "resumed" if analysed.how == "reused" else "auto"
        yield KindChosen(path, run, run_title(run), how, analysed.genre, analysed.reason)
    else:
        run = _FALLBACK_KIND
        reason = analysed.reason if analysed else ""
        yield KindChosen(path, run, run_title(run), "fallback", reason=reason)

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
        async for event in _estimate(path, run, language, chunks, remaining, settings):
            yield event

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
    """Processes fragments, up to `parallel` at once; events go through a queue.

    Each finished fragment is written to the dataset and marked in the state right away,
    so interrupted processing can be resumed from the same place.
    """
    total = len(chunks)
    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    semaphore = asyncio.Semaphore(max(settings.parallel, 1))
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
        examples = [
            replace(example, quote_match=match_quote(example.quote, chunks[index]))
            for example in generation.examples
        ]
        async with write_lock:
            written = await asyncio.to_thread(writer.add, index, examples)
        await queue.put(
            ChunkFinished(
                index,
                total,
                examples,
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
            except Exception as exc:  # one fragment's error must not stop the others
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

    card = card_path(path, settings)
    stats = await asyncio.to_thread(record_datasets, card, writer.paths.values())
    yield FileFinished(
        path,
        writer.output_label,
        dict(writer.records),
        failed,
        spent,
        card,
        verified_share(stats),
    )


async def _analyse_source(
    path: Path,
    sha256: str,
    language: Language,
    text: str,
    chunks: list[str],
    settings: Settings,
    backend: Backend,
) -> AsyncIterator[Event]:
    """The source card: taken from a previous run of the same file or filled in by the model.

    Ends with SourceAnalysed, or with FileFailed if the API cannot be used at all.
    """
    card_file = card_path(path, settings)
    if card := await asyncio.to_thread(load_card, card_file, sha256):
        yield _analysed(path, card_file, "reused", card)
        return

    yield AnalysisStarted(path)
    samples = _samples(chunks)
    whole = len(chunks) <= _ANALYSIS_SAMPLES
    analysis: SourceAnalysis | None = None
    error = ""
    spent = 0.0
    for think in _attempt_modes(settings):
        try:
            async for event in analyze_source(
                backend, source=path.stem, samples=samples, whole=whole, think=think
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

    sampled = sum(len(sample) for sample in samples)
    card = new_card(
        path,
        sha256=sha256,
        language=language.code,
        chars=len(text),
        model=settings.model,
        based_on="the whole text"
        if whole
        else f"{len(samples)} fragments, {sampled} of {len(text)} characters",
        description=analysis.model_dump() if analysis else None,
        error="" if analysis else error,
        previous=await asyncio.to_thread(read_card, card_file),
    )
    await asyncio.to_thread(save_card, card_file, card)
    how = "new" if analysis else "failed"
    yield _analysed(path, card_file, how, card, cost=spent, error=error)


def _analysed(
    path: Path,
    card_file: Path,
    how: Literal["new", "reused", "failed"],
    card: dict[str, Any],
    *,
    cost: float = 0.0,
    error: str = "",
) -> SourceAnalysed:
    kind = card.get("recommended_kind", "")
    return SourceAnalysed(
        path,
        card_file,
        how,
        title=card.get("title", ""),
        genre=card.get("genre", ""),
        evidence=tuple(card.get("evidence_basis", ())),
        suitability=card.get("suitability", ""),
        kind=kind if kind in KINDS else "",
        reason=error if how == "failed" else card.get("kind_reason", ""),
        cost=cost,
    )


async def _estimate(
    path: Path,
    run: str,
    language: Language,
    chunks: list[str],
    remaining: list[int],
    settings: Settings,
) -> AsyncIterator[Event]:
    """Time and cost estimate for the remaining fragments (refined later in practice)."""
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
    schema_chars = len(json.dumps(schema.model_json_schema(), separators=(",", ":")))
    estimate, price = estimate_api(
        [chunks[index] for index in remaining],
        output_chars=expected_output_chars(run, max_items),
        think=settings.think,
        effort=settings.effort,
        system_chars=len(messages[0]["content"]) + schema_chars,
        parallel=settings.parallel,
        model=settings.model,
    )
    known = price_for(settings.model) is not None
    yield EstimateReady(path, estimate, settings.parallel, price if known else None)


def _cost(usage: Usage | None, settings: Settings) -> float:
    return cost(usage, settings.model) if usage else 0.0


def _attempt_modes(settings: Settings) -> list[bool]:
    """Thinking mode for each attempt.

    The last attempt runs without thinking: failures are most often caused by overly long
    thinking, and without it the model answers right away. Called anew for each fragment,
    so toggling thinking in the TUI applies from the next one.
    """
    attempts = settings.retries + 1
    return [settings.think and (n < attempts or attempts == 1) for n in range(1, attempts + 1)]


def _samples(chunks: list[str]) -> list[str]:
    """Fragments for the source card.

    A short document is used whole. Otherwise: the first two fragments (title, abstract,
    introduction, often the methods), then evenly spaced ones up to the last (results,
    discussion, conclusions), each cut to a fixed size.
    """
    if len(chunks) <= _ANALYSIS_SAMPLES:
        return chunks
    rest = _ANALYSIS_SAMPLES - 2
    last = len(chunks) - 1
    indices = [0, 1] + [2 + round(i * (last - 2) / (rest - 1)) for i in range(rest)]
    return [chunks[index][:_ANALYSIS_SAMPLE_CHARS] for index in sorted(set(indices))]
