"""Простой потоковый вывод в терминал без TUI (для фоновых запусков и логов)."""

import sys
from pathlib import Path

from ollama import AsyncClient
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from dataset_generator.config import Settings
from dataset_generator.estimate import Eta, format_duration
from dataset_generator.events import (
    AnalysisStarted,
    ChunkFailed,
    ChunkFinished,
    ChunkRetry,
    ChunkStarted,
    ContentDelta,
    EstimateReady,
    FileFailed,
    FileFinished,
    FileLoaded,
    FileStarted,
    KindChosen,
    SpeedMeasured,
    SpeedMeasureStarted,
    TextExtracted,
    ThinkingDelta,
)
from dataset_generator.kind_picker import describe_example
from dataset_generator.kinds import (
    AUTO,
    AUTO_ICON,
    AUTO_SUMMARY,
    AUTO_TITLE,
    KINDS,
    MIXED,
    MIXED_ICON,
    MIXED_SUMMARY,
    MIXED_TITLE,
    run_title,
)
from dataset_generator.llm import ModelUnavailableError, check_model
from dataset_generator.pipeline import Job, process_file


def ask_kind(console: Console) -> str:
    """Меню выбора типа датасета с примерами; без терминала — авто по фрагментам."""
    if not sys.stdin.isatty():
        return MIXED
    console.rule("[bold]Which dataset should be built?")
    autos = [
        (MIXED, MIXED_ICON, MIXED_TITLE, MIXED_SUMMARY),
        (AUTO, AUTO_ICON, AUTO_TITLE, AUTO_SUMMARY),
    ]
    for number, (_, icon, title, summary) in enumerate(autos):
        console.print(Text.assemble((f"{number}. ", "bold"), (f"{icon} {title}", "bold magenta")))
        console.print(f"   {summary}", style="dim")
    for number, kind in enumerate(KINDS.values(), len(autos)):
        console.print(
            Text.assemble((f"\n{number}. ", "bold"), (f"{kind.icon} {kind.title}", "bold"))
        )
        console.print(f"   {kind.summary}. Best for: {kind.suits}", style="dim")
        console.print(describe_example(kind.sample))
    keys = [key for key, *_ in autos] + list(KINDS)
    choices = [str(n) for n in range(len(keys))]
    answer = Prompt.ask("Type number", choices=choices, default="0", console=console)
    return keys[int(answer)]


async def run_console(settings: Settings, files: list[Path]) -> int:
    console = Console(highlight=False)
    kind = settings.dataset_kind or ask_kind(console)
    client = AsyncClient(host=settings.host)
    try:
        for warning in await check_model(client, settings):
            console.print(f"! {warning}", style="yellow", markup=False)
    except ModelUnavailableError as exc:
        console.print(f"✗ {exc}", style="bold red", markup=False)
        return 1

    def stream(text: str, style: str) -> None:
        console.print(text, style=style, end="", markup=False, soft_wrap=True)

    def line(text: str, style: str = "") -> None:
        console.print(text, style=style, markup=False, soft_wrap=True)

    exit_code = 0
    answering = False
    eta: Eta | None = None

    def progress_line(elapsed: float) -> str:
        if not eta:
            return ""
        eta.chunk_done(elapsed)
        return f", ≈ {format_duration(eta.seconds_left())} left" if eta.remaining else ""

    for path in files:
        async for event in process_file(Job(path, kind, settings.dataset_format), settings, client):
            match event:
                case ThinkingDelta():
                    stream(event.text, "italic #c9b37e")
                case ContentDelta():
                    if not answering:
                        answering = True
                        console.print("\n📝 ", end="")
                    stream(event.text, "dim")
                case FileStarted():
                    console.rule(f"[bold]{path.name}")
                case TextExtracted():
                    how = "auto" if event.detected else "manual"
                    refs = (
                        f" Removed reference list entries: {event.references_removed}."
                        if event.references_removed
                        else ""
                    )
                    line(
                        f"{event.chars} characters → {event.total_chunks} fragments. "
                        f"Language: {event.language} ({how}).{refs}",
                        "dim",
                    )
                case AnalysisStarted():
                    answering = False
                    line("\n🔎 Analysing the source to pick a dataset type", "bold")
                    console.print("🧠 ", end="")
                case KindChosen():
                    details = f" (source: {event.genre}). {event.reason}" if event.genre else ""
                    if event.how == "fallback":
                        details = f" — analysis failed: {event.reason}"
                    line(f"\nDataset type: {event.title}{details}", "bold magenta")
                case FileLoaded():
                    line(f"Already done: {event.already_done}. Dataset: {event.output}", "dim")
                    eta = None
                case SpeedMeasureStarted():
                    line("⏱ Measuring model speed on this PC…", "dim")
                case SpeedMeasured():
                    speed = event.speed
                    basis = "measured" if speed.measured else "rough hardware estimate"
                    line(f"💻 {event.system.label}", "dim")
                    line(
                        f"⏱ Reading {speed.prompt_tps:.0f} tok/s, generation {speed.gen_tps:.1f} "
                        f"tok/s ({basis}), {speed.device}",
                        "dim",
                    )
                case EstimateReady():
                    estimate = event.estimate
                    eta = Eta(estimate.chunks, estimate.per_chunk)
                    line(
                        f"⏱ Time estimate: {estimate.label} — {estimate.chunks} fragments, "
                        f"~{format_duration(estimate.per_chunk)} per fragment",
                        "bold yellow",
                    )
                case ChunkStarted():
                    answering = False
                    mode = "" if event.think else " (no thinking)"
                    line(f"\n── Fragment {event.index + 1}/{event.total}{mode} ──", "bold")
                    if event.think:
                        console.print("🧠 ", end="")
                case ChunkRetry():
                    line(f"\n! {event.reason} — retrying", "yellow")
                case ChunkFinished():
                    tps = event.tokens / event.seconds if event.seconds else 0
                    kinds = ""
                    if kind == MIXED and event.examples:
                        kinds = (
                            " ["
                            + ", ".join(
                                sorted({run_title(example.kind) for example in event.examples})
                            )
                            + "]"
                        )
                    line(
                        f"\n✓ +{event.written} examples{kinds} in "
                        f"{format_duration(event.elapsed)} ({tps:.1f} tok/s)"
                        f"{progress_line(event.elapsed)}",
                        "green",
                    )
                case ChunkFailed():
                    line(
                        f"\n✗ fragment skipped: {event.reason}{progress_line(event.elapsed)}",
                        "red",
                    )
                case FileFinished():
                    total = sum(event.records.values())
                    parts = [f"{run_title(k)}: {n}" for k, n in event.records.items() if n]
                    by_kind = f" ({', '.join(parts)})" if len(event.records) > 1 and parts else ""
                    line(f"✔ {total} examples{by_kind} → {event.output}", "bold green")
                case FileFailed():
                    line(f"✗ {event.path.name}: {event.reason}", "bold red")
                    exit_code = 1
    return exit_code
