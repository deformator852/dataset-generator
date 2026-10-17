"""TUI на Textual: очередь файлов, живые размышления модели, ответы и журнал."""

import time
from collections import deque
from pathlib import Path
from typing import ClassVar

from ollama import AsyncClient
from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.suggester import Suggester
from textual.widgets import Footer, Header, Input, ProgressBar, RichLog, Static
from textual.worker import Worker

from dataset_generator.config import Settings
from dataset_generator.estimate import Eta, Speed, SystemInfo, format_duration
from dataset_generator.events import (
    AnalysisStarted,
    ChunkFailed,
    ChunkFinished,
    ChunkRetry,
    ChunkStarted,
    ContentDelta,
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
    ThinkingDelta,
)
from dataset_generator.extract import is_supported
from dataset_generator.kind_picker import KindPicker, describe_example
from dataset_generator.kinds import MIXED, run_title
from dataset_generator.llm import ModelUnavailableError, check_model
from dataset_generator.pipeline import Job, process_file, resolve_input


class StreamView(VerticalScroll):
    """Прокручиваемая панель с переносом строк для потокового вывода модели.

    Токены копятся в буфере и перерисовываются по таймеру, а не на каждый токен.
    """

    DEFAULT_CSS = """
    StreamView > Static { width: 1fr; }
    """

    def __init__(self, *, text_style: str = "", max_chars: int = 60_000, id: str | None = None):
        super().__init__(id=id)
        self._text_style = text_style
        self._max_chars = max_chars
        self._buffer = ""
        self._dirty = False
        self._body = Static()

    def compose(self) -> ComposeResult:
        yield self._body

    def on_mount(self) -> None:
        self.set_interval(1 / 15, self._flush)

    def append(self, text: str) -> None:
        self._buffer = (self._buffer + text)[-self._max_chars :]
        self._dirty = True

    def reset(self, text: str = "") -> None:
        self._buffer = text
        self._dirty = True

    def _flush(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        follow = self.is_vertical_scroll_end
        self._body.update(Text(self._buffer, style=self._text_style))
        if follow:
            self.call_after_refresh(self.scroll_end, animate=False)


class PathSuggester(Suggester):
    """Автодополнение путей (принимается клавишей →); ищет и в папке входных файлов."""

    def __init__(self, input_dir: Path) -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self.input_dir = input_dir

    async def get_suggestion(self, value: str) -> str | None:
        if not value or value.startswith(("'", '"')):
            return None
        head, sep, prefix = value.rpartition("/")
        dir_part = head + sep
        roots = [Path(dir_part or ".").expanduser()]
        if not dir_part.startswith(("/", "~")):
            roots.append(self.input_dir / dir_part)
        for root in roots:
            try:
                entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith(".") and not prefix.startswith("."):
                    continue
                if entry.name.startswith(prefix) and entry.name != prefix:
                    if entry.is_dir():
                        return f"{dir_part}{entry.name}/"
                    if is_supported(entry):
                        return f"{dir_part}{entry.name}"
        return None


class DatasetApp(App[None]):
    TITLE = "Dataset Generator"
    SUB_TITLE = "forensic science datasets · Ollama"

    CSS = """
    #path { margin: 0 1; }
    #body { height: 1fr; }
    #sidebar { width: 46; padding: 0 1; }
    #info, #queue, #stats, #hardware { height: auto; border: round $primary; padding: 0 1; }
    #info { border-title-color: $primary; }
    #progress { margin: 1 0; }
    #progress Bar { width: 22; }
    #main { width: 1fr; }
    #thinking { height: 2fr; border: round $warning; padding: 0 1; }
    #answer { height: 1fr; border: round $success; padding: 0 1; }
    #journal { height: 1fr; border: round $accent; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("f2", "toggle_think", "Thinking on/off"),
        Binding("f3", "stop", "Stop"),
        Binding("f4", "clear", "Clear"),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self, settings: Settings, files: list[Path]) -> None:
        super().__init__()
        self.settings = settings
        self.initial_files = files
        self.queue: deque[Job] = deque()
        self.current: Job | None = None
        self.current_details = ""
        self.client: AsyncClient | None = None
        self.model_ready = False
        self.worker: Worker[None] | None = None
        self.chunk_label = "—"
        self.chunk_started_at: float | None = None
        self.examples_written = 0
        self.chunks_done = 0
        self.chunks_empty = 0
        self.chunks_failed = 0
        self.last_speed: float | None = None
        self.system: SystemInfo | None = None
        self.speed: Speed | None = None
        self.eta: Eta | None = None
        self.estimate_label = "—"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(
            placeholder=(
                "Add another file or folder to the queue (optional — the file from the command "
                "line is already queued) — Enter"
                if self.initial_files
                else "File name or path (pdf, epub, fb2, mobi, djvu, docx, txt…) — Enter"
            ),
            suggester=PathSuggester(self.settings.input_dir),
            id="path",
        )
        with Horizontal(id="body"):
            with VerticalScroll(id="sidebar"):
                yield Static(id="info")
                yield Static(id="queue")
                yield ProgressBar(id="progress", show_eta=False)
                yield Static(id="stats")
                yield Static(id="hardware")
            with Vertical(id="main"):
                yield StreamView(id="thinking", text_style="italic #c9b37e")
                yield StreamView(id="answer")
                yield RichLog(id="journal", wrap=True, max_lines=5000)
        yield Footer()

    def on_mount(self) -> None:
        self.client = AsyncClient(host=self.settings.host)
        self.query_one("#info").border_title = "Settings"
        self.query_one("#queue").border_title = "Queue"
        self.query_one("#stats").border_title = "Progress and time"
        self.query_one("#hardware").border_title = "PC and model speed"
        self.query_one("#thinking").border_title = "🧠 Model thinking"
        self.query_one("#answer").border_title = "📝 Model response (JSON)"
        self.query_one("#journal").border_title = "📚 Dataset and log"
        self.refresh_sidebar()
        self.refresh_hardware()
        self.set_interval(1, self.refresh_stats)
        self.query_one(Input).focus()
        self.log_line(
            f'Enter a file name from "{self.settings.input_dir}/" or a path to a file/folder '
            "and press Enter. → accepts autocompletion, an empty Enter restarts the queue.",
            "dim",
        )
        if self.initial_files:
            self.request_jobs(self.initial_files)

    @property
    def journal(self) -> RichLog:
        return self.query_one("#journal", RichLog)

    def log_line(self, message: str | Text, style: str = "") -> None:
        self.journal.write(Text(message, style=style) if isinstance(message, str) else message)

    # --- ввод и очередь ---------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.clear()
        if not value:
            self.start_processing()
            return
        files = resolve_input(value, self.settings)
        if not files:
            self.log_line(f"✗ File or folder not found: {value}", "bold red")
            return
        supported = [f for f in files if is_supported(f)]
        for path in files:
            if path not in supported:
                self.log_line(
                    f"✗ Unsupported format {path.suffix or '(no extension)'}: {path.name}",
                    "red",
                )
        if supported:
            self.request_jobs(supported)

    def request_jobs(self, files: list[Path]) -> None:
        """Ставит файлы в очередь; если тип датасета не задан флагом — спрашивает."""
        busy = {job.path.resolve() for job in self.queue}
        if self.current:
            busy.add(self.current.path.resolve())
        for path in files:
            if path.resolve() in busy:
                self.log_line(f"· {path.name} is already queued — not adding it twice", "dim")
        files = [path for path in files if path.resolve() not in busy]
        if not files:
            return
        if self.settings.dataset_kind:
            self.enqueue(files, self.settings.dataset_kind, self.settings.dataset_format)
            return

        def on_choice(choice: tuple[str, str] | None) -> None:
            if choice is None:
                self.log_line("Cancelled — no files added.", "dim")
            else:
                self.enqueue(files, *choice)

        self.push_screen(KindPicker(files, self.settings), on_choice)

    def enqueue(self, files: list[Path], kind: str, dataset_format: str) -> None:
        self.queue.extend(Job(path, kind, dataset_format) for path in files)
        names = ", ".join(p.name for p in files[:5])
        more = f" and {len(files) - 5} more" if len(files) > 5 else ""
        self.log_line(f"+ Queued: {names}{more} · {run_title(kind)} · {dataset_format}", "cyan")
        self.refresh_sidebar()
        self.start_processing()

    def start_processing(self) -> None:
        if self.worker is not None and self.worker.is_running:
            return
        if self.queue:
            self.worker = self.run_worker(self.process_queue(), group="pipeline")

    async def process_queue(self) -> None:
        assert self.client is not None
        if not self.model_ready:
            try:
                warnings = await check_model(self.client, self.settings)
            except ModelUnavailableError as exc:
                self.log_line(f"✗ {exc}", "bold red")
                self.log_line("Once fixed, press Enter in the empty input field.", "dim")
                return
            for warning in warnings:
                self.log_line(f"! {warning}", "yellow")
            self.model_ready = True
            self.refresh_sidebar()

        while self.queue:
            self.current = self.queue.popleft()
            self.current_details = ""
            self.refresh_sidebar()
            try:
                async for event in process_file(self.current, self.settings, self.client):
                    self.handle_event(event)
            except Exception as exc:  # показываем ошибку, а не роняем TUI
                name = self.current.path.name
                self.log_line(f"✗ {name}: unexpected error: {exc!r}", "bold red")
            finally:
                self.current = None
                self.chunk_started_at = None
                self.eta = None
                self.refresh_sidebar()
        self.notify("Queue finished", title="Done")

    # --- события конвейера -------------------------------------------------

    def handle_event(self, event: Event) -> None:
        thinking = self.query_one("#thinking", StreamView)
        answer = self.query_one("#answer", StreamView)
        progress = self.query_one("#progress", ProgressBar)

        match event:
            case ThinkingDelta():
                thinking.append(event.text)
            case ContentDelta():
                answer.append(event.text)
            case FileStarted():
                self.log_line(
                    Text.assemble(("▶ ", "bold"), (event.path.name, "bold"), " — extracting text…")
                )
            case TextExtracted():
                chars = f"{event.chars:,}".replace(",", " ")
                how = "detected automatically" if event.detected else "set manually"
                self.current_details = event.language
                refs = (
                    f" Removed reference list entries: {event.references_removed}."
                    if event.references_removed
                    else ""
                )
                self.log_line(
                    f"  {chars} characters → {event.total_chunks} fragments. "
                    f"Language: {event.language} ({how}).{refs}",
                    "dim",
                )
                self.refresh_sidebar()
            case AnalysisStarted():
                self.chunk_label = "analysing document"
                self.chunk_started_at = time.monotonic()
                thinking.reset()
                answer.reset()
                thinking.border_subtitle = "source analysis — choosing the dataset type"
                self.log_line(
                    "  🔎 The model is studying the document to choose a dataset type…", "dim"
                )
            case KindChosen():
                self.chunk_started_at = None
                self.current_details = f"{event.title} · {self.current_details}"
                self.refresh_sidebar()
                if event.how == "auto":
                    self.log_line(
                        Text.assemble(
                            ("  🔎 Dataset type: ", "bold"),
                            (event.title, "bold magenta"),
                            f" (source: {event.genre}). {event.reason}",
                        )
                    )
                elif event.how == "resumed":
                    self.log_line(
                        f"  Dataset type from the previous analysis: {event.title}", "dim"
                    )
                elif event.how == "fallback":
                    self.log_line(
                        f'! Analysis failed ({event.reason}) — using "{event.title}"', "yellow"
                    )
            case FileLoaded():
                progress.update(total=event.total_chunks, progress=event.already_done)
                resumed = f"resuming from {event.already_done}. " if event.already_done else ""
                self.log_line(f"  {resumed}Dataset: {event.output}", "dim")
                self.estimate_label = "—"
            case SpeedMeasureStarted():
                self.chunk_label = "measuring speed"
                self.chunk_started_at = time.monotonic()
                self.log_line("  ⏱ Measuring model speed on this PC (once per session)…", "dim")
            case SpeedMeasured():
                self.chunk_started_at = None
                self.speed, self.system = event.speed, event.system
                self.log_line(f"  💻 {event.system.label}", "dim")
                self.log_line(f"  ⏱ {self.speed_label(event.speed)}", "dim")
                self.refresh_hardware()
            case EstimateReady():
                estimate = event.estimate
                self.eta = Eta(estimate.chunks, estimate.per_chunk)
                self.estimate_label = format_duration(estimate.seconds)
                self.log_line(
                    Text.assemble(
                        ("  ⏱ Time estimate: ", "bold"),
                        (estimate.label, "bold yellow"),
                        f" — {estimate.chunks} fragments, ~{format_duration(estimate.per_chunk)} "
                        "per fragment. Refined as work progresses.",
                    )
                )
            case ChunkStarted():
                self.chunk_label = f"{event.index + 1}/{event.total}"
                self.chunk_started_at = time.monotonic()
                thinking.reset("" if event.think else "(thinking is off — answering directly)")
                answer.reset()
                mode = "" if event.think else " · no thinking"
                thinking.border_subtitle = f"fragment {self.chunk_label}{mode}"
            case ChunkRetry():
                self.log_line(
                    f"! Fragment {event.index + 1}: {event.reason} — retrying "
                    f"(attempt {event.attempt + 1})",
                    "yellow",
                )
            case ChunkFinished():
                self.on_chunk_finished(event)
                progress.advance(1)
                if self.eta:
                    self.eta.chunk_done(event.elapsed)
            case ChunkFailed():
                self.chunks_failed += 1
                progress.advance(1)
                if self.eta:
                    self.eta.chunk_done(event.elapsed)
                self.log_line(
                    f"✗ Fragment {event.index + 1} skipped: {event.reason}. "
                    "It will be retried the next time this file is processed.",
                    "red",
                )
            case FileFinished():
                failed = f", failed: {event.failed_chunks}" if event.failed_chunks else ""
                total = sum(event.records.values())
                by_kind = ""
                if len(event.records) > 1:
                    parts = [f"{run_title(k)}: {n}" for k, n in event.records.items() if n]
                    by_kind = f" ({', '.join(parts)})" if parts else ""
                self.log_line(
                    f"✔ {event.path.name}: {total} examples in total{by_kind}{failed} → "
                    f"{event.output}",
                    "bold green",
                )
            case FileFailed():
                self.log_line(f"✗ {event.path.name}: {event.reason}", "bold red")
        if not isinstance(event, ThinkingDelta | ContentDelta):
            self.refresh_stats()

    def on_chunk_finished(self, event: ChunkFinished) -> None:
        self.chunks_done += 1
        self.examples_written += event.written
        self.chunk_started_at = None
        if event.seconds > 0:
            self.last_speed = event.tokens / event.seconds
        if not event.examples:
            self.chunks_empty += 1
            self.log_line(f"· Fragment {event.index + 1}/{event.total}: no useful content", "dim")
            return
        dropped = len(event.examples) - event.written
        note = f", dropped duplicates/too short: {dropped}" if dropped else ""
        if self.current and self.current.kind == MIXED:
            counts: dict[str, int] = {}
            for example in event.examples:
                counts[example.kind] = counts.get(example.kind, 0) + 1
            note = " — " + ", ".join(f"{run_title(k)} ×{n}" for k, n in counts.items()) + note
        self.log_line(
            f"✓ Fragment {event.index + 1}/{event.total}: +{event.written} examples "
            f"in {format_duration(event.elapsed)}{note}",
            "bold green",
        )
        for example in event.examples:
            self.log_line(describe_example(example))
            self.log_line(f"   [{example.category} · {example.subtype}]", "dim")

    # --- боковая панель ----------------------------------------------------

    def refresh_sidebar(self) -> None:
        s = self.settings
        think = "[green]on[/]" if s.think else "[red]off[/]"
        kind = "ask" if s.dataset_kind is None else run_title(s.dataset_kind)
        language = "same as source" if s.language == "auto" else escape(s.language)
        self.query_one("#info", Static).update(
            f"Model:    [b]{escape(s.model)}[/] "
            f"{'' if self.model_ready else '[dim](not checked)[/]'}\n"
            f"Thinking: {think} [dim](F2)[/]\n"
            f"Threads:  {s.threads or 'auto'}\n"
            f"Type:     {kind}\n"
            f"Language: {language}\n"
            f"Fragment: {s.chunk_size} chars{' + system' if s.include_system else ''}\n"
            f"Input:    {escape(str(s.input_dir))}/\n"
            f"Output:   {escape(str(s.output_dir))}/"
        )
        lines = []
        if self.current:
            lines.append(Text(f"▶ {self.current.path.name}", style="bold yellow"))
            if self.current_details:
                lines.append(Text(f"  {self.current_details}", style="yellow"))
        for job in self.queue:
            lines.append(Text.assemble(f"  {job.path.name} ", (f"· {run_title(job.kind)}", "dim")))
        queue = Text("\n").join(lines) if lines else Text("empty", style="dim")
        self.query_one("#queue", Static).update(queue)

    def refresh_stats(self) -> None:
        elapsed = ""
        current = 0.0
        if self.chunk_started_at is not None:
            current = time.monotonic() - self.chunk_started_at
            seconds = int(current)
            elapsed = f" [yellow]⏳ {seconds // 60}:{seconds % 60:02d}[/]"
        speed = f"{self.last_speed:.1f} tok/s" if self.last_speed else "—"
        left = "—"
        if self.eta and self.eta.remaining:
            left = f"≈ {format_duration(self.eta.seconds_left(current))}"
        elif self.eta:
            left = "done"
        self.query_one("#stats", Static).update(
            f"Fragment:      {self.chunk_label}{elapsed}\n"
            f"Examples:      [b green]{self.examples_written}[/]\n"
            f"Processed:     {self.chunks_done}\n"
            f"No content:    {self.chunks_empty}\n"
            f"Errors:        {self.chunks_failed}\n"
            f"Speed:         {speed}\n"
            f"File estimate: {self.estimate_label}\n"
            f"Time left:     [b yellow]{left}[/]"
        )

    @staticmethod
    def speed_label(speed: Speed) -> str:
        basis = "measured" if speed.measured else "rough hardware estimate"
        return (
            f"Reading {speed.prompt_tps:.0f} tok/s, generation {speed.gen_tps:.1f} tok/s "
            f"({basis}), {speed.device}"
        )

    def refresh_hardware(self) -> None:
        lines = []
        if self.system:
            lines += [
                escape(self.system.cpu),
                f"{self.system.threads} threads · {self.system.ram_total_gb:.0f} GB RAM",
            ]
        if self.speed:
            basis = "" if self.speed.measured else " [dim](rough)[/]"
            lines += [
                escape(self.speed.device),
                f"reading:    {self.speed.prompt_tps:.0f} tok/s{basis}",
                f"generation: {self.speed.gen_tps:.1f} tok/s{basis}",
            ]
        self.query_one("#hardware", Static).update(
            "\n".join(lines) or "[dim]measured on the first file[/]"
        )

    # --- действия ------------------------------------------------------------

    def action_toggle_think(self) -> None:
        self.settings.think = not self.settings.think
        state = "on" if self.settings.think else "off"
        self.notify(f"Thinking {state} — applies from the next fragment")
        self.refresh_sidebar()

    def action_stop(self) -> None:
        if self.worker is None or not self.worker.is_running:
            self.notify("Nothing is being processed right now")
            return
        self.queue.clear()
        self.worker.cancel()
        self.log_line("■ Stopped. Progress is saved — add the file again to continue.", "yellow")

    def action_clear(self) -> None:
        self.journal.clear()
        self.query_one("#thinking", StreamView).reset()
        self.query_one("#answer", StreamView).reset()


def run_tui(settings: Settings, files: list[Path]) -> None:
    DatasetApp(settings, files).run()
