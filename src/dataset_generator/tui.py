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
    SUB_TITLE = "датасеты по криминалистике · Ollama"

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
        Binding("f2", "toggle_think", "Мышление вкл/выкл"),
        Binding("f3", "stop", "Стоп"),
        Binding("f4", "clear", "Очистить"),
        Binding("ctrl+q", "quit", "Выход", priority=True),
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
            placeholder="Имя файла или путь (pdf, epub, fb2, mobi, djvu, docx, txt…) — Enter",
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
        self.query_one("#info").border_title = "Настройки"
        self.query_one("#queue").border_title = "Очередь"
        self.query_one("#stats").border_title = "Статистика и время"
        self.query_one("#hardware").border_title = "ПК и скорость модели"
        self.query_one("#thinking").border_title = "🧠 Размышления модели"
        self.query_one("#answer").border_title = "📝 Ответ модели (JSON)"
        self.query_one("#journal").border_title = "📚 Датасет и журнал"
        self.refresh_sidebar()
        self.refresh_hardware()
        self.set_interval(1, self.refresh_stats)
        self.query_one(Input).focus()
        self.log_line(
            f"Введите имя файла из «{self.settings.input_dir}/» или путь к файлу/папке и "
            "нажмите Enter. → — автодополнение, пустой Enter — перезапустить очередь.",
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
            self.log_line(f"✗ Не найден файл или папка: {value}", "bold red")
            return
        supported = [f for f in files if is_supported(f)]
        for path in files:
            if path not in supported:
                self.log_line(
                    f"✗ Формат {path.suffix or '(без расширения)'} не поддерживается: {path.name}",
                    "red",
                )
        if supported:
            self.request_jobs(supported)

    def request_jobs(self, files: list[Path]) -> None:
        """Ставит файлы в очередь; если тип датасета не задан флагом — спрашивает."""
        if self.settings.dataset_kind:
            self.enqueue(files, self.settings.dataset_kind, self.settings.dataset_format)
            return

        def on_choice(choice: tuple[str, str] | None) -> None:
            if choice is None:
                self.log_line("Отменено — файлы не добавлены.", "dim")
            else:
                self.enqueue(files, *choice)

        self.push_screen(KindPicker(files, self.settings), on_choice)

    def enqueue(self, files: list[Path], kind: str, dataset_format: str) -> None:
        self.queue.extend(Job(path, kind, dataset_format) for path in files)
        names = ", ".join(p.name for p in files[:5])
        more = f" и ещё {len(files) - 5}" if len(files) > 5 else ""
        self.log_line(f"+ В очередь: {names}{more} · {run_title(kind)} · {dataset_format}", "cyan")
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
                self.log_line("Когда исправите — нажмите Enter в пустом поле ввода.", "dim")
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
                self.log_line(f"✗ {name}: непредвиденная ошибка: {exc!r}", "bold red")
            finally:
                self.current = None
                self.chunk_started_at = None
                self.eta = None
                self.refresh_sidebar()
        self.notify("Очередь обработана", title="Готово")

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
                    Text.assemble(("▶ ", "bold"), (event.path.name, "bold"), " — извлекаю текст…")
                )
            case TextExtracted():
                chars = f"{event.chars:,}".replace(",", " ")
                how = "определён автоматически" if event.detected else "задан вручную"
                self.current_details = event.language
                refs = (
                    f" Вырезан список литературы: {event.references_removed} записей."
                    if event.references_removed
                    else ""
                )
                self.log_line(
                    f"  {chars} символов → {event.total_chunks} фрагментов. "
                    f"Язык: {event.language} ({how}).{refs}",
                    "dim",
                )
                self.refresh_sidebar()
            case AnalysisStarted():
                self.chunk_label = "анализ документа"
                self.chunk_started_at = time.monotonic()
                thinking.reset()
                answer.reset()
                thinking.border_subtitle = "анализ источника — выбор типа датасета"
                self.log_line("  🔎 Модель изучает документ, чтобы выбрать тип датасета…", "dim")
            case KindChosen():
                self.chunk_started_at = None
                self.current_details = f"{event.title} · {self.current_details}"
                self.refresh_sidebar()
                if event.how == "auto":
                    self.log_line(
                        Text.assemble(
                            ("  🔎 Тип датасета: ", "bold"),
                            (event.title, "bold magenta"),
                            f" (источник: {event.genre}). {event.reason}",
                        )
                    )
                elif event.how == "resumed":
                    self.log_line(f"  Тип датасета из прошлого анализа: {event.title}", "dim")
                elif event.how == "fallback":
                    self.log_line(
                        f"! Анализ не удался ({event.reason}) — беру «{event.title}»", "yellow"
                    )
            case FileLoaded():
                progress.update(total=event.total_chunks, progress=event.already_done)
                resumed = f"продолжаю с {event.already_done}. " if event.already_done else ""
                self.log_line(f"  {resumed}Датасет: {event.output}", "dim")
                self.estimate_label = "—"
            case SpeedMeasureStarted():
                self.chunk_label = "замер скорости"
                self.chunk_started_at = time.monotonic()
                self.log_line("  ⏱ Замеряю скорость модели на этом ПК (один раз за сеанс)…", "dim")
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
                        ("  ⏱ Оценка времени: ", "bold"),
                        (estimate.label, "bold yellow"),
                        f" — {estimate.chunks} фрагм., ~{format_duration(estimate.per_chunk)} "
                        "на фрагмент. Уточняется по ходу работы.",
                    )
                )
            case ChunkStarted():
                self.chunk_label = f"{event.index + 1}/{event.total}"
                self.chunk_started_at = time.monotonic()
                thinking.reset("" if event.think else "(мышление выключено — ответ сразу)")
                answer.reset()
                mode = "" if event.think else " · без мышления"
                thinking.border_subtitle = f"фрагмент {self.chunk_label}{mode}"
            case ChunkRetry():
                self.log_line(
                    f"! Фрагмент {event.index + 1}: {event.reason} — повтор "
                    f"(попытка {event.attempt + 1})",
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
                    f"✗ Фрагмент {event.index + 1} пропущен: {event.reason}. "
                    "Его повторят при следующем запуске этого файла.",
                    "red",
                )
            case FileFinished():
                failed = f", не удалось: {event.failed_chunks}" if event.failed_chunks else ""
                total = sum(event.records.values())
                by_kind = ""
                if len(event.records) > 1:
                    parts = [f"{run_title(k)}: {n}" for k, n in event.records.items() if n]
                    by_kind = f" ({', '.join(parts)})" if parts else ""
                self.log_line(
                    f"✔ {event.path.name}: всего {total} примеров{by_kind}{failed} → "
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
            self.log_line(
                f"· Фрагмент {event.index + 1}/{event.total}: полезного содержания нет", "dim"
            )
            return
        dropped = len(event.examples) - event.written
        note = f", отброшено дублей/коротких: {dropped}" if dropped else ""
        if self.current and self.current.kind == MIXED:
            counts: dict[str, int] = {}
            for example in event.examples:
                counts[example.kind] = counts.get(example.kind, 0) + 1
            note = " — " + ", ".join(f"{run_title(k)} ×{n}" for k, n in counts.items()) + note
        self.log_line(
            f"✓ Фрагмент {event.index + 1}/{event.total}: +{event.written} примеров "
            f"за {format_duration(event.elapsed)}{note}",
            "bold green",
        )
        for example in event.examples:
            self.log_line(describe_example(example))
            self.log_line(f"   [{example.category} · {example.subtype}]", "dim")

    # --- боковая панель ----------------------------------------------------

    def refresh_sidebar(self) -> None:
        s = self.settings
        think = "[green]вкл[/]" if s.think else "[red]выкл[/]"
        kind = "спрашивать" if s.dataset_kind is None else run_title(s.dataset_kind)
        language = "как у источника" if s.language == "auto" else escape(s.language)
        self.query_one("#info", Static).update(
            f"Модель:   [b]{escape(s.model)}[/] "
            f"{'' if self.model_ready else '[dim](не проверена)[/]'}\n"
            f"Мышление: {think} [dim](F2)[/]\n"
            f"Тип:      {kind}\n"
            f"Язык:     {language}\n"
            f"Фрагмент: {s.chunk_size} симв.{' + system' if s.include_system else ''}\n"
            f"Вход:     {escape(str(s.input_dir))}/\n"
            f"Выход:    {escape(str(s.output_dir))}/"
        )
        lines = []
        if self.current:
            lines.append(Text(f"▶ {self.current.path.name}", style="bold yellow"))
            if self.current_details:
                lines.append(Text(f"  {self.current_details}", style="yellow"))
        for job in self.queue:
            lines.append(Text.assemble(f"  {job.path.name} ", (f"· {run_title(job.kind)}", "dim")))
        queue = Text("\n").join(lines) if lines else Text("пусто", style="dim")
        self.query_one("#queue", Static).update(queue)

    def refresh_stats(self) -> None:
        elapsed = ""
        current = 0.0
        if self.chunk_started_at is not None:
            current = time.monotonic() - self.chunk_started_at
            seconds = int(current)
            elapsed = f" [yellow]⏳ {seconds // 60}:{seconds % 60:02d}[/]"
        speed = f"{self.last_speed:.1f} ток/с" if self.last_speed else "—"
        left = "—"
        if self.eta and self.eta.remaining:
            left = f"≈ {format_duration(self.eta.seconds_left(current))}"
        elif self.eta:
            left = "готово"
        self.query_one("#stats", Static).update(
            f"Фрагмент:      {self.chunk_label}{elapsed}\n"
            f"Примеров:      [b green]{self.examples_written}[/]\n"
            f"Обработано:    {self.chunks_done}\n"
            f"Без пользы:    {self.chunks_empty}\n"
            f"Ошибок:        {self.chunks_failed}\n"
            f"Скорость:      {speed}\n"
            f"Оценка файла:  {self.estimate_label}\n"
            f"Осталось:      [b yellow]{left}[/]"
        )

    @staticmethod
    def speed_label(speed: Speed) -> str:
        basis = "замер" if speed.measured else "грубая оценка по железу"
        return (
            f"Чтение {speed.prompt_tps:.0f} ток/с, генерация {speed.gen_tps:.1f} ток/с "
            f"({basis}), {speed.device}"
        )

    def refresh_hardware(self) -> None:
        lines = []
        if self.system:
            lines += [
                escape(self.system.cpu),
                f"{self.system.threads} потоков · ОЗУ {self.system.ram_total_gb:.0f} ГБ",
            ]
        if self.speed:
            basis = "" if self.speed.measured else " [dim](грубо)[/]"
            lines += [
                escape(self.speed.device),
                f"чтение:    {self.speed.prompt_tps:.0f} ток/с{basis}",
                f"генерация: {self.speed.gen_tps:.1f} ток/с{basis}",
            ]
        self.query_one("#hardware", Static).update(
            "\n".join(lines) or "[dim]замер при первом файле[/]"
        )

    # --- действия ------------------------------------------------------------

    def action_toggle_think(self) -> None:
        self.settings.think = not self.settings.think
        state = "включено" if self.settings.think else "выключено"
        self.notify(f"Мышление {state} — применится со следующего фрагмента")
        self.refresh_sidebar()

    def action_stop(self) -> None:
        if self.worker is None or not self.worker.is_running:
            self.notify("Сейчас ничего не обрабатывается")
            return
        self.queue.clear()
        self.worker.cancel()
        self.log_line(
            "■ Остановлено. Прогресс сохранён — добавьте файл снова, чтобы продолжить.", "yellow"
        )

    def action_clear(self) -> None:
        self.journal.clear()
        self.query_one("#thinking", StreamView).reset()
        self.query_one("#answer", StreamView).reset()


def run_tui(settings: Settings, files: list[Path]) -> None:
    DatasetApp(settings, files).run()
