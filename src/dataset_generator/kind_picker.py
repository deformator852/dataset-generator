"""Окно выбора типа и формата датасета с живым примером записи."""

import json
from pathlib import Path
from typing import ClassVar

from rich.console import Group, RenderableType
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList, RadioButton, RadioSet, Static
from textual.widgets.option_list import Option

from dataset_generator.config import DATASET_FORMATS, Settings
from dataset_generator.dataset import sample_record
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
    Example,
)


def describe_example(example: Example) -> Text:
    """Человекочитаемый вид примера: реплики пользователя и ассистента."""
    text = Text()
    for number, (user, assistant) in enumerate(example.turns):
        if number:
            text.append("\n")
        text.append("Пользователь: ", style="bold cyan")
        text.append(f"{user}\n")
        if example.reasoning and number == len(example.turns) - 1:
            text.append("Рассуждение: ", style="bold yellow")
            text.append(f"{example.reasoning}\n", style="italic")
        text.append("Ассистент: ", style="bold green")
        text.append(f"{assistant}\n")
    return text


class KindPicker(ModalScreen[tuple[str, str] | None]):
    """Возвращает (тип датасета, формат) или None, если пользователь передумал."""

    DEFAULT_CSS = """
    KindPicker { align: center middle; }
    KindPicker #dialog {
        width: 94%; max-width: 160; height: 92%;
        border: thick $primary; background: $surface; padding: 0 1;
    }
    KindPicker #title { text-style: bold; width: 1fr; padding: 1 0 0 0; }
    KindPicker #hint { color: $text-muted; width: 1fr; padding-bottom: 1; }
    KindPicker #choice { height: 1fr; }
    KindPicker #kinds { width: 40; height: 1fr; }
    KindPicker #preview { width: 1fr; height: 1fr; border: round $accent; padding: 0 1; }
    KindPicker #bottom { height: auto; padding: 1 0 0 0; }
    KindPicker #format-label { padding: 1 1 0 0; }
    KindPicker #formats { layout: horizontal; width: auto; }
    KindPicker #formats > RadioButton { width: auto; margin-right: 2; }
    KindPicker #spacer { width: 1fr; }
    KindPicker Button { margin-left: 1; }
    """

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Отмена")]

    def __init__(self, files: list[Path], settings: Settings) -> None:
        super().__init__()
        self.files = files
        self.settings = settings
        self.kind = settings.dataset_kind or MIXED
        self.dataset_format = settings.dataset_format

    def compose(self) -> ComposeResult:
        name = f"«{self.files[0].name}»"
        if len(self.files) > 1:
            name += f" и ещё {len(self.files) - 1} файл(ов)"
        options = [
            Option(
                self._option_prompt(
                    MIXED_ICON,
                    MIXED_TITLE,
                    "рекомендуется: несколько датасетов, тип у каждого фрагмента свой",
                ),
                id=MIXED,
            ),
            Option(
                self._option_prompt(AUTO_ICON, AUTO_TITLE, "один датасет, тип на весь документ"),
                id=AUTO,
            ),
        ]
        options += [
            Option(self._option_prompt(kind.icon, kind.title, kind.summary), id=kind.key)
            for kind in KINDS.values()
        ]
        with Vertical(id="dialog"):
            yield Label(f"Какой датасет сделать из {name}?", id="title")
            language = (
                "Язык датасета определится автоматически — как у источника."
                if self.settings.language == "auto"
                else f"Язык датасета: {self.settings.language}."
            )
            yield Label(f"{language} ↑↓ — тип, Enter — начать, Esc — отмена.", id="hint")
            with Horizontal(id="choice"):
                yield OptionList(*options, id="kinds")
                with VerticalScroll(id="preview"):
                    yield Static(id="preview-body")
            with Horizontal(id="bottom"):
                yield Label("Формат записи:", id="format-label")
                with RadioSet(id="formats"):
                    for fmt in DATASET_FORMATS:
                        yield RadioButton(fmt, value=fmt == self.dataset_format, name=fmt)
                yield Static(id="spacer")
                yield Button("Отмена", id="cancel")
                yield Button("Начать ▶", id="start", variant="primary")

    def on_mount(self) -> None:
        kinds = self.query_one("#kinds", OptionList)
        kinds.highlighted = kinds.get_option_index(self.kind)
        kinds.focus()
        # Пример листается колесом мыши, а Tab сразу переходит к выбору формата.
        self.query_one("#preview").can_focus = False
        self.query_one("#preview").border_title = "Пример"
        self._update_preview()

    @staticmethod
    def _option_prompt(icon: str, title: str, summary: str) -> Text:
        return Text.assemble((f"{icon} {title}", "bold"), "\n", (f"   {summary}", "dim"))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_id:
            self.kind = event.option_id
            self._update_preview()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self.kind = event.option_id
            self.action_start()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed.name:
            self.dataset_format = event.pressed.name
            self._update_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        else:
            self.action_cancel()

    def action_start(self) -> None:
        self.dismiss((self.kind, self.dataset_format))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _update_preview(self) -> None:
        self.query_one("#preview-body", Static).update(self._preview())
        self.query_one("#preview", VerticalScroll).scroll_home(animate=False)

    def _preview(self) -> RenderableType:
        if self.kind == MIXED:
            name = self.files[0].name
            parts: list[RenderableType] = [
                Text(f"{MIXED_ICON} {MIXED_TITLE}", style="bold"),
                Text(MIXED_SUMMARY),
                Text(""),
                Text("Какой фрагмент в какой датасет попадёт:", style="bold"),
            ]
            parts += [
                Text.assemble(
                    ("  • ", "dim"),
                    (f"{kind.icon} {kind.title}", "bold"),
                    f" — если во фрагменте {kind.fragment_hint}\n",
                    (f"    → datasets/{name}.mixed.{kind.key}.jsonl", "cyan"),
                )
                for kind in KINDS.values()
            ]
            parts.append(
                Text(
                    "\nФайлы появляются только для тех типов, которые модель реально выбрала. "
                    "Примеры каждого типа — в списке слева. Запрос к модели длиннее обычного, "
                    "поэтому фрагмент обрабатывается немного дольше.",
                    style="dim",
                )
            )
            return Group(*parts)

        if self.kind == AUTO:
            parts = [
                Text(f"{AUTO_ICON} {AUTO_TITLE}", style="bold"),
                Text(AUTO_SUMMARY),
                Text(""),
                Text("Что выберет модель в зависимости от источника:", style="bold"),
            ]
            parts += [
                Text.assemble(
                    ("  • ", "dim"), (f"{kind.icon} {kind.title}", "bold"), f" — {kind.suits}"
                )
                for kind in KINDS.values()
            ]
            parts.append(
                Text(
                    "\nВыбор и его обоснование появятся в журнале. Решение "
                    "запоминается: при повторном запуске анализ не повторяется.",
                    style="dim",
                )
            )
            return Group(*parts)

        kind = KINDS[self.kind]
        record = sample_record(kind, self.dataset_format, self.settings.include_system)
        record_json = json.dumps(record, ensure_ascii=False, indent=2)
        return Group(
            Text(f"{kind.icon} {kind.title}", style="bold"),
            Text(kind.summary),
            Text.assemble(("Подходит для: ", "bold"), kind.suits),
            Text(""),
            describe_example(kind.sample),
            Text(f"Так выглядит строка датасета ({self.dataset_format}, .jsonl):", style="bold"),
            Syntax(
                record_json, "json", theme="ansi_dark", word_wrap=True, background_color="default"
            ),
        )
