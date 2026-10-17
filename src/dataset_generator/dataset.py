"""Запись датасета в JSONL с поддержкой возобновления прерванной обработки."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_generator.config import Settings
from dataset_generator.kinds import KINDS, MIXED, DatasetKind, Example
from dataset_generator.language import Language
from dataset_generator.prompts import TRAINING_SYSTEM_PROMPTS, training_system_prompt

_MIN_USER_CHARS = 10
_MIN_ASSISTANT_CHARS = 30
_NON_WORD = re.compile(r"[\W_]+")


def to_record(
    example: Example, *, dataset_format: str, system: str | None, metadata: dict[str, Any]
) -> dict[str, Any]:
    turns = [(user.strip(), assistant.strip()) for user, assistant in example.turns]
    if example.reasoning:
        # Рассуждение кладём в <think> перед последним ответом — так учат Qwen3/R1-модели.
        user, assistant = turns[-1]
        turns[-1] = (user, f"<think>\n{example.reasoning.strip()}\n</think>\n\n{assistant}")

    match dataset_format:
        case "chatml":
            messages = [{"role": "system", "content": system}] if system else []
            for user, assistant in turns:
                messages += [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            record: dict[str, Any] = {"messages": messages}
        case "alpaca":
            *history, (user, assistant) = turns
            record = {"instruction": user, "input": "", "output": assistant}
            if system:
                record["system"] = system
            if history:
                record["history"] = [list(turn) for turn in history]
        case "sharegpt":
            conversation = [{"from": "system", "value": system}] if system else []
            for user, assistant in turns:
                conversation += [
                    {"from": "human", "value": user},
                    {"from": "gpt", "value": assistant},
                ]
            record = {"conversations": conversation}
        case other:
            raise ValueError(f"неизвестный формат датасета: {other}")
    record["metadata"] = metadata
    return record


def sample_record(kind: DatasetKind, dataset_format: str, include_system: bool) -> dict[str, Any]:
    """Пример записи датасета — для показа пользователю при выборе типа."""
    example = kind.sample
    system = TRAINING_SYSTEM_PROMPTS["ru"].split(":")[0] + "…" if include_system else None
    metadata = {
        "source": "trasology.pdf",
        "chunk": 12,
        "kind": kind.key,
        "category": example.category,
        "type": example.subtype,
        "language": "ru",
        "model": "qwen3:8b",
    }
    return to_record(example, dataset_format=dataset_format, system=system, metadata=metadata)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_fingerprint(
    sha256: str, settings: Settings, dataset_format: str, language: Language
) -> dict[str, Any]:
    return {
        "sha256": sha256,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "format": dataset_format,
        "include_system": settings.include_system,
        "language": language.code,
    }


def find_auto_kind(source: Path, settings: Settings, fingerprint: dict[str, Any]) -> str | None:
    """Тип, который авто-анализ уже выбрал для этого файла в прошлый запуск."""
    candidates = []
    for key in KINDS:
        state_path = settings.output_dir / f"{source.name}.{key}.state.json"
        state = _read_state(state_path)
        if state and state.get("auto") and state.get("fingerprint") == fingerprint:
            candidates.append((state_path.stat().st_mtime, key))
    return max(candidates)[1] if candidates else None


class DatasetWriter:
    """Датасеты одного запуска для одного файла + файл состояния.

    Конкретный тип → `<имя файла>.<тип>.jsonl`. Режим «по фрагментам» (mixed) →
    `<имя файла>.mixed.<тип>.jsonl`, по файлу на каждый тип, который выбрала модель.
    Состояние хранит номера обработанных фрагментов и хеши вопросов, поэтому
    повторный запуск продолжает с места остановки и не плодит дубликаты.
    """

    def __init__(
        self,
        source: Path,
        settings: Settings,
        *,
        run: str,
        auto: bool,
        dataset_format: str,
        language: Language,
        fingerprint: dict[str, Any],
    ) -> None:
        self.source = source
        self.settings = settings
        self.run = run
        self.auto = auto
        self.dataset_format = dataset_format
        self.language = language
        self.fingerprint = fingerprint
        self.system = training_system_prompt(language) if settings.include_system else None
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{source.name}.{run}"
        if run == MIXED:
            self.paths = {key: settings.output_dir / f"{prefix}.{key}.jsonl" for key in KINDS}
            self.output_label = str(settings.output_dir / f"{prefix}.<тип>.jsonl")
        else:
            self.paths = {run: settings.output_dir / f"{prefix}.jsonl"}
            self.output_label = str(self.paths[run])
        self.state_path = settings.output_dir / f"{prefix}.state.json"
        self.done: set[int] = set()
        self.seen: set[str] = set()
        self.records: dict[str, int] = dict.fromkeys(self.paths, 0)
        self._load_state()

    @property
    def total_records(self) -> int:
        return sum(self.records.values())

    def _load_state(self) -> None:
        state = _read_state(self.state_path)
        if state and state.get("fingerprint") == self.fingerprint:
            self.done = set(state["done"])
            self.seen = set(state["seen"])
            self.records.update(state["records"])
            return
        # Файл или настройки изменились — старые датасеты не затираем.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in self.paths.values():
            if path.exists():
                path.rename(path.with_name(f"{path.name}.bak-{stamp}"))
        self._save_state()

    def add(self, chunk_index: int, examples: list[Example]) -> int:
        """Дописывает примеры фрагмента в датасеты и отмечает фрагмент обработанным."""
        lines: dict[str, list[str]] = {}
        for example in examples:
            if example.kind not in self.paths or not _is_valid(example):
                continue
            key = f"{example.kind}:{_question_key(example.turns[0][0])}"
            if key in self.seen:
                continue
            self.seen.add(key)
            metadata = {
                "source": self.source.name,
                "chunk": chunk_index,
                "kind": example.kind,
                "category": example.category.strip(),
                "type": example.subtype,
                "language": self.language.code,
                "model": self.settings.model,
            }
            record = to_record(
                example, dataset_format=self.dataset_format, system=self.system, metadata=metadata
            )
            lines.setdefault(example.kind, []).append(json.dumps(record, ensure_ascii=False) + "\n")
        for kind, kind_lines in lines.items():
            with self.paths[kind].open("a", encoding="utf-8") as file:
                file.writelines(kind_lines)
            self.records[kind] += len(kind_lines)
        self.done.add(chunk_index)
        self._save_state()
        return sum(len(kind_lines) for kind_lines in lines.values())

    def _save_state(self) -> None:
        state = {
            "source": str(self.source),
            "run": self.run,
            "auto": self.auto,
            "fingerprint": self.fingerprint,
            "records": self.records,
            "done": sorted(self.done),
            "seen": sorted(self.seen),
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)


def _is_valid(example: Example) -> bool:
    if not example.turns or len(example.turns[0][0].strip()) < _MIN_USER_CHARS:
        return False
    return all(
        user.strip() and len(assistant.strip()) >= _MIN_ASSISTANT_CHARS
        for user, assistant in example.turns
    )


def _question_key(question: str) -> str:
    normalized = _NON_WORD.sub(" ", question.lower()).strip()
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
