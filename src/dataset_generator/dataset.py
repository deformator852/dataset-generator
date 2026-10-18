"""Writing datasets to JSONL with support for resuming interrupted processing."""

import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset_generator.config import DEFAULT_MODEL, Settings
from dataset_generator.kinds import KINDS, MIXED, DatasetKind, Example
from dataset_generator.language import Language
from dataset_generator.prompts import TRAINING_SYSTEM_PROMPT
from dataset_generator.source_card import source_id

# Bumped when the record layout changes, so a rerun does not mix old and new records in
# one file. 2: quality labels (evidence, support, quote) and source_id.
RECORD_VERSION = 2

_MIN_USER_CHARS = 10
_MIN_ASSISTANT_CHARS = 30
_NON_WORD = re.compile(r"[\W_]+")


def to_record(
    example: Example, *, dataset_format: str, system: str | None, metadata: dict[str, Any]
) -> dict[str, Any]:
    turns = [(user.strip(), assistant.strip()) for user, assistant in example.turns]
    if example.reasoning:
        # Reasoning goes into <think> before the last answer, as Qwen3/R1 models are trained.
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
            raise ValueError(f"unknown dataset format: {other}")
    record["metadata"] = metadata
    return record


def sample_record(kind: DatasetKind, dataset_format: str, include_system: bool) -> dict[str, Any]:
    """An example dataset record, shown to the user in the type picker."""
    example = replace(kind.sample, kind=kind.key)
    system = TRAINING_SYSTEM_PROMPT.split(":")[0] + "…" if include_system else None
    metadata = _metadata(
        example, source="trasology.pdf", source_id="3f9a1c0d2b7e4a65", chunk=12, language="en"
    ) | {"model": DEFAULT_MODEL, "quote_match": "exact"}
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
        "version": RECORD_VERSION,
        "sha256": sha256,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "format": dataset_format,
        "include_system": settings.include_system,
        "language": language.code,
    }


class DatasetWriter:
    """The datasets of one run for one file, plus a state file.

    A specific type → `<file name>.<type>.jsonl`. Auto — per fragment (mixed) →
    `<file name>.mixed.<type>.jsonl`, one file per type the model picked.
    The state keeps the indices of processed fragments and question hashes, so a
    rerun continues where it stopped and does not create duplicates.
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
        self.system = TRAINING_SYSTEM_PROMPT if settings.include_system else None
        self.source_id = source_id(fingerprint["sha256"])
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{source.name}.{run}"
        if run == MIXED:
            self.paths = {key: settings.output_dir / f"{prefix}.{key}.jsonl" for key in KINDS}
            self.output_label = str(settings.output_dir / f"{prefix}.<type>.jsonl")
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
        # The file or the settings changed: keep the old datasets instead of overwriting.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for path in self.paths.values():
            if path.exists():
                path.rename(path.with_name(f"{path.name}.bak-{stamp}"))
        self._save_state()

    def add(self, chunk_index: int, examples: list[Example]) -> int:
        """Appends a fragment's examples to the datasets and marks the fragment as done."""
        lines: dict[str, list[str]] = {}
        for example in examples:
            if example.kind not in self.paths or not _is_valid(example):
                continue
            key = f"{example.kind}:{_question_key(example.turns[0][0])}"
            if key in self.seen:
                continue
            self.seen.add(key)
            metadata = _metadata(
                example,
                source=self.source.name,
                source_id=self.source_id,
                chunk=chunk_index,
                language=self.language.code,
            ) | {"model": self.settings.model}
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


def _metadata(
    example: Example, *, source: str, source_id: str, chunk: int, language: str
) -> dict[str, Any]:
    return {
        "source": source,
        "source_id": source_id,
        "chunk": chunk,
        "kind": example.kind,
        "category": example.category.strip(),
        "type": example.subtype,
        # Quality labels: what the example rests on, how it relates to the fragment, the
        # supporting passage and whether the program found that passage in the fragment.
        "evidence": example.evidence,
        "support": example.support,
        "quote": example.quote,
        "quote_match": example.quote_match,
        "language": language,
    }


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
