"""Запросы к модели: анализ источника и генерация примеров (через любой бэкенд)."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from dataset_generator.backends import Backend, Structured
from dataset_generator.config import Settings
from dataset_generator.events import ContentDelta, ThinkingDelta
from dataset_generator.kinds import (
    KINDS,
    MIXED,
    MIXED_RESULT,
    Example,
    max_items_for,
    mixed_examples,
)
from dataset_generator.language import Language
from dataset_generator.pricing import Usage
from dataset_generator.prompts import (
    build_analysis_messages,
    build_generation_messages,
    build_mixed_messages,
)


class SourceAnalysis(BaseModel):
    genre: str
    topic: str
    kind: Literal[tuple(KINDS)]  # type: ignore[valid-type]
    reason: str


@dataclass(slots=True, frozen=True)
class Generation:
    examples: list[Example]
    tokens: int
    seconds: float
    usage: Usage | None = None


def generation_request(
    run: str, language: Language, *, source: str, chunk: str, index: int, total: int, max_items: int
) -> tuple[list[dict[str, str]], type[BaseModel]]:
    """Сообщения и схема ответа для фрагмента: конкретный тип или режим «по фрагментам»."""
    if run == MIXED:
        messages = build_mixed_messages(
            language, source=source, chunk=chunk, number=index + 1, total=total, max_items=max_items
        )
        return messages, MIXED_RESULT
    kind = KINDS[run]
    messages = build_generation_messages(
        kind,
        language,
        source=source,
        chunk=chunk,
        number=index + 1,
        total=total,
        max_items=max_items,
    )
    return messages, kind.result_model


async def generate_examples(
    backend: Backend,
    settings: Settings,
    run: str,
    language: Language,
    *,
    source: str,
    chunk: str,
    index: int,
    total: int,
    think: bool,
) -> AsyncIterator[ThinkingDelta | ContentDelta | Generation]:
    max_items = max_items_for(run, settings.max_items)
    messages, schema = generation_request(
        run, language, source=source, chunk=chunk, index=index, total=total, max_items=max_items
    )
    async for event in backend.stream(messages, schema, think=think):
        if isinstance(event, ThinkingDelta):
            yield ThinkingDelta(event.text, index)
        elif isinstance(event, ContentDelta):
            yield ContentDelta(event.text, index)
        else:
            if run == MIXED:
                examples = mixed_examples(event.value, max_items)
            else:
                items = event.value.items[:max_items]  # type: ignore[attr-defined]
                examples = [KINDS[run].make_example(item) for item in items]
            yield Generation(examples, event.tokens, event.seconds, event.usage)


async def analyze_source(
    backend: Backend, *, source: str, samples: list[str], think: bool
) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
    messages = build_analysis_messages(source, samples)
    async for event in backend.stream(messages, SourceAnalysis, think=think):
        yield event
