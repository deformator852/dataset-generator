"""Работа с моделью через Ollama: проверка модели, анализ источника и генерация примеров."""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ValidationError

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
from dataset_generator.prompts import (
    build_analysis_messages,
    build_generation_messages,
    build_mixed_messages,
)


class GenerationError(Exception):
    """Ошибка конкретной попытки генерации — запрос можно повторить."""


class ModelUnavailableError(Exception):
    pass


class SourceAnalysis(BaseModel):
    genre: str
    topic: str
    kind: Literal[tuple(KINDS)]  # type: ignore[valid-type]
    reason: str


@dataclass(slots=True, frozen=True)
class Structured:
    value: BaseModel
    tokens: int
    seconds: float


@dataclass(slots=True, frozen=True)
class Generation:
    examples: list[Example]
    tokens: int
    seconds: float


async def check_model(client: AsyncClient, settings: Settings) -> list[str]:
    """Проверяет, что Ollama доступна и модель скачана. Возвращает предупреждения."""
    try:
        info = await client.show(settings.model)
    except ConnectionError as exc:
        raise ModelUnavailableError(
            "Ollama is not responding. Start it: sudo systemctl start ollama (or ollama serve)"
        ) from exc
    except ResponseError as exc:
        if exc.status_code == 404:
            raise ModelUnavailableError(
                f"model {settings.model} not found. Download it: ollama pull {settings.model}"
            ) from exc
        raise ModelUnavailableError(f"Ollama: {exc.error}") from exc

    warnings = []
    settings.think_supported = "thinking" in (info.capabilities or [])
    if settings.think and not settings.think_supported:
        settings.think = False
        warnings.append(f"model {settings.model} does not support thinking mode — thinking is off")
    return warnings


async def stream_structured(
    client: AsyncClient,
    settings: Settings,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    *,
    think: bool,
) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
    """Стримит размышления и ответ модели, в конце отдаёт ответ, разобранный по схеме."""
    started = time.monotonic()
    tokens = 0
    content: list[str] = []
    done_reason = None
    try:
        stream = await client.chat(
            model=settings.model,
            messages=messages,
            # Моделям без режима мышления (например, Qwen3 Instruct-2507) параметр не передаём.
            think=think if settings.think_supported else None,
            format=schema.model_json_schema(),
            options=settings.sampling_options(think),
            stream=True,
        )
        async for part in stream:
            if part.message.thinking:
                tokens += 1
                yield ThinkingDelta(part.message.thinking)
            if part.message.content:
                tokens += 1
                content.append(part.message.content)
                yield ContentDelta(part.message.content)
            if part.done:
                done_reason = part.done_reason
    except ResponseError as exc:
        raise GenerationError(f"Ollama: {exc.error}") from exc

    if done_reason == "length":
        raise GenerationError(
            "response truncated: the model hit the token limit (increase --num-predict)"
        )
    try:
        value = schema.model_validate_json("".join(content))
    except ValidationError as exc:
        errors = exc.error_count()
        raise GenerationError(f"the model returned invalid JSON ({errors} errors)") from exc
    yield Structured(value, tokens, time.monotonic() - started)


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
    client: AsyncClient,
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
    async for event in stream_structured(client, settings, messages, schema, think=think):
        if not isinstance(event, Structured):
            yield event
            continue
        if run == MIXED:
            examples = mixed_examples(event.value, max_items)
        else:
            items = event.value.items[:max_items]  # type: ignore[attr-defined]
            examples = [KINDS[run].make_example(item) for item in items]
        yield Generation(examples, event.tokens, event.seconds)


async def analyze_source(
    client: AsyncClient, settings: Settings, *, source: str, samples: list[str], think: bool
) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
    messages = build_analysis_messages(source, samples)
    async for event in stream_structured(client, settings, messages, SourceAnalysis, think=think):
        yield event
