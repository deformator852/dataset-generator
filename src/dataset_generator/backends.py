"""Бэкенды модели: DeepSeek API (по умолчанию) и локальная Ollama.

Оба отдают одинаковый поток событий: размышления, ответ и в конце ответ, разобранный
по JSON-схеме. Остальная программа не знает, где работает модель.
"""

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ollama import AsyncClient, ResponseError
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
)
from pydantic import BaseModel, ValidationError

from dataset_generator.config import DEEPSEEK_BASE_URL, DEEPSEEK_KEY_ENV, Settings
from dataset_generator.events import ContentDelta, ThinkingDelta
from dataset_generator.pricing import Usage, price_for


class GenerationError(Exception):
    """Ошибка конкретной попытки генерации — запрос можно повторить."""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage  # токены неудачной попытки тоже оплачены


class FatalBackendError(Exception):
    """Продолжать бессмысленно: нет связи с Ollama, неверный ключ, кончились деньги."""


class ModelUnavailableError(Exception):
    pass


@dataclass(slots=True, frozen=True)
class Structured:
    value: BaseModel
    tokens: int
    seconds: float
    usage: Usage | None = None


class Backend:
    label = ""

    async def check(self) -> list[str]:
        """Проверяет доступность модели. Возвращает предупреждения."""
        raise NotImplementedError

    def stream(
        self, messages: list[dict[str, str]], schema: type[BaseModel], *, think: bool
    ) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
        raise NotImplementedError


def create_backend(settings: Settings) -> Backend:
    return OllamaBackend(settings) if settings.local else DeepSeekBackend(settings)


class OllamaBackend(Backend):
    label = "Ollama (local)"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncClient(host=settings.host)

    async def check(self) -> list[str]:
        settings = self.settings
        try:
            info = await self.client.show(settings.model)
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
            warnings.append(f"model {settings.model} does not support thinking mode — it is off")
        return warnings

    async def stream(
        self, messages: list[dict[str, str]], schema: type[BaseModel], *, think: bool
    ) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
        settings = self.settings
        started = time.monotonic()
        tokens = 0
        content: list[str] = []
        done_reason = None
        try:
            stream = await self.client.chat(
                model=settings.model,
                messages=messages,
                # Моделям без режима мышления (например, Qwen3 Instruct-2507) его не передаём.
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
        except ConnectionError as exc:
            raise FatalBackendError(f"lost connection to Ollama: {exc}") from exc

        if done_reason == "length":
            raise GenerationError(
                "response truncated: the model hit the token limit (increase --num-predict)"
            )
        value = _parse(schema, "".join(content))
        yield Structured(value, tokens, time.monotonic() - started)


class DeepSeekBackend(Backend):
    """DeepSeek и любой другой OpenAI-совместимый API (адрес задаётся через --host)."""

    label = "DeepSeek API"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.api_key or "missing",
            base_url=settings.host or DEEPSEEK_BASE_URL,
            max_retries=3,  # 429 и 5xx SDK повторяет сам, с паузой
            timeout=600,
        )

    async def check(self) -> list[str]:
        settings = self.settings
        if not settings.api_key:
            raise ModelUnavailableError(
                f"no API key: set {DEEPSEEK_KEY_ENV} (create a key at "
                "https://platform.deepseek.com/api_keys), or use --provider ollama"
            )
        try:
            page = await self.client.models.list()
        except AuthenticationError as exc:
            raise ModelUnavailableError(
                f"DeepSeek rejected the API key ({DEEPSEEK_KEY_ENV})"
            ) from exc
        except APIConnectionError as exc:
            raise ModelUnavailableError(f"cannot reach the API at {self.client.base_url}") from exc
        except APIStatusError as exc:
            raise ModelUnavailableError(f"API error {exc.status_code}: {exc.message}") from exc

        warnings = []
        available = {model.id for model in page.data}
        if available and settings.model not in available:
            warnings.append(
                f"model {settings.model} is not in the API's model list "
                f"({', '.join(sorted(available))})"
            )
        if price_for(settings.model) is None:
            warnings.append(f"no price table for {settings.model} — cost will not be shown")
        return warnings

    async def stream(
        self, messages: list[dict[str, str]], schema: type[BaseModel], *, think: bool
    ) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
        settings = self.settings
        request: dict = {
            "model": settings.model,
            "messages": _with_schema(messages, schema),
            "stream": True,
            "stream_options": {"include_usage": True},
            # У DeepSeek нет строгой JSON-схемы: JSON-режим + схема в промпте + проверка.
            "response_format": {"type": "json_object"},
            "max_tokens": settings.num_predict * (2 if think else 1),
        }
        if think:
            request["reasoning_effort"] = settings.effort
            extra = {"thinking": {"type": "enabled"}}
        else:
            # У DeepSeek мышление включено по умолчанию — выключаем явно.
            request["temperature"] = 0.7
            extra = {"thinking": {"type": "disabled"}}

        started = time.monotonic()
        tokens = 0
        content: list[str] = []
        finish = None
        usage: Usage | None = None
        try:
            stream = await self.client.chat.completions.create(**request, extra_body=extra)
            async for chunk in stream:
                if chunk.usage:
                    usage = _usage(chunk.usage)
                for choice in chunk.choices:
                    delta = choice.delta
                    # DeepSeek: reasoning_content; Ollama /v1 и другие: reasoning
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if reasoning:
                        tokens += 1
                        yield ThinkingDelta(reasoning)
                    if delta.content:
                        tokens += 1
                        content.append(delta.content)
                        yield ContentDelta(delta.content)
                    if choice.finish_reason:
                        finish = choice.finish_reason
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise FatalBackendError(f"the API rejected the key: {exc.message}") from exc
        except APIStatusError as exc:
            if exc.status_code == 402:
                raise FatalBackendError(
                    "insufficient DeepSeek balance — top up at https://platform.deepseek.com/top_up"
                ) from exc
            raise GenerationError(f"API error {exc.status_code}: {exc.message}") from exc
        except APIConnectionError as exc:
            raise GenerationError(f"network error: {exc}") from exc

        if finish == "length":
            raise GenerationError("response truncated: the model hit the token limit", usage)
        if finish == "content_filter":
            raise GenerationError(
                "the response was blocked by the provider's content filter", usage
            )
        if finish == "insufficient_system_resource":
            raise GenerationError("the provider is overloaded", usage)
        try:
            value = _parse(schema, "".join(content))
        except GenerationError as exc:
            raise GenerationError(str(exc), usage) from exc
        yield Structured(value, tokens, time.monotonic() - started, usage)


def _with_schema(messages: list[dict[str, str]], schema: type[BaseModel]) -> list[dict[str, str]]:
    """Добавляет JSON-схему в системный промпт. Схема одна на весь запуск, поэтому
    префикс промпта не меняется и DeepSeek берёт его из кеша почти бесплатно."""
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    system = (
        f"{messages[0]['content']}\n\nThe response must be a single JSON object that follows "
        f"this JSON schema exactly:\n{schema_text}"
    )
    return [{**messages[0], "content": system}, *messages[1:]]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _parse(schema: type[BaseModel], raw: str) -> BaseModel:
    try:
        return schema.model_validate_json(_FENCE.sub("", raw.strip()))
    except ValidationError as exc:
        errors = exc.error_count()
        raise GenerationError(f"the model returned invalid JSON ({errors} errors)") from exc


def _usage(raw: object) -> Usage:
    hit = getattr(raw, "prompt_cache_hit_tokens", None)
    miss = getattr(raw, "prompt_cache_miss_tokens", None)
    prompt = getattr(raw, "prompt_tokens", 0) or 0
    if hit is None and miss is None:
        details = getattr(raw, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", 0) or 0
        miss = prompt - hit
    details = getattr(raw, "completion_tokens_details", None)
    return Usage(
        cache_hit=hit or 0,
        cache_miss=miss or 0,
        output=getattr(raw, "completion_tokens", 0) or 0,
        reasoning=getattr(details, "reasoning_tokens", 0) or 0,
    )
