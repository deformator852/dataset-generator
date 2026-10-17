"""The DeepSeek API client (any other OpenAI-compatible API works too).

The backend yields a stream of events: thinking, the answer, and finally the answer
parsed with the JSON schema. The rest of the program does not depend on the API used.
"""

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
    """A failed generation attempt; the request can be retried."""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage  # tokens of a failed attempt are billed too


class FatalBackendError(Exception):
    """No point in continuing: invalid key, no balance left."""


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
        """Checks that the model is available. Returns warnings."""
        raise NotImplementedError

    def stream(
        self, messages: list[dict[str, str]], schema: type[BaseModel], *, think: bool
    ) -> AsyncIterator[ThinkingDelta | ContentDelta | Structured]:
        raise NotImplementedError


def create_backend(settings: Settings) -> Backend:
    return DeepSeekBackend(settings)


class DeepSeekBackend(Backend):
    """DeepSeek or any other OpenAI-compatible API (address set with --api-url)."""

    label = "DeepSeek API"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.api_key or "missing",
            base_url=settings.api_url or DEEPSEEK_BASE_URL,
            max_retries=3,  # the SDK retries 429 and 5xx itself, with backoff
            timeout=600,
        )

    async def check(self) -> list[str]:
        settings = self.settings
        if not settings.api_key:
            raise ModelUnavailableError(
                f"no API key: set {DEEPSEEK_KEY_ENV} (create a key at "
                "https://platform.deepseek.com/api_keys)"
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
            # DeepSeek has no strict JSON schema: JSON mode + schema in the prompt + checks.
            "response_format": {"type": "json_object"},
            "max_tokens": settings.num_predict * (2 if think else 1),
        }
        if think:
            request["reasoning_effort"] = settings.effort
            extra = {"thinking": {"type": "enabled"}}
        else:
            # DeepSeek thinks by default, so thinking is turned off explicitly.
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
                    # DeepSeek: reasoning_content; some other APIs: reasoning
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
    """Adds the JSON schema to the system prompt. The schema is the same for the whole run,
    so the prompt prefix does not change and DeepSeek serves it from cache almost for free."""
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
