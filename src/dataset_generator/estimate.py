"""Оценка времени обработки: характеристики ПК, замер скорости модели, прогноз и ETA.

Модель сама не знает, как быстро работает на этом ПК, поэтому скорость не спрашиваем,
а измеряем: отправляем настоящий запрос первого фрагмента и просим всего несколько
токенов ответа. Скорость сильно зависит от длины контекста и JSON-схемы, поэтому
короткий «синтетический» тест врёт в разы. Если модель только загружается, кеш Ollama
пуст: замер честный, а прочитанный промпт потом переиспользует реальный запрос. Если
модель уже была загружена, в кеше может лежать наш промпт с прошлого запуска — тогда
в начало добавляем случайную метку, иначе чтение промпта покажется мгновенным.
Если замер не удался — грубая оценка по размеру модели и тому, где она запущена.
"""

import os
import platform
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ollama import AsyncClient, ProcessResponse, ResponseError
from pydantic import BaseModel

from dataset_generator.config import Settings

# Ожидаемый объём размышлений Qwen3 на одном фрагменте, токенов.
THINK_TOKENS = 1000
# Накладные расходы на запрос (разбор JSON, запись, сетевые задержки), секунд.
REQUEST_OVERHEAD = 1.5
# Служебный текст запроса (шаблон пользовательского сообщения), символов.
USER_TEMPLATE_CHARS = 200
# Разброс прогноза: длина размышлений и ответов сильно гуляет.
LOW_FACTOR, HIGH_FACTOR = 0.6, 1.7

_BENCH_PREDICT = 48
# Если токенов меньше, чем символы/6, число токенов недостоверно.
_MAX_CHARS_PER_TOKEN = 6
# Чтение промпта в сотни раз быстрее генерации бывает только при попадании в кеш.
_MAX_PROMPT_TO_GEN = 200
_speed_cache: dict[tuple[str | None, str], "Speed"] = {}


@dataclass(slots=True, frozen=True)
class SystemInfo:
    cpu: str
    threads: int
    ram_total_gb: float
    ram_free_gb: float

    @property
    def label(self) -> str:
        ram = f"{self.ram_total_gb:.0f} GB RAM"
        if self.ram_free_gb:
            ram += f" ({self.ram_free_gb:.0f} free)"
        return f"{self.cpu} · {self.threads} threads · {ram}"


@dataclass(slots=True, frozen=True)
class Speed:
    prompt_tps: float  # скорость чтения промпта, токенов/с
    gen_tps: float  # скорость генерации, токенов/с
    chars_per_token: float
    gpu_share: float | None  # доля модели в видеопамяти, None — неизвестно
    measured: bool  # True — замер, False — грубая оценка по железу

    @property
    def device(self) -> str:
        if self.gpu_share is None:
            return "device unknown"
        if self.gpu_share >= 0.99:
            return "model on GPU"
        if self.gpu_share <= 0.01:
            return "model on CPU"
        return f"model {self.gpu_share:.0%} on GPU, the rest on CPU"


@dataclass(slots=True, frozen=True)
class Estimate:
    chunks: int
    per_chunk: float
    seconds: float

    @property
    def low(self) -> float:
        return self.seconds * LOW_FACTOR

    @property
    def high(self) -> float:
        return self.seconds * HIGH_FACTOR

    @property
    def label(self) -> str:
        return (
            f"≈ {format_duration(self.seconds)} "
            f"({format_duration(self.low)} to {format_duration(self.high)})"
        )


def system_info() -> SystemInfo:
    return SystemInfo(_cpu_name(), os.cpu_count() or 1, *_memory_gb())


def speed_cached(settings: Settings) -> bool:
    return (settings.host, settings.model) in _speed_cache


async def measure_speed(
    client: AsyncClient,
    settings: Settings,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
) -> Speed:
    """Замер скорости модели на реальном запросе (результат кешируется на сеанс)."""
    key = (settings.host, settings.model)
    if key not in _speed_cache:
        try:
            _speed_cache[key] = await _benchmark(client, settings, messages, schema)
        except (ResponseError, ZeroDivisionError, TypeError):
            _speed_cache[key] = await _speed_from_hardware(client, settings)
    return _speed_cache[key]


def estimate_chunks(
    chunks: list[str],
    speed: Speed,
    *,
    output_chars: int,
    think: bool,
    system_chars: int,
) -> Estimate:
    """Прогноз для оставшихся фрагментов.

    System-промпт одинаков для всех запросов, и Ollama переиспользует его из кеша,
    поэтому читается он один раз; каждый фрагмент — заново.
    """
    if not chunks:
        return Estimate(0, 0.0, 0.0)
    output_tokens = output_chars / speed.chars_per_token * 1.15 + (THINK_TOKENS if think else 0)
    generation = output_tokens / speed.gen_tps
    total = system_chars / speed.chars_per_token / speed.prompt_tps
    for chunk in chunks:
        prompt_tokens = (len(chunk) + USER_TEMPLATE_CHARS) / speed.chars_per_token
        total += prompt_tokens / speed.prompt_tps + generation + REQUEST_OVERHEAD
    return Estimate(len(chunks), total / len(chunks), total)


class Eta:
    """Оставшееся время: прогноз уточняется по фактическому времени фрагментов."""

    def __init__(self, remaining: int, prior_per_chunk: float) -> None:
        self.remaining = remaining
        self.prior = prior_per_chunk
        self.samples: list[float] = []

    def chunk_done(self, elapsed: float) -> None:
        self.remaining = max(self.remaining - 1, 0)
        self.samples.append(elapsed)

    @property
    def per_chunk(self) -> float:
        recent = self.samples[-10:]
        return (sum(recent) + self.prior) / (len(recent) + 1)

    def seconds_left(self, current_elapsed: float = 0.0) -> float:
        if not self.remaining:
            return 0.0
        current = max(self.per_chunk - current_elapsed, 0.0)
        return current + self.per_chunk * (self.remaining - 1)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h" if hours else f"{days} d"


async def _benchmark(
    client: AsyncClient,
    settings: Settings,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
) -> Speed:
    if await _running_model(client, settings) is not None:
        nonce = f"[{secrets.token_hex(4)}]\n"
        messages = [{**messages[0], "content": nonce + messages[0]["content"]}, *messages[1:]]
    request = {
        "model": settings.model,
        "messages": messages,
        "format": schema.model_json_schema(),
        # Тот же num_ctx, что и в работе, иначе Ollama перезагрузит модель.
        "options": {**settings.sampling_options(False), "num_predict": _BENCH_PREDICT},
    }
    try:
        # Без мышления: замеряем чистую скорость генерации ответа.
        response = await client.chat(**request, think=False)
    except ResponseError:
        response = await client.chat(**request)  # модель без поддержки параметра think
    gen_tps = response.eval_count / (response.eval_duration / 1e9)
    gpu_share = _gpu_share(await _running_model(client, settings))
    chars = sum(len(message["content"]) for message in messages)
    tokens = response.prompt_eval_count or 0
    chars_per_token = chars / tokens if tokens >= chars / _MAX_CHARS_PER_TOKEN else 3.0
    prompt_tps = tokens / max(response.prompt_eval_duration / 1e9, 1e-9)
    if prompt_tps > gen_tps * _MAX_PROMPT_TO_GEN:
        # Промпт уже был в кеше Ollama (повторный запуск): токены посчитаны, а время —
        # почти ноль. Берём типичное соотношение чтения и генерации для CPU/GPU.
        prompt_tps = gen_tps * (30 if gpu_share and gpu_share > 0.5 else 4)
    return Speed(prompt_tps, gen_tps, chars_per_token, gpu_share, True)


async def _speed_from_hardware(client: AsyncClient, settings: Settings) -> Speed:
    """Грубая оценка: генерация упирается в пропускную способность памяти."""
    size = 5e9
    try:
        listing = await client.list()
        size = next((m.size for m in listing.models if m.model == settings.model and m.size), size)
    except ResponseError:
        pass
    gpu_share = _gpu_share(await _running_model(client, settings))
    on_gpu = bool(gpu_share and gpu_share > 0.5)
    bandwidth = 300e9 if on_gpu else 40e9  # байт/с: типичная видеокарта / двухканальная DDR
    gen_tps = bandwidth / float(size)
    return Speed(gen_tps * (40 if on_gpu else 8), gen_tps, 3.0, gpu_share, False)


async def _running_model(client: AsyncClient, settings: Settings) -> ProcessResponse.Model | None:
    """Модель, если она уже загружена в Ollama."""
    try:
        running = await client.ps()
    except ResponseError:
        return None
    return next((m for m in running.models if settings.model in (m.model, m.name)), None)


def _gpu_share(model: ProcessResponse.Model | None) -> float | None:
    if model is None or not model.size:
        return None
    return (model.size_vram or 0) / model.size


def _cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.startswith("model name"):
                return " ".join(
                    line.split(":", 1)[1].replace("(R)", "").replace("(TM)", "").split()
                )
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine() or "CPU"


def _memory_gb() -> tuple[float, float]:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = {}
        for line in meminfo.read_text().splitlines():
            name, _, rest = line.partition(":")
            values[name] = int(rest.split()[0]) * 1024
        return values.get("MemTotal", 0) / 1e9, values.get("MemAvailable", 0) / 1e9
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0.0, 0.0
    return total / 1e9, 0.0
