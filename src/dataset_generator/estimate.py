"""Прогноз времени и стоимости обработки и оставшееся время (ETA).

Прогноз строится по объёму текста, тарифу DeepSeek и типичной скорости API, а по ходу
работы ETA уточняется по фактическому времени фрагментов.
"""

from dataclasses import dataclass

from dataset_generator.pricing import Usage, cost

# Служебный текст запроса (шаблон пользовательского сообщения), символов.
USER_TEMPLATE_CHARS = 200
# Разброс прогноза: длина размышлений и ответов сильно гуляет.
LOW_FACTOR, HIGH_FACTOR = 0.6, 1.7
# Осторожные допущения о DeepSeek Flash — ETA уточнит их по факту.
API_FIRST_TOKEN_SECONDS = 2.0
API_TOKENS_PER_SECOND = 50.0
API_THINK_TOKENS = {"low": 600, "high": 2500, "max": 6000}
API_CHARS_PER_TOKEN = 3.8


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


def estimate_api(
    chunks: list[str],
    *,
    output_chars: int,
    think: bool,
    effort: str,
    system_chars: int,
    parallel: int,
    model: str,
) -> tuple[Estimate, float]:
    """Прогноз времени и стоимости для облачного API (DeepSeek).

    Системный промпт одинаков во всех запросах: DeepSeek кеширует его, и после первой
    волны параллельных запросов он оплачивается по тарифу «cache hit».
    """
    if not chunks:
        return Estimate(0, 0.0, 0.0), 0.0
    output_tokens = output_chars / API_CHARS_PER_TOKEN * 1.15
    if think:
        output_tokens += API_THINK_TOKENS.get(effort, 600)
    per_chunk = API_FIRST_TOKEN_SECONDS + output_tokens / API_TOKENS_PER_SECOND
    waves = -(-len(chunks) // max(parallel, 1))
    estimate = Estimate(len(chunks), per_chunk, per_chunk * waves)

    system_tokens = system_chars / API_CHARS_PER_TOKEN
    cold = min(parallel, len(chunks))  # первая волна ещё не попадает в кеш
    usage = Usage(
        cache_hit=int(system_tokens * (len(chunks) - cold)),
        cache_miss=int(
            system_tokens * cold
            + sum(len(chunk) + USER_TEMPLATE_CHARS for chunk in chunks) / API_CHARS_PER_TOKEN
        ),
        output=int(output_tokens * len(chunks)),
    )
    return estimate, cost(usage, model)


class Eta:
    """Оставшееся время: прогноз уточняется по фактическому времени фрагментов.

    При параллельной обработке фрагменты идут «волнами», поэтому время делится на
    число одновременных запросов.
    """

    def __init__(self, remaining: int, prior_per_chunk: float, parallel: int = 1) -> None:
        self.remaining = remaining
        self.prior = prior_per_chunk
        self.parallel = max(parallel, 1)
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
        if self.parallel > 1:
            return self.per_chunk * -(-self.remaining // self.parallel)
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
