"""DeepSeek pricing and cost calculation.

Prices are from https://api-docs.deepseek.com/quick_start/pricing (October 2026), per
1M tokens at peak hours. Off-peak everything is half price. If DeepSeek changes its
prices, update the table.
"""

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(slots=True, frozen=True)
class Price:
    cache_hit: float
    cache_miss: float
    output: float


PEAK_PRICES = {
    "deepseek-flash": Price(cache_hit=0.006, cache_miss=0.30, output=1.20),
    "deepseek-v4-pro": Price(cache_hit=0.044, cache_miss=1.32, output=3.96),
}
# Legacy names are still accepted and billed as Flash.
ALIASES = {"deepseek-v4-flash": "deepseek-flash", "deepseek-v4-flash-vision-exp": "deepseek-flash"}
# Peak hours (UTC): Mon–Fri 01:00–04:00 and 06:00–10:00.
_PEAK_HOURS = {1, 2, 3, 6, 7, 8, 9}


@dataclass(slots=True, frozen=True)
class Usage:
    cache_hit: int = 0
    cache_miss: int = 0
    output: int = 0  # including thinking
    reasoning: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.cache_hit + other.cache_hit,
            self.cache_miss + other.cache_miss,
            self.output + other.output,
            self.reasoning + other.reasoning,
        )


def is_peak(moment: datetime | None = None) -> bool:
    moment = (moment or datetime.now(UTC)).astimezone(UTC)
    return moment.weekday() < 5 and moment.hour in _PEAK_HOURS


def price_for(model: str, moment: datetime | None = None) -> Price | None:
    price = PEAK_PRICES.get(ALIASES.get(model, model))
    if price is None:
        return None
    if is_peak(moment):
        return price
    return Price(price.cache_hit / 2, price.cache_miss / 2, price.output / 2)


def cost(usage: Usage, model: str, moment: datetime | None = None) -> float:
    price = price_for(model, moment)
    if price is None:
        return 0.0
    return (
        usage.cache_hit * price.cache_hit
        + usage.cache_miss * price.cache_miss
        + usage.output * price.output
    ) / 1e6


def format_cost(dollars: float) -> str:
    return f"${dollars:.3f}" if dollars < 1 else f"${dollars:.2f}"
