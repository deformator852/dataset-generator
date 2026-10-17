from dataclasses import dataclass
from pathlib import Path

DATASET_FORMATS = ("chatml", "alpaca", "sharegpt")
PROVIDERS = ("deepseek", "ollama")
EFFORTS = ("low", "high", "max")

# Модель по умолчанию для каждого провайдера.
# deepseek-flash (V4.1-Flash): для примеров по готовому тексту её хватает с запасом,
# а V4-Pro дороже в ~4 раза. Локально — MoE Qwen3-30B-A3B: на CPU она быстрее Qwen3-8B
# и пишет заметно лучше (нужно ~20 ГБ ОЗУ; на слабом ПК — `-m qwen3:8b`).
DEFAULT_MODELS = {
    "deepseek": "deepseek-flash",
    "ollama": "qwen3:30b-a3b-instruct-2507-q4_K_M",
}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass(slots=True)
class Settings:
    provider: str = "deepseek"
    model: str = DEFAULT_MODELS["deepseek"]
    # Ollama: None — клиент сам возьмёт OLLAMA_HOST или http://localhost:11434.
    # DeepSeek: адрес OpenAI-совместимого API.
    host: str | None = None
    api_key: str | None = None
    input_dir: Path = Path("input")
    output_dir: Path = Path("datasets")
    # None — спросить у пользователя; "auto"/"mixed" — модель выберет тип сама
    dataset_kind: str | None = None
    dataset_format: str = "chatml"
    include_system: bool = True
    # "auto" — язык источника; иначе код языка (ru, en, uk…)
    language: str = "auto"
    chunk_size: int = 3000
    chunk_overlap: int = 300
    # None — своё значение по умолчанию для каждого типа датасета
    max_items: int | None = None
    num_ctx: int = 16384
    num_predict: int = 8192
    # Мышление: у DeepSeek включено (стоит центы, а модель точнее выбирает типы и реже
    # ошибается), локально выключено (на CPU почти удваивает время). Флаги --think /
    # --no-think и клавиша F2 переключают его.
    think: bool = True
    # Глубина мышления DeepSeek: low — дёшево и достаточно для примеров по тексту
    effort: str = "low"
    retries: int = 2
    # Сколько фрагментов обрабатывать одновременно. API держит десятки запросов сразу;
    # локальная Ollama считает по одному, параллельность ей не помогает.
    parallel: int = 16
    # Ollama: None — как решит Ollama (обычно только производительные ядра)
    threads: int | None = None
    # Заполняется при проверке модели: умеет ли она режим мышления
    think_supported: bool = True

    @property
    def local(self) -> bool:
        return self.provider == "ollama"

    def sampling_options(self, think: bool) -> dict[str, float | int]:
        """Параметры Ollama: рекомендованный Qwen сэмплинг для режимов с мышлением и без."""
        if think:
            sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        else:
            sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
        options = {**sampling, "num_ctx": self.num_ctx, "num_predict": self.num_predict}
        if self.threads:
            options["num_thread"] = self.threads
        return options
