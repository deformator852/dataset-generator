from dataclasses import dataclass
from pathlib import Path

DATASET_FORMATS = ("chatml", "alpaca", "sharegpt")


@dataclass(slots=True)
class Settings:
    # MoE-модель: 30B параметров, но на токен работают ~3B — на CPU она быстрее Qwen3-8B
    # и пишет заметно лучше. Нужно ~20 ГБ ОЗУ; на слабом ПК — `-m qwen3:8b`.
    model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    # None — клиент Ollama сам возьмёт OLLAMA_HOST или http://localhost:11434
    host: str | None = None
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
    # Мышление по умолчанию выключено: на CPU оно почти удваивает время, а для примеров
    # по готовому тексту даёт немного. Включается флагом --think или клавишей F2.
    think: bool = False
    retries: int = 2
    # None — как решит Ollama (обычно только производительные ядра)
    threads: int | None = None
    # Заполняется при проверке модели: умеет ли она режим мышления
    think_supported: bool = True

    def sampling_options(self, think: bool) -> dict[str, float | int]:
        # Рекомендованные Qwen параметры сэмплинга для режимов с мышлением и без.
        if think:
            sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        else:
            sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
        options = {**sampling, "num_ctx": self.num_ctx, "num_predict": self.num_predict}
        if self.threads:
            options["num_thread"] = self.threads
        return options
