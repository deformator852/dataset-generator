from dataclasses import dataclass
from pathlib import Path

DATASET_FORMATS = ("chatml", "alpaca", "sharegpt")


@dataclass(slots=True)
class Settings:
    model: str = "qwen3:8b"
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
    think: bool = True
    retries: int = 2

    def sampling_options(self, think: bool) -> dict[str, float | int]:
        # Рекомендованные Qwen параметры сэмплинга для режимов с мышлением и без.
        if think:
            sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        else:
            sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
        return {**sampling, "num_ctx": self.num_ctx, "num_predict": self.num_predict}
