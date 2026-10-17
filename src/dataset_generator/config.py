from dataclasses import dataclass
from pathlib import Path

DATASET_FORMATS = ("chatml", "alpaca", "sharegpt")
EFFORTS = ("low", "high", "max")

# deepseek-flash (V4.1-Flash): для примеров по готовому тексту её хватает с запасом,
# а deepseek-v4-pro дороже примерно в 4 раза.
DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass(slots=True)
class Settings:
    model: str = DEFAULT_MODEL
    # Адрес OpenAI-совместимого API; None — DeepSeek
    api_url: str | None = None
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
    # Лимит генерации на фрагмент; с мышлением — вдвое больше
    num_predict: int = 8192
    # Мышление стоит центы, а модель точнее выбирает типы и реже ошибается.
    # Переключается флагами --think / --no-think и клавишей F2.
    think: bool = True
    # Глубина мышления: low — дёшево и достаточно для примеров по тексту
    effort: str = "low"
    retries: int = 2
    # Сколько фрагментов отправлять в API одновременно
    parallel: int = 16
