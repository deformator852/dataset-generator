from dataclasses import dataclass
from pathlib import Path

DATASET_FORMATS = ("chatml", "alpaca", "sharegpt")
EFFORTS = ("low", "high", "max")

# deepseek-flash (V4.1-Flash) is more than enough for examples based on a given text;
# deepseek-v4-pro costs about 4 times as much.
DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"


@dataclass(slots=True)
class Settings:
    model: str = DEFAULT_MODEL
    # Address of an OpenAI-compatible API; None means DeepSeek
    api_url: str | None = None
    api_key: str | None = None
    input_dir: Path = Path("input")
    output_dir: Path = Path("datasets")
    # None: ask the user; "auto"/"mixed": the model picks the type itself
    dataset_kind: str | None = None
    dataset_format: str = "chatml"
    include_system: bool = True
    # "auto": the language of the source; otherwise a language code (ru, en, uk…)
    language: str = "auto"
    chunk_size: int = 3000
    chunk_overlap: int = 300
    # None: each dataset type uses its own default
    max_items: int | None = None
    # Generation limit per fragment; doubled when thinking is on
    num_predict: int = 8192
    # Thinking costs cents, and the model picks types more accurately and errs less.
    # Toggled with --think / --no-think and the F2 key.
    think: bool = True
    # Thinking depth: low is cheap and enough for examples based on a text
    effort: str = "low"
    retries: int = 2
    # How many fragments to send to the API at once
    parallel: int = 16
