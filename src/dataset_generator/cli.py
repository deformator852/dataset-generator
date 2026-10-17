import argparse
import asyncio
import sys
from pathlib import Path

from dataset_generator.config import DATASET_FORMATS, Settings
from dataset_generator.extract import SUPPORTED_SUFFIXES, is_supported
from dataset_generator.kinds import AUTO, KINDS, MIXED
from dataset_generator.pipeline import resolve_input


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="dataset-generator",
        description=(
            "Превращает статьи, исследования, книги и другие документы в обучающий датасет по "
            "криминалистике "
            "с помощью локальной модели в Ollama."
        ),
        epilog="Поддерживаемые форматы: " + ", ".join(sorted(SUPPORTED_SUFFIXES)),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="имена файлов (ищутся также в папке --input-dir) или папки; "
        "без аргументов TUI спросит файл сам",
    )
    parser.add_argument("-m", "--model", default=defaults.model, help="модель Ollama (%(default)s)")
    parser.add_argument(
        "--host", default=None, help="адрес Ollama (по умолчанию OLLAMA_HOST или localhost:11434)"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=defaults.input_dir,
        help="папка с исходными файлами (%(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="папка для датасетов (%(default)s)",
    )
    parser.add_argument(
        "-k",
        "--kind",
        choices=[MIXED, AUTO, *KINDS],
        default=None,
        help="тип датасета: mixed — модель выбирает тип для каждого фрагмента (несколько "
        "датасетов); auto — модель выбирает один тип на весь документ; "
        + ", ".join(f"{key} — {kind.title}" for key, kind in KINDS.items())
        + ". Не указан — программа спросит с примерами",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=DATASET_FORMATS,
        default=defaults.dataset_format,
        help="формат записей JSONL (%(default)s)",
    )
    parser.add_argument(
        "--no-system", action="store_true", help="не добавлять system-промпт в обучающие примеры"
    )
    parser.add_argument(
        "--language",
        default=defaults.language,
        help="язык датасета: auto — как у источника, или код (ru, en, uk…) (%(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.chunk_size,
        help="размер фрагмента текста в символах (%(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=defaults.chunk_overlap,
        help="перекрытие фрагментов в символах (%(default)s)",
    )
    parser.add_argument(
        "-n",
        "--max-items",
        type=int,
        default=None,
        help="максимум примеров на фрагмент (по умолчанию своё для каждого типа)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=defaults.num_ctx,
        help="контекстное окно модели в токенах (%(default)s)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=defaults.num_predict,
        help="лимит генерируемых токенов, включая мышление (%(default)s)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=defaults.retries,
        help="повторы при ошибке генерации (%(default)s)",
    )
    parser.add_argument(
        "--no-think", action="store_true", help="выключить режим мышления Qwen3 (в разы быстрее)"
    )
    parser.add_argument(
        "--no-tui", action="store_true", help="простой потоковый вывод в терминал вместо TUI"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_size < 500:
        parser.error("--chunk-size должен быть не меньше 500")
    if not 0 <= args.overlap < args.chunk_size // 2:
        parser.error("--overlap должен быть от 0 до половины --chunk-size")
    if (args.max_items is not None and args.max_items < 1) or args.retries < 0:
        parser.error("--max-items должен быть ≥ 1, --retries ≥ 0")

    settings = Settings(
        model=args.model,
        host=args.host,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_kind=args.kind,
        dataset_format=args.format,
        include_system=not args.no_system,
        language=args.language,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        max_items=args.max_items,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        think=not args.no_think,
        retries=args.retries,
    )
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for raw in args.files:
        found = [path for path in resolve_input(raw, settings) if is_supported(path)]
        if not found:
            parser.error(f"не найден поддерживаемый файл: {raw}")
        files.extend(found)

    if args.no_tui:
        if not files:
            parser.error("в режиме --no-tui укажите хотя бы один файл")
        try:
            sys.exit(asyncio.run(run_console_lazy(settings, files)))
        except KeyboardInterrupt:
            print("\nОстановлено. Прогресс сохранён — повторный запуск продолжит с того же места.")
            sys.exit(130)

    from dataset_generator.tui import run_tui

    run_tui(settings, files)


async def run_console_lazy(settings: Settings, files: list[Path]) -> int:
    from dataset_generator.console import run_console

    return await run_console(settings, files)
