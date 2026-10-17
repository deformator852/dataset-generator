import argparse
import asyncio
import os
import sys
from pathlib import Path

from dataset_generator.config import (
    DATASET_FORMATS,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_KEY_ENV,
    DEFAULT_MODELS,
    EFFORTS,
    PROVIDERS,
    Settings,
)
from dataset_generator.extract import SUPPORTED_SUFFIXES, is_supported
from dataset_generator.kinds import AUTO, KINDS, MIXED
from dataset_generator.pipeline import resolve_input


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="dataset-generator",
        description=(
            "Turns articles, studies, books and other documents into forensic science "
            "training datasets using DeepSeek (cloud API) or a local model in Ollama."
        ),
        epilog="Supported formats: " + ", ".join(sorted(SUPPORTED_SUFFIXES)),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="file names (also looked up in --input-dir) or folders; "
        "without arguments the TUI asks for a file",
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=PROVIDERS,
        default=defaults.provider,
        help=f"where the model runs: deepseek — cloud API (key in {DEEPSEEK_KEY_ENV}), "
        "ollama — locally (%(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="model name (default: "
        + ", ".join(f"{p} — {m}" for p, m in DEFAULT_MODELS.items())
        + ")",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"API address (default: {DEEPSEEK_BASE_URL}) or Ollama address "
        "(default: OLLAMA_HOST or localhost:11434)",
    )
    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=defaults.parallel,
        help="fragments processed at once with the API (%(default)s; Ollama always uses 1)",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORTS,
        default=defaults.effort,
        help="DeepSeek thinking depth (%(default)s): higher is slower and costs more",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=defaults.input_dir,
        help="folder with source files (%(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="folder for datasets (%(default)s)",
    )
    parser.add_argument(
        "-k",
        "--kind",
        choices=[MIXED, AUTO, *KINDS],
        default=None,
        help="dataset type: mixed — the model picks types for each fragment (several "
        "datasets); auto — the model picks one type for the whole document; "
        + ", ".join(f"{key} — {kind.title}" for key, kind in KINDS.items())
        + ". If omitted, the app asks and shows examples",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=DATASET_FORMATS,
        default=defaults.dataset_format,
        help="JSONL record format (%(default)s)",
    )
    parser.add_argument(
        "--no-system", action="store_true", help="do not add a system prompt to training examples"
    )
    parser.add_argument(
        "--language",
        default=defaults.language,
        help="dataset language: auto — same as the source, or a code (en, ru, uk…) (%(default)s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.chunk_size,
        help="text fragment size in characters (%(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=defaults.chunk_overlap,
        help="fragment overlap in characters (%(default)s)",
    )
    parser.add_argument(
        "-n",
        "--max-items",
        type=int,
        default=None,
        help="max examples per fragment (default depends on the type)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=defaults.num_ctx,
        help="model context window in tokens (%(default)s)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=defaults.num_predict,
        help="generated token limit, including thinking (%(default)s)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=defaults.retries,
        help="retries on generation errors (%(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=None,
        help="CPU threads for the model (default: Ollama decides, usually only performance cores)",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--think",
        action="store_true",
        help="enable thinking mode (default for DeepSeek; locally about twice as slow)",
    )
    thinking.add_argument(
        "--no-think",
        action="store_true",
        help="disable thinking mode (default for Ollama; with DeepSeek cheaper and faster)",
    )
    parser.add_argument(
        "--no-tui", action="store_true", help="plain streaming terminal output instead of the TUI"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_size < 500:
        parser.error("--chunk-size must be at least 500")
    if not 0 <= args.overlap < args.chunk_size // 2:
        parser.error("--overlap must be between 0 and half of --chunk-size")
    if (args.max_items is not None and args.max_items < 1) or args.retries < 0:
        parser.error("--max-items must be ≥ 1 and --retries ≥ 0")
    if args.parallel < 1:
        parser.error("--parallel must be ≥ 1")
    if args.think:
        think = True
    elif args.no_think:
        think = False
    else:
        # Мышление у DeepSeek стоит центы, а на локальном CPU почти удваивает время.
        think = args.provider == "deepseek"

    settings = Settings(
        provider=args.provider,
        model=args.model or DEFAULT_MODELS[args.provider],
        host=args.host,
        api_key=os.environ.get(DEEPSEEK_KEY_ENV) if args.provider == "deepseek" else None,
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
        think=think,
        effort=args.effort,
        retries=args.retries,
        parallel=args.parallel,
        threads=args.threads,
    )
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for raw in args.files:
        found = [path for path in resolve_input(raw, settings) if is_supported(path)]
        if not found:
            parser.error(f"no supported file found: {raw}")
        files.extend(found)

    if args.no_tui:
        if not files:
            parser.error("--no-tui needs at least one file")
        try:
            sys.exit(asyncio.run(run_console_lazy(settings, files)))
        except KeyboardInterrupt:
            print("\nStopped. Progress is saved — run again to continue where you left off.")
            sys.exit(130)

    from dataset_generator.tui import run_tui

    run_tui(settings, files)


async def run_console_lazy(settings: Settings, files: list[Path]) -> int:
    from dataset_generator.console import run_console

    return await run_console(settings, files)
