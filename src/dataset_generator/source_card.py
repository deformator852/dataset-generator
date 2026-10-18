"""Source cards: what a source is, what its content rests on, and what came out of it.

A card is `<file name>.source.json` next to the datasets. The model fills in the
description once per file (title, genre, methods, sample, findings, limitations…); the
program adds the statistics of every dataset made from the file: how many examples there
are, what evidence they rest on, and how many supporting quotes were found in the source.
"""

import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dataset_generator.config import Settings

_LABELS = ("evidence", "support", "quote_match")
_TOP_CATEGORIES = 10


def source_id(sha256: str) -> str:
    """A short identifier that links examples to their card even if the file is renamed."""
    return sha256[:16]


def card_path(source: Path, settings: Settings) -> Path:
    return settings.output_dir / f"{source.name}.source.json"


def load_card(path: Path, sha256: str) -> dict[str, Any] | None:
    """A card of this very file whose analysis succeeded, or None."""
    card = read_card(path)
    if not card or card.get("sha256") != sha256 or "genre" not in card:
        return None
    return card


def new_card(
    source: Path,
    *,
    sha256: str,
    language: str,
    chars: int,
    model: str,
    based_on: str,
    description: dict[str, Any] | None,
    error: str = "",
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "source": source.name,
        "source_id": source_id(sha256),
        "sha256": sha256,
        "language": language,
        "chars": chars,
    }
    if description:
        description = dict(description)
        kind, reason = description.pop("kind"), description.pop("reason")
        description["evidence_basis"] = [e for e in description["evidence_basis"] if e]
        card |= description
        card |= {"recommended_kind": kind, "kind_reason": reason}
    card["analysis"] = {
        "model": model,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "based_on": based_on,
    } | ({"error": error} if error else {})
    # Statistics of the datasets made earlier from the same file stay.
    same_file = previous and previous.get("sha256") == sha256
    card["datasets"] = previous.get("datasets", {}) if same_file else {}  # type: ignore[union-attr]
    return card


def save_card(path: Path, card: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_datasets(path: Path, datasets: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Recounts the given datasets into the card. Returns their statistics."""
    card = read_card(path)
    if card is None:
        return {}
    entries: dict[str, Any] = card.setdefault("datasets", {})
    stats = {dataset.name: dataset_stats(dataset) for dataset in datasets if dataset.exists()}
    entries.update(stats)
    # Datasets renamed to .bak or deleted since are no longer described by the card.
    for name in [name for name in entries if not (path.parent / name).exists()]:
        del entries[name]
    save_card(path, card)
    return stats


def dataset_stats(path: Path) -> dict[str, Any]:
    counts = {label: Counter[str]() for label in _LABELS}
    categories = Counter[str]()
    examples = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                metadata = json.loads(line).get("metadata", {})
            except (json.JSONDecodeError, AttributeError):
                continue
            examples += 1
            for label, counter in counts.items():
                counter[metadata.get(label) or "unlabelled"] += 1
            categories[str(metadata.get("category", "")).strip().lower()] += 1
    return {
        "examples": examples,
        **{label: dict(counter.most_common()) for label, counter in counts.items()},
        "top_categories": dict(categories.most_common(_TOP_CATEGORIES)),
    }


def verified_share(stats: dict[str, dict[str, Any]]) -> float | None:
    """Share of examples whose supporting quote was found in the source (exact or partial)."""
    total = sum(entry["examples"] for entry in stats.values())
    found = sum(
        entry["quote_match"].get("exact", 0) + entry["quote_match"].get("partial", 0)
        for entry in stats.values()
    )
    return found / total if total else None


def read_card(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
