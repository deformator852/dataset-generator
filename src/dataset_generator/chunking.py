"""Splitting text into fragments at paragraph and sentence boundaries."""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Groups paragraphs into fragments of up to `size` characters.

    Neighbouring fragments overlap by the tail of the previous one (up to `overlap`
    characters), so a thought split by the boundary is not lost.
    """
    units: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        units.extend([line] if len(line) <= size else _split_long(line, size))

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    has_new_text = False
    for unit in units:
        if has_new_text and length + len(unit) > size:
            chunks.append("\n".join(current))
            current = _tail(current, overlap)
            length = sum(len(u) + 1 for u in current)
            if length + len(unit) > size:
                current, length = [], 0
            has_new_text = False
        current.append(unit)
        length += len(unit) + 1
        has_new_text = True
    if has_new_text:
        chunks.append("\n".join(current))
    return chunks


def _split_long(paragraph: str, size: int) -> list[str]:
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END.split(paragraph):
        while len(sentence) > size:
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if buffer and len(buffer) + len(sentence) + 1 > size:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}" if buffer else sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def _tail(units: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        if total + len(unit) > overlap:
            if not tail:
                # The paragraph is longer than the overlap: take its last sentences.
                partial = _last_sentences(unit, overlap)
                tail = [partial] if partial else []
            break
        tail.insert(0, unit)
        total += len(unit) + 1
    return tail


def _last_sentences(paragraph: str, limit: int) -> str:
    picked: list[str] = []
    total = 0
    for sentence in reversed(_SENTENCE_END.split(paragraph)):
        if total + len(sentence) > limit:
            break
        picked.insert(0, sentence)
        total += len(sentence) + 1
    return " ".join(picked)


# Signs of a bibliographic entry: a year plus publication details or author initials.
# The Cyrillic parts match Russian-language bibliographies ("С. 45" = p. 45, "М.:" =
# Moscow, "СПб." = St Petersburg, headings like "Список литературы" = References);
# they are patterns for the input text, not text to translate.
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_BIBLIO_MARKERS = re.compile(
    r"//|\bС\.\s*\d|\d+\s*с\.|\bpp?\.\s*\d|\bvol\.|\bт\.\s*\d|№\s*\d|doi|https?://|isbn|"
    r"\bМ\.\s*:|\bСПб\.|et al\.|\(\d{4}[a-z]?\)",
    re.IGNORECASE,
)
_INITIALS = re.compile(r"[A-ZА-ЯЁ][a-zа-яё]+,?\s+[A-ZА-ЯЁ]\.\s?(?:[A-ZА-ЯЁ]\.)?")
_REFERENCES_HEADING = re.compile(
    r"^\s*(список\s+(использованной\s+|рекомендуемой\s+)?литературы|"
    r"список\s+использованных\s+источников|библиографи(я|ческий\s+список)|литература|"
    r"references|bibliography|works\s+cited|literature)\s*:?\s*$",
    re.IGNORECASE,
)
_NUMBERED = re.compile(r"^\s*\[?\d{1,3}[.)\]]\s.{0,200}$")
_REFERENCE_LINE_CHARS = 350
_MIN_REFERENCE_RUN = 4
_GAP_LINE_CHARS = 200  # a long entry wrapped onto a second line does not break the list


def strip_references(text: str) -> tuple[str, int]:
    """Removes reference lists (up to a third of an article); the model does not need them.

    A list is at least four bibliographic entries in a row: short lines with a year and
    publication details or initials. A heading such as "References" before it is removed
    too. Ordinary paragraphs citing laws are longer and are not affected.
    Returns the text and the number of removed entries.
    """
    lines = text.split("\n")
    is_reference = [_is_reference(line) for line in lines]
    keep = [True] * len(lines)
    removed = 0
    i = 0
    while i < len(lines):
        if not is_reference[i]:
            i += 1
            continue
        j, count, last = i, 0, i
        while j < len(lines):
            if is_reference[j]:
                count, last = count + 1, j
            elif not (
                j + 1 < len(lines) and is_reference[j + 1] and len(lines[j]) <= _GAP_LINE_CHARS
            ):
                break
            j += 1
        if count >= _MIN_REFERENCE_RUN:
            start = i
            # The start of the first entry wrapped onto its own line ("1. Smith, J. …").
            if start > 0 and _NUMBERED.match(lines[start - 1]):
                start -= 1
            if start > 0 and _REFERENCES_HEADING.match(lines[start - 1]):
                start -= 1
            keep[start : last + 1] = [False] * (last + 1 - start)
            removed += count
        i = last + 1
    return "\n".join(line for line, kept in zip(lines, keep, strict=True) if kept), removed


def _is_reference(line: str) -> bool:
    return bool(
        len(line) <= _REFERENCE_LINE_CHARS
        and _YEAR.search(line)
        and (_BIBLIO_MARKERS.search(line) or _INITIALS.search(line))
    )
