"""Detecting the source language, which is also the dataset language."""

from collections import Counter
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic results across runs

# code → English name of the language
_NAMES = {
    "ru": "Russian",
    "uk": "Ukrainian",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "mk": "Macedonian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "hr": "Croatian",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "tr": "Turkish",
    "zh-cn": "Chinese",
}

_SAMPLES = 9
_SAMPLE_CHARS = 1200


@dataclass(slots=True, frozen=True)
class Language:
    code: str
    label: str  # English name, used in the interface and in the model prompt


def language_from_code(code: str) -> Language:
    code = code.strip().lower()
    return Language(code, _NAMES.get(code, code))


def detect_language(text: str) -> Language:
    """Majority vote over several samples from different parts of the text."""
    step = max(len(text) // _SAMPLES, 1)
    votes: Counter[str] = Counter()
    for start in range(0, len(text), step):
        sample = text[start : start + _SAMPLE_CHARS]
        try:
            votes[detect(sample)] += 1
        except LangDetectException:
            continue
    code = votes.most_common(1)[0][0] if votes else "en"
    return language_from_code(code)
