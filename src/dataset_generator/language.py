"""Detecting the source language, which is also the dataset language."""

from collections import Counter
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic results across runs

# code → (English name, native name)
_NAMES = {
    "ru": ("Russian", "русский"),
    "uk": ("Ukrainian", "українська"),
    "be": ("Belarusian", "беларуская"),
    "bg": ("Bulgarian", "български"),
    "mk": ("Macedonian", "македонски"),
    "en": ("English", "English"),
    "de": ("German", "Deutsch"),
    "fr": ("French", "français"),
    "es": ("Spanish", "español"),
    "it": ("Italian", "italiano"),
    "pt": ("Portuguese", "português"),
    "pl": ("Polish", "polski"),
    "cs": ("Czech", "čeština"),
    "sk": ("Slovak", "slovenčina"),
    "hr": ("Croatian", "hrvatski"),
    "sl": ("Slovenian", "slovenščina"),
    "lt": ("Lithuanian", "lietuvių"),
    "lv": ("Latvian", "latviešu"),
    "et": ("Estonian", "eesti"),
    "tr": ("Turkish", "Türkçe"),
    "zh-cn": ("Chinese", "中文"),
}

_SAMPLES = 9
_SAMPLE_CHARS = 1200


@dataclass(slots=True, frozen=True)
class Language:
    code: str
    english: str
    native: str

    @property
    def label(self) -> str:
        """A label like "Russian (русский)", used in the interface and the model prompt."""
        return self.english if self.english == self.native else f"{self.english} ({self.native})"


def language_from_code(code: str) -> Language:
    code = code.strip().lower()
    english, native = _NAMES.get(code, (code, code))
    return Language(code, english, native)


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
