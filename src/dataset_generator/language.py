"""Определение языка источника — на нём же пишется датасет."""

from collections import Counter
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # детерминированный результат между запусками

# код → (название по-русски для интерфейса, по-английски для промпта, самоназвание)
_NAMES = {
    "ru": ("русский", "Russian", "русский"),
    "uk": ("украинский", "Ukrainian", "українська"),
    "be": ("белорусский", "Belarusian", "беларуская"),
    "bg": ("болгарский", "Bulgarian", "български"),
    "mk": ("македонский", "Macedonian", "македонски"),
    "en": ("английский", "English", "English"),
    "de": ("немецкий", "German", "Deutsch"),
    "fr": ("французский", "French", "français"),
    "es": ("испанский", "Spanish", "español"),
    "it": ("итальянский", "Italian", "italiano"),
    "pt": ("португальский", "Portuguese", "português"),
    "pl": ("польский", "Polish", "polski"),
    "cs": ("чешский", "Czech", "čeština"),
    "sk": ("словацкий", "Slovak", "slovenčina"),
    "hr": ("хорватский", "Croatian", "hrvatski"),
    "sl": ("словенский", "Slovenian", "slovenščina"),
    "lt": ("литовский", "Lithuanian", "lietuvių"),
    "lv": ("латышский", "Latvian", "latviešu"),
    "et": ("эстонский", "Estonian", "eesti"),
    "tr": ("турецкий", "Turkish", "Türkçe"),
    "zh-cn": ("китайский", "Chinese", "中文"),
}

_SAMPLES = 9
_SAMPLE_CHARS = 1200


@dataclass(slots=True, frozen=True)
class Language:
    code: str
    name: str  # по-русски, для интерфейса
    english: str  # по-английски, для промпта
    native: str

    @property
    def label(self) -> str:
        return self.name if self.name == self.native else f"{self.name} ({self.native})"

    @property
    def prompt_label(self) -> str:
        return self.english if self.english == self.native else f"{self.english} ({self.native})"


def language_from_code(code: str) -> Language:
    code = code.strip().lower()
    name, english, native = _NAMES.get(code, (code, code, code))
    return Language(code, name, english, native)


def detect_language(text: str) -> Language:
    """Голосование по нескольким фрагментам из разных частей текста."""
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
