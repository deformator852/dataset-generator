"""Quality labels: what an example's knowledge rests on and how well the source backs it."""

import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Annotated

from pydantic import AfterValidator, WithJsonSchema

# The nature of the knowledge in an example (or a source), as the source presents it.
EVIDENCE = {
    "empirical": "results of a specific study, experiment, survey or data analysis",
    "review": "a synthesis of other studies or of the literature",
    "case": "a description of one specific case or example from casework",
    "theory": "a hypothesis, model, typology or interpretation that the source does not "
    "present as tested",
    "expert_opinion": "the author's position, practical experience or recommendation "
    "without supporting data",
    "established": "generally accepted knowledge: definitions, textbook principles, standard "
    "methods",
    "normative": "a law, regulation, standard or official procedure",
}

# How an example relates to the fragment it was generated from.
SUPPORT = {
    "stated": "everything in the answer is stated in the fragment",
    "inferred": "the answer combines or generalises statements of the fragment, or draws a "
    "conclusion that the fragment implies but does not state",
    "constructed": "a hypothetical scenario or task built to apply the material; the solution "
    "follows from the fragment",
}

GENRES = {
    "research_article": "a primary research paper",
    "review": "a review article",
    "meta_analysis": "a systematic review or meta-analysis",
    "thesis": "a dissertation or thesis",
    "monograph": "a scholarly book on one subject",
    "textbook": "a textbook, lecture notes or a handbook",
    "guideline": "a guideline, manual, SOP, protocol or standard",
    "report": "a study, agency or technical report",
    "expert_report": "a forensic expert report or casework report",
    "case_report": "a description of one or several cases",
    "legal_text": "a law, a commentary on legislation or case law",
    "other": "anything else",
}

RATINGS = ("high", "medium", "low")


def _known(values: Iterable[str]) -> Callable[[str], str]:
    """A label outside the list becomes "" instead of failing the whole response."""
    allowed = frozenset(values)

    def normalize(value: str) -> str:
        value = value.strip().lower().replace("-", "_").replace(" ", "_")
        return value if value in allowed else ""

    return normalize


Evidence = Annotated[
    str, AfterValidator(_known(EVIDENCE)), WithJsonSchema({"type": "string", "enum": [*EVIDENCE]})
]
Support = Annotated[
    str, AfterValidator(_known(SUPPORT)), WithJsonSchema({"type": "string", "enum": [*SUPPORT]})
]
Genre = Annotated[
    str, AfterValidator(_known(GENRES)), WithJsonSchema({"type": "string", "enum": [*GENRES]})
]
Rating = Annotated[
    str, AfterValidator(_known(RATINGS)), WithJsonSchema({"type": "string", "enum": [*RATINGS]})
]

_WORD = re.compile(r"\w+")
_MIN_PARTIAL = 0.6  # share of the quote's word triples that must occur in the fragment


def match_quote(quote: str, text: str) -> str:
    """Checks the supporting quote against the fragment the example was generated from.

    exact — the quote occurs in the fragment, ignoring case, punctuation and spacing;
    partial — most of its word triples do (a lightly edited or stitched quote);
    none — it does not: the quote was altered beyond recognition or invented.
    """
    words = _words(quote)
    if not words:
        return "none"
    source = _words(text)
    if f" {' '.join(words)} " in f" {' '.join(source)} ":
        return "exact"
    if len(words) < 3:
        return "none"
    triples = list(zip(words, words[1:], words[2:], strict=False))
    known = set(zip(source, source[1:], source[2:], strict=False))
    found = sum(triple in known for triple in triples)
    return "partial" if found / len(triples) >= _MIN_PARTIAL else "none"


def _words(text: str) -> list[str]:
    # NFKC turns ligatures (ﬁ) and other compatibility characters into plain letters.
    text = unicodedata.normalize("NFKC", text).replace("\xad", "").lower()
    return _WORD.findall(text)
