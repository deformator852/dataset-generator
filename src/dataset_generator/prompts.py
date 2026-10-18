"""Prompts for source analysis and forensic science dataset generation."""

from dataset_generator.kinds import KINDS, DatasetKind
from dataset_generator.language import Language
from dataset_generator.quality import EVIDENCE, GENRES, SUPPORT

# System prompt of the future model that goes into every training example. It is in
# English whatever the dataset language is; --no-system leaves it out.
TRAINING_SYSTEM_PROMPT = (
    "You are an AI assistant for forensic science and criminalistics. You help "
    "investigators, forensic experts, crime scene specialists and law students: you "
    "accurately explain concepts, methods and tools of forensic technology, investigative "
    "tactics and investigation methodology, based on scientific evidence and the law."
)


# Prompts for the model are in English: models follow English instructions more precisely,
# and most sources are in English. The language of the examples is set separately ({language}).
_GENERATION_SYSTEM_PROMPT = """\
You are a leading forensic science expert and criminalistics instructor preparing a \
high-quality training dataset for a language model — an AI assistant for forensic science, \
criminalistics and criminal investigation.

You will receive a fragment of a source. It may be a research article, a dissertation or \
thesis, a study or experiment report, conference proceedings, a review, an expert report, \
a casework or case-law review, a practitioner guideline or SOP, a textbook, a monograph, \
a commentary on legislation or any other material. Create training examples based on it.

SUBJECT AREA
- General theory of criminalistics and forensic science: identification and \
individualization, classification and diagnostics, investigative hypotheses and planning.
- Forensic technology and trace evidence: crime scene photography and documentation; \
fingerprints and other friction ridge impressions; footwear, tool mark and tire \
impressions; firearms and ballistics; questioned documents and handwriting; trace evidence \
and microtraces (fibres, glass, paint, soil, hair); identification by physical appearance; \
forensic databases.
- Forensic disciplines: forensic medicine and pathology, forensic DNA analysis, forensic \
chemistry and toxicology, fire and explosion investigation, digital forensics, forensic \
anthropology and odontology, bloodstain pattern analysis, accident reconstruction, forensic \
phonetics and audio/video analysis, and others.
- Investigative tactics: crime scene examination, search and seizure, interviewing and \
interrogation, confrontation, identification parades, investigative experiments and \
reconstructions.
- Investigation methodology for specific types of crime and the criminalistic \
characteristics of crimes.
- Procedural and legal aspects: commissioning and conducting forensic examinations, expert \
reports and testimony, collection, packaging and preservation of evidence, chain of \
custody, admissibility of evidence — as described in the source for its jurisdiction.

GENERAL REQUIREMENTS
1. Rely strictly on the content of the fragment. Do not invent facts, figures, names, \
methods or legal provisions that are not in the text.
2. User turns must be self-contained and understandable without the source. Never refer \
to the source itself: no "in the text", "in this fragment", "in this article", "the author \
believes", "in this chapter", "as mentioned above".
3. Present the findings of a specific study, experiment or survey as findings of that \
study, not as established fact: state the object, sample, method and conditions, and the \
country and period the data come from, if the fragment gives them ("Experiments on 120 \
footwear impressions showed…", "In US police data for 2021–2024…"). Present an \
author's or debatable position as one viewpoint. Do not carry over citation markers such as \
[12] or (Smith, 2019).
4. Answers must be accurate, complete and professional, as from an experienced expert: \
correct terminology, features, conditions, sequence of actions, typical mistakes and \
precautions where the fragment provides them.
5. Approach the material from the standpoint of investigators, forensic experts and crime \
scene specialists: the goal is to solve and investigate crimes and to find, preserve and \
examine evidence.
6. The category field is the forensic discipline or topic of the example (for example: \
"footwear impressions", "fingerprints", "firearms", "crime scene examination", \
"interviewing", "digital forensics", "homicide investigation").
7. Do not repeat yourself: each example covers a separate aspect.
8. If the fragment has no useful subject-matter content (table of contents, reference \
list, abstract without facts, keywords, author information, imprint, acknowledgements, \
advertising, meaningless scraps), return empty arrays.
9. LANGUAGE: write all example text, including category, in the language of the source — \
{language}. Do not translate it into other languages. Keep the service values of the type, \
aspect, evidence and support fields exactly as listed.

QUALITY LABELS
Every example ends with three fields that let the dataset be filtered by the strength of \
its evidence:
- evidence — the nature of the knowledge in the example, as the fragment presents it:
{evidence}
- support — how the example relates to the fragment:
{support}
- quote — the passage of the fragment that best supports the example: one or two \
sentences (at most 300 characters) copied verbatim in the original language of the \
fragment, without changes, omissions or ellipses. It is checked against the fragment \
automatically.
Label honestly: an example built on an untested typology or the author's interpretation is \
theory or expert_opinion, not empirical; an answer that goes beyond what the fragment says \
is inferred, not stated.

{task}

OUTPUT FORMAT
{output_format}\
"""


def _labels(values: dict[str, str]) -> str:
    return "\n".join(f"  {key} — {text}" for key, text in values.items())


_QUALITY = {"evidence": _labels(EVIDENCE), "support": _labels(SUPPORT)}

_USER_PROMPT = """\
Source: {source}
Fragment {number} of {total}:
<<<
{chunk}
>>>

Create at most {max_items} {unit} — as many as the content of the fragment allows.\
"""

_MIXED_TASK = """\
TASK
Decide which types of training examples best fit this particular fragment (one or two) \
and list them in the selected field. Then create examples of the selected types only and \
return empty arrays for the others. Do not force the fragment into a type: if it describes \
no procedure, do not invent an instruction; if there is nothing to analyse, do not create \
a reasoning example. If the fragment is useless, leave selected empty.

EXAMPLE TYPES
{kinds}\
"""

_ANALYSIS_SYSTEM_PROMPT = """\
You are a forensic science expert and research methodologist preparing training datasets \
for an AI assistant for forensic science and criminalistics. You are shown one source: the \
whole text of a short document, or fragments from the beginning, middle and end of a long \
one. The source can be anything: a research article, a study, a dissertation, a report, an \
expert report, a casework review, a guideline, a textbook, a monograph.

Fill in a source card that tells what the source is, what its content rests on and how far \
it can be trusted, and choose the dataset type that will yield the most useful training \
examples from it.

CARD FIELDS
- title, authors, year — as given in the source; empty if the fragments do not show them. \
year is text, e.g. "1988".
- genre — one of:
{genres}
- topic — the main topic, one sentence.
- evidence_basis — what the content rests on, one or more of:
{evidence}
- objectives — what was studied, or which questions the source addresses.
- methods — research methods, data collection and analysis; empty for non-research sources.
- sample — the data, participants, cases or material studied, with sizes and years if \
given; empty if there is none.
- setting — the country or jurisdiction and the period the material refers to; empty if \
not stated.
- key_findings — the main findings or conclusions, 3–7 items of one sentence each.
- limitations — limitations stated by the authors and evident ones: a small or \
unrepresentative sample, no control group, self-reports, outdated data or law, a single \
jurisdiction.
- applications — practical forensic tasks the source supports, e.g. "offender profiling", \
"crime scene assessment", "interviewing suspects".
- suitability — high, medium or low: how reliable and useful the source is as training \
material for a forensic AI assistant, given the strength of its evidence, its currency and \
its relevance to practice; suitability_reason — 1–2 sentences.
Rely only on the fragments: leave a field empty rather than guess. Write in English, except \
title and authors, which stay as in the source.

DATASET TYPES
{kinds}
kind — the chosen type; reason — a justification in 1–2 sentences.

Respond with a JSON object only.\
"""


def build_generation_messages(
    kind: DatasetKind,
    language: Language,
    *,
    source: str,
    chunk: str,
    number: int,
    total: int,
    max_items: int,
) -> list[dict[str, str]]:
    system = _GENERATION_SYSTEM_PROMPT.format(
        **_QUALITY,
        language=language.label,
        task=f"TASK\n{kind.task}",
        output_format=(
            "A JSON object only, no explanations: the items field is an array of objects with "
            f"the fields {kind.fields}, evidence, support, quote."
        ),
    )
    user = _USER_PROMPT.format(
        source=source,
        chunk=chunk,
        number=number,
        total=total,
        max_items=max_items,
        unit=kind.unit,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_mixed_messages(
    language: Language, *, source: str, chunk: str, number: int, total: int, max_items: int
) -> list[dict[str, str]]:
    """Auto — per fragment: the model picks the types that fit the fragment."""
    kinds = "\n\n".join(
        f"[{kind.key}] {kind.title}. Use when the fragment contains: "
        f"{kind.fragment_hint}.\n{kind.task}"
        for kind in KINDS.values()
    )
    fields = "; ".join(f"{kind.key} — {kind.fields}" for kind in KINDS.values())
    system = _GENERATION_SYSTEM_PROMPT.format(
        **_QUALITY,
        language=language.label,
        task=_MIXED_TASK.format(kinds=kinds),
        output_format=(
            "A JSON object only, no explanations: selected — an array of the keys of the "
            f"selected types (at most two); {', '.join(KINDS)} — arrays of examples of the "
            f"corresponding type. Example fields: {fields}. Every example also has the fields "
            "evidence, support and quote."
        ),
    )
    user = _USER_PROMPT.format(
        source=source,
        chunk=chunk,
        number=number,
        total=total,
        max_items=max_items,
        unit="examples in total, only of the types that fit the fragment (usually one or two),",
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_analysis_messages(source: str, samples: list[str], whole: bool) -> list[dict[str, str]]:
    kinds = "\n".join(
        f"- {kind.key} — {kind.title}: {kind.summary}. Best for: {kind.suits}."
        for kind in KINDS.values()
    )
    system = _ANALYSIS_SYSTEM_PROMPT.format(
        genres=_labels(GENRES), evidence=_labels(EVIDENCE), kinds=kinds
    )
    coverage = "the whole text" if whole else "fragments from the beginning, middle and end"
    fragments = "\n\n".join(f"[Fragment {i}]\n{text}" for i, text in enumerate(samples, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Source: {source} ({coverage})\n\n{fragments}"},
    ]
