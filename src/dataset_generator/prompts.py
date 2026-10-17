"""Промпты для анализа источника и генерации датасета по криминалистике (на английском)."""

from dataset_generator.kinds import KINDS, DatasetKind
from dataset_generator.language import Language

# System-промпт будущей модели, который попадает в обучающие примеры, — на языке датасета.
TRAINING_SYSTEM_PROMPTS = {
    "ru": (
        "Ты — ИИ-ассистент по криминалистике и судебной экспертизе. Ты помогаешь следователям, "
        "экспертам-криминалистам, специалистам и студентам юридических вузов: точно объясняешь "
        "понятия, методы и средства криминалистической техники, тактику следственных действий "
        "и методику расследования, опираясь на научные данные и требования закона."
    ),
    "uk": (
        "Ти — ШІ-асистент із криміналістики та судової експертизи. Ти допомагаєш слідчим, "
        "експертам-криміналістам, фахівцям і студентам юридичних вишів: точно пояснюєш поняття, "
        "методи й засоби криміналістичної техніки, тактику слідчих дій і методику "
        "розслідування, спираючись на наукові дані та вимоги закону."
    ),
    "en": (
        "You are an AI assistant for forensic science and criminalistics. You help "
        "investigators, forensic experts, crime scene specialists and law students: you "
        "accurately explain concepts, methods and tools of forensic technology, investigative "
        "tactics and investigation methodology, based on scientific evidence and the law."
    ),
}


def training_system_prompt(language: Language) -> str:
    return TRAINING_SYSTEM_PROMPTS.get(language.code, TRAINING_SYSTEM_PROMPTS["en"])


# Промпты для модели — на английском: Qwen3 точнее следует английским инструкциям, а
# большинство источников англоязычные. Язык самих примеров задаётся отдельно ({language}).
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
study, not as established fact: state the object, sample, method and conditions if the \
fragment gives them ("Experiments on 120 footwear impressions showed…"). Present an \
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
{language}. Do not translate it into other languages. Keep the service values of the type \
and aspect fields exactly as listed.

{task}

OUTPUT FORMAT
{output_format}\
"""

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
You are a forensic science expert and instructional designer preparing training datasets \
for an AI assistant for forensic science and criminalistics. You are shown several \
fragments of one source (from its beginning, middle and end; for a short document — the \
whole text). The source can be anything: a research article, a study, a dissertation, a \
report, an expert report, a casework review, a guideline, a textbook. Determine what kind \
of source it is and choose the dataset type that will yield the highest-quality and most \
useful training examples from it.

DATASET TYPES
{kinds}

Response fields: genre — the genre of the source (research article, dissertation, study \
report, expert report, casework review, guideline, textbook, monograph, etc.); topic — the \
main topic, briefly; kind — the chosen type; reason — a justification in 1–2 sentences. \
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
        language=language.label,
        task=f"TASK\n{kind.task}",
        output_format=(
            "A JSON object only, no explanations: the items field is an array of objects with "
            f"the fields {kind.fields}."
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
    """Режим «по фрагментам»: модель сама выбирает подходящие фрагменту типы."""
    kinds = "\n\n".join(
        f"[{kind.key}] {kind.title}. Use when the fragment contains: "
        f"{kind.fragment_hint}.\n{kind.task}"
        for kind in KINDS.values()
    )
    fields = "; ".join(f"{kind.key} — {kind.fields}" for kind in KINDS.values())
    system = _GENERATION_SYSTEM_PROMPT.format(
        language=language.label,
        task=_MIXED_TASK.format(kinds=kinds),
        output_format=(
            "A JSON object only, no explanations: selected — an array of the keys of the "
            f"selected types (at most two); {', '.join(KINDS)} — arrays of examples of the "
            f"corresponding type. Example fields: {fields}."
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


def build_analysis_messages(source: str, samples: list[str]) -> list[dict[str, str]]:
    kinds = "\n".join(
        f"- {kind.key} — {kind.title}: {kind.summary}. Best for: {kind.suits}."
        for kind in KINDS.values()
    )
    fragments = "\n\n".join(f"[Fragment {i}]\n{text}" for i, text in enumerate(samples, 1))
    return [
        {"role": "system", "content": _ANALYSIS_SYSTEM_PROMPT.format(kinds=kinds)},
        {
            "role": "user",
            "content": f"Source: {source}\n\n{fragments}",
        },
    ]
