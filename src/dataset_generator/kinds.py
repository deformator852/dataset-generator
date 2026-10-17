"""Dataset types: what the model is asked for and how it becomes training examples."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import chain, zip_longest
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

AUTO = "auto"  # the model picks one type for the whole document
MIXED = "mixed"  # the model picks types for each fragment → several datasets


@dataclass(slots=True, frozen=True)
class Example:
    """A training example in a generic form, independent of dataset type and format."""

    turns: tuple[tuple[str, str], ...]  # (user turn, assistant answer)
    category: str
    subtype: str
    reasoning: str = ""  # reasoning before the last answer (the reasoning type)
    kind: str = ""  # key of the dataset type the example goes to


class QAItem(BaseModel):
    question: str
    answer: str
    category: str
    type: Literal["definition", "explanation", "procedure", "comparison", "legal"]


class ResearchItem(BaseModel):
    question: str
    answer: str
    category: str
    aspect: Literal["background", "method", "results", "limitations", "application"]


class CaseItem(BaseModel):
    situation: str
    task: str
    solution: str
    category: str


class InstructionItem(BaseModel):
    instruction: str
    response: str
    category: str


class ReasoningItem(BaseModel):
    question: str
    reasoning: str
    answer: str
    category: str


class DialogueTurn(BaseModel):
    user: str
    assistant: str


class DialogueItem(BaseModel):
    category: str
    turns: list[DialogueTurn]


@dataclass(slots=True, frozen=True)
class DatasetKind:
    """A dataset type. The texts are used both in the interface and in the model prompt."""

    key: str
    title: str
    icon: str
    summary: str
    suits: str  # which sources it suits; shown to the user and used by auto analysis
    fragment_hint: str  # which fragments it suits; used by "Auto — per fragment"
    unit: str  # "at most N <unit>" in the request to the model
    fields: str  # description of the JSON fields for the prompt
    task: str  # the prompt section with the task
    default_items: int
    item_chars: int  # average size of one example, for estimates
    item_model: type[BaseModel]
    convert: Callable[[Any], Example]
    sample: Example  # example for the type picker
    result_model: type[BaseModel] = field(init=False)

    def __post_init__(self) -> None:
        result = create_model(
            f"{self.item_model.__name__}Result", items=(list[self.item_model], ...)
        )
        object.__setattr__(self, "result_model", result)

    def make_example(self, item: BaseModel) -> Example:
        return replace(self.convert(item), kind=self.key)


KINDS: dict[str, DatasetKind] = {
    kind.key: kind
    for kind in (
        DatasetKind(
            key="qa",
            title="Question–answer",
            icon="❓",
            summary="definitions, explanations, classifications, procedures, legal questions",
            suits="textbooks, lecture notes, reference works, review articles, "
            "commentaries on legislation",
            fragment_hint="definitions, classifications, characteristic features, "
            "theoretical principles",
            unit="question–answer pairs",
            fields="question, answer, category, type",
            task="""\
Create question–answer pairs based on the fragment. Vary the type field:
- definition — the definition of a concept or term;
- explanation — essence, principle, scientific basis, classification;
- procedure — a sequence of actions: detection, documentation, collection, packaging, \
examination;
- comparison — a comparison of methods, types of traces, examinations, tactical techniques;
- legal — procedural and legal aspects.
An answer is usually 3–8 sentences; for procedures, a numbered list of steps.""",
            default_items=5,
            item_chars=600,
            item_model=QAItem,
            convert=lambda item: Example(((item.question, item.answer),), item.category, item.type),
            sample=Example(
                (
                    (
                        "What does the forensic study of footwear impressions cover, and what "
                        "questions can it answer?",
                        "Footwear impression examination studies how shoe soles leave marks, "
                        "and how those marks are found, documented, recovered and compared. It "
                        "can help reconstruct events at the scene, link an impression to a "
                        "specific shoe through class and individual characteristics, and give "
                        "investigators leads on the offender's movements.",
                    ),
                ),
                "footwear impressions",
                "definition",
            ),
        ),
        DatasetKind(
            key="research",
            title="Research analysis",
            icon="🔬",
            summary="aims, methods, results, limitations and practical value of studies",
            suits="research articles, dissertations and theses, study and experiment "
            "reports, conference papers",
            fragment_hint="a description of a study or experiment: aim, materials and "
            "methods, sample, results, limitations, practical significance",
            unit="questions about the research",
            fields="question, answer, category, aspect",
            task="""\
Create questions and answers that teach how to understand research in forensic science \
and criminalistics. Vary the aspect field:
- background — what problem the study addresses and why it matters for practice;
- method — objects, materials, sample, methods, equipment and experimental conditions;
- results — the findings and conclusions, with figures if the fragment gives them;
- limitations — limitations, conditions of applicability, sources of error;
- application — how the findings can be used in forensic casework and investigations.
A question names the subject of the research ("studies of footwear identification from \
impressions in snow"), not "this study". An answer is 3–6 sentences.""",
            default_items=4,
            item_chars=700,
            item_model=ResearchItem,
            convert=lambda item: Example(
                ((item.question, item.answer),), item.category, item.aspect
            ),
            sample=Example(
                (
                    (
                        "How do studies assess the reliability of automated footwear "
                        "impression comparison, and what limits their results?",
                        "Reliability is assessed on a set of impression pairs with known "
                        "sources: researchers count correct matches and false positives. The "
                        "results are limited by the quality and completeness of the impressions "
                        "and by how representative the sample is: laboratory prints are cleaner "
                        "than crime scene marks, so accuracy in casework is usually lower and "
                        "the system's output must be verified by an examiner.",
                    ),
                ),
                "footwear impressions",
                "limitations",
            ),
        ),
        DatasetKind(
            key="cases",
            title="Case problems",
            icon="🧩",
            summary="an incident scenario → analysis of the investigator's and expert's actions",
            suits="investigation methodology, investigative and forensic casework "
            "materials, collections of case problems, case-law reviews",
            fragment_hint="actions at a crime scene, tactics of investigative actions, "
            "investigation methodology or examples from practice",
            unit="case problems",
            fields="situation, task, solution, category",
            task="""\
Create case problems whose solution follows from the fragment.
- situation — a short realistic scenario (2–5 sentences): what happened, what was found \
at the scene, what the investigation knows. The scenario may be fictional; refer to \
people generically (Mr. A., victim B.).
- task — the assignment: which actions to take, which traces to look for, how to document \
and collect them, which examination to commission and which questions to put to the \
expert, how to plan an interview.
- solution — a detailed expert solution strictly based on the fragment: the sequence of \
actions with justification, typical mistakes.""",
            default_items=3,
            item_chars=1100,
            item_model=CaseItem,
            convert=lambda item: Example(
                ((f"{item.situation}\n\n{item.task}", item.solution),), item.category, "case"
            ),
            sample=Example(
                (
                    (
                        "A motorcycle was stolen overnight from Mr. A.'s garage. The padlock "
                        "on the door was forced, and three-dimensional footwear impressions are "
                        "visible in the soft soil nearby.\n\n"
                        "How should the impressions be documented and recovered, and which "
                        "examination should be requested?",
                        "1. Photograph each impression with a scale, the camera axis "
                        "perpendicular to the impression.\n"
                        "2. Record the location, shape, dimensions and visible sole pattern in "
                        "the scene notes.\n"
                        "3. Remove loose debris without disturbing the impression and cast it "
                        "with dental stone.\n"
                        "4. Package and seal the cast; obtain suspects' footwear for comparison.\n"
                        "5. Request a footwear examination: could the impression have been made "
                        "by the shoes seized from the suspect?",
                    ),
                ),
                "crime scene examination",
                "case",
            ),
        ),
        DatasetKind(
            key="instructions",
            title="Instructions and tasks",
            icon="📋",
            summary='"Draw up a plan…", "List the features…" → an exemplary response',
            suits="guidelines, manuals, SOPs, protocols, handbooks on conducting "
            "forensic examinations",
            fragment_hint="sequences of actions, algorithms, requirements, checklists, "
            "documentation rules",
            unit="instruction tasks",
            fields="instruction, response, category",
            task="""\
Create tasks that an investigator, forensic expert or student might give to a forensic \
AI assistant, together with an exemplary response to each.
- instruction — an imperative request: "Draw up a list of…", "Describe the procedure \
for…", "List the features of…", "Draw up a crime scene examination plan for…", \
"Formulate questions for the expert…", "Explain in simple terms…".
- response — a high-quality response strictly based on the fragment, well structured \
(lists, steps).""",
            default_items=4,
            item_chars=800,
            item_model=InstructionItem,
            convert=lambda item: Example(
                ((item.instruction, item.response),), item.category, "instruction"
            ),
            sample=Example(
                (
                    (
                        "Draw up a list of questions for the expert when requesting a "
                        "fingerprint examination.",
                        "1. Are there fingermarks on the submitted items, and are they suitable "
                        "for identification?\n"
                        "2. Which fingers or areas of the palm left the marks?\n"
                        "3. Were the marks left by the persons whose reference prints are "
                        "submitted?\n"
                        "4. Were all the marks left by the same person?",
                    ),
                ),
                "fingerprints",
                "instruction",
            ),
        ),
        DatasetKind(
            key="reasoning",
            title="Reasoning (CoT)",
            icon="🧠",
            summary="a question → the expert's step-by-step reasoning → a conclusion",
            suits="monographs, expert reports, analytical and discussion papers, casework reviews",
            fragment_hint="argumentation, analysis of features, comparison of hypotheses "
            "or methods, justification of conclusions",
            unit="reasoning questions",
            fields="question, reasoning, answer, category",
            task="""\
Create questions that require analysis rather than restating a definition: why a method \
works, what conclusion follows from a combination of features, which hypothesis to put \
forward, how to assess the reliability of a result, what to choose under specific \
conditions.
- reasoning — the expert's step-by-step reasoning (3–7 steps) from the facts in the \
fragment to the conclusion.
- answer — the final answer (2–4 sentences).""",
            default_items=3,
            item_chars=1200,
            item_model=ReasoningItem,
            convert=lambda item: Example(
                ((item.question, item.answer),), item.category, "reasoning", item.reasoning
            ),
            sample=Example(
                (
                    (
                        "A trail of footprints at the scene shows shortened steps, an increased "
                        "foot angle and an irregular walking line. What can be inferred about "
                        "the person?",
                        "The person moved slowly and unsteadily — probably carrying a heavy "
                        "load or intoxicated or exhausted. The depth of the impressions and "
                        "accompanying traces help choose between these hypotheses.",
                    ),
                ),
                "footwear impressions",
                "reasoning",
                "1. Shortened steps indicate slow or impeded movement.\n"
                "2. An increased foot angle and an irregular walking line indicate "
                "instability.\n"
                "3. This combination is typical of carrying a load, intoxication or fatigue.\n"
                "4. The hypotheses can be distinguished by impression depth and drag marks.",
            ),
        ),
        DatasetKind(
            key="dialogue",
            title="Dialogues",
            icon="💬",
            summary="a multi-turn conversation: follow-up questions and the assistant's answers",
            suits="review and popular-science texts, lectures, study guides that present "
            "a topic step by step",
            fragment_hint="a step-by-step explanation of a topic that naturally unfolds "
            "through follow-up questions",
            unit="dialogues",
            fields="category, turns (an array of objects with the fields user and assistant)",
            task="""\
Create dialogues between a user (an investigator, forensic expert or student) and a \
forensic AI assistant. Each dialogue has 2–4 exchanges (turns): each next user question \
clarifies or builds on the previous answer. The assistant's answers are professional and \
strictly based on the fragment.""",
            default_items=2,
            item_chars=1800,
            item_model=DialogueItem,
            convert=lambda item: Example(
                tuple((turn.user, turn.assistant) for turn in item.turns), item.category, "dialogue"
            ),
            sample=Example(
                (
                    (
                        "What is the difference between a three-dimensional and a "
                        "two-dimensional impression?",
                        "A three-dimensional impression forms when an object deforms a soft "
                        "surface such as soil, snow or clay. A two-dimensional impression is "
                        "left on a hard surface when material is deposited or removed.",
                    ),
                    (
                        "So how do you recover a footwear impression in snow?",
                        "After photographing it with a scale, spray the impression with a "
                        "snow wax or dust it lightly with dry casting material to limit heat "
                        "from setting, then cast it and lift the cast once it has hardened.",
                    ),
                ),
                "footwear impressions",
                "dialogue",
            ),
        ),
    )
}

AUTO_TITLE = "Auto — per document"
AUTO_ICON = "🔎"
AUTO_SUMMARY = (
    "One dataset. The model reads fragments from the beginning, middle and end of the "
    "document, identifies its genre (article, study, textbook, guideline…) and picks one "
    "dataset type for the whole document."
)

MIXED_TITLE = "Auto — per fragment"
MIXED_ICON = "🔀"
MIXED_SUMMARY = (
    "Several datasets. For each fragment the model decides which example types fit it "
    "(definitions → question–answer, procedures → instructions, casework → case problems…) "
    "and sorts the examples into separate files, one per type."
)
MIXED_DEFAULT_ITEMS = 3
MIXED_MAX_KINDS = 2
# The selected field comes first: the model picks the types, then writes examples. The schema
# limits the choice to two types, and examples of unselected types are discarded.
MIXED_RESULT = create_model(
    "MixedResult",
    # maxItems is only a hint for the model; extra items are cut off here instead of
    # failing validation.
    selected=(
        list[Literal[tuple(KINDS)]],  # type: ignore[valid-type]
        Field(json_schema_extra={"maxItems": MIXED_MAX_KINDS}),
    ),
    **{key: (list[kind.item_model], ...) for key, kind in KINDS.items()},  # type: ignore[call-overload]
)


def mixed_examples(result: BaseModel, limit: int) -> list[Example]:
    """Examples of the types the model selected, alternating between types, at most limit."""
    selected = set(result.selected[:MIXED_MAX_KINDS])  # type: ignore[attr-defined]
    per_kind = [
        [kind.make_example(item) for item in getattr(result, key)]
        for key, kind in KINDS.items()
        if key in selected
    ]
    interleaved = chain.from_iterable(zip_longest(*per_kind))
    return [example for example in interleaved if example is not None][:limit]


def run_title(run: str) -> str:
    """Title of a mode: a specific type or one of the auto modes."""
    if run == AUTO:
        return AUTO_TITLE
    if run == MIXED:
        return MIXED_TITLE
    return KINDS[run].title


def max_items_for(run: str, override: int | None) -> int:
    if override:
        return override
    return MIXED_DEFAULT_ITEMS if run == MIXED else KINDS[run].default_items


def expected_output_chars(run: str, max_items: int) -> int:
    """Roughly how much text the model generates per fragment."""
    if run in KINDS:
        item_chars = KINDS[run].item_chars
    else:
        item_chars = sum(kind.item_chars for kind in KINDS.values()) // len(KINDS)
    # The model does not always produce the maximum number of examples; about 80% on average.
    return int(item_chars * max_items * 0.8)
