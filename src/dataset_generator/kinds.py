"""Типы датасетов: что просим у модели и как это превращается в обучающие примеры."""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import chain, zip_longest
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

AUTO = "auto"  # модель выбирает один тип для всего документа
MIXED = "mixed"  # модель выбирает типы для каждого фрагмента → несколько датасетов


@dataclass(slots=True, frozen=True)
class Example:
    """Обучающий пример в общем виде, независимо от типа и формата датасета."""

    turns: tuple[tuple[str, str], ...]  # (реплика пользователя, ответ ассистента)
    category: str
    subtype: str
    reasoning: str = ""  # рассуждение перед последним ответом (тип reasoning)
    kind: str = ""  # ключ типа датасета, в который пойдёт пример


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
class KindPrompt:
    """Описание типа для модели. Все промпты на английском: так модель лучше понимает
    задачу, а большинство источников англоязычные. Язык датасета задаётся отдельно."""

    name: str
    summary: str
    suits: str  # для каких источников подходит — для авто-анализа
    fragment_hint: str  # какой фрагмент подходит — для режима «по фрагментам»
    unit: str  # «at most N <unit>» в запросе к модели
    fields: str  # описание полей JSON
    task: str  # раздел промпта с постановкой задачи


@dataclass(slots=True, frozen=True)
class DatasetKind:
    # Поля интерфейса (на русском)
    key: str
    title: str
    icon: str
    summary: str
    suits: str
    fragment_hint: str
    # Всё, что уходит модели (на английском)
    prompt: KindPrompt
    default_items: int
    item_chars: int  # средний объём одного примера — для оценки времени
    item_model: type[BaseModel]
    convert: Callable[[Any], Example]
    sample: Example
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
            title="Вопрос — ответ",
            icon="❓",
            summary="определения, объяснения, классификации, процессуальные вопросы",
            suits="учебники, лекции, справочники, обзорные статьи, комментарии к законам",
            fragment_hint="определения, классификации, признаки, теоретические положения",
            prompt=KindPrompt(
                name="Question–answer",
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
            ),
            default_items=5,
            item_chars=600,
            item_model=QAItem,
            convert=lambda item: Example(((item.question, item.answer),), item.category, item.type),
            sample=Example(
                (
                    (
                        "Что изучает трасология и какие задачи она решает?",
                        "Трасология — отрасль криминалистической техники, изучающая "
                        "закономерности образования следов, а также средства и методы их "
                        "обнаружения, фиксации, изъятия и исследования. Её задачи — установить "
                        "механизм происшествия, идентифицировать следообразующий объект "
                        "(обувь, орудие взлома, транспортное средство) и получить данные для "
                        "розыска преступника.",
                    ),
                ),
                "трасология",
                "definition",
            ),
        ),
        DatasetKind(
            key="research",
            title="Разбор исследований",
            icon="🔬",
            summary="цель, методы, результаты, ограничения и практическое значение исследований",
            suits="научные статьи, диссертации и авторефераты, отчёты об исследованиях и "
            "экспериментах, тезисы конференций",
            fragment_hint=(
                "описание исследования или эксперимента: цель, материалы и методы, выборка, "
                "результаты, ограничения, практическая значимость"
            ),
            prompt=KindPrompt(
                name="Research analysis",
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
            ),
            default_items=4,
            item_chars=700,
            item_model=ResearchItem,
            convert=lambda item: Example(
                ((item.question, item.answer),), item.category, item.aspect
            ),
            sample=Example(
                (
                    (
                        "Как в исследованиях оценивают надёжность автоматизированного сравнения "
                        "следов обуви и что ограничивает такие результаты?",
                        "Надёжность оценивают на выборке пар следов с заранее известным "
                        "источником: считают долю верных совпадений и ложных срабатываний. "
                        "Результаты ограничены качеством и полнотой следов и однородностью "
                        "выборки: лабораторные отпечатки чище реальных, поэтому на месте "
                        "происшествия точность обычно ниже, и вывод системы требует проверки "
                        "экспертом.",
                    ),
                ),
                "трасология",
                "limitations",
            ),
        ),
        DatasetKind(
            key="cases",
            title="Ситуационные задачи",
            icon="🧩",
            summary="фабула происшествия → разбор действий следователя и эксперта",
            suits="методики расследования, материалы следственной и экспертной практики, "
            "сборники задач, обзоры судебной практики",
            fragment_hint=(
                "описание действий на месте происшествия, тактики следственных действий, "
                "методики расследования или примеры из практики"
            ),
            prompt=KindPrompt(
                name="Case problems",
                summary="an incident scenario → analysis of the investigator's and expert's "
                "actions",
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
            ),
            default_items=3,
            item_chars=1100,
            item_model=CaseItem,
            convert=lambda item: Example(
                ((f"{item.situation}\n\n{item.task}", item.solution),), item.category, "case"
            ),
            sample=Example(
                (
                    (
                        "Ночью из гаража гражданина А. похищен мотоцикл. Навесной замок на "
                        "воротах сорван, рядом на рыхлом грунте видны объёмные следы обуви.\n\n"
                        "Как зафиксировать и изъять следы обуви и какую экспертизу назначить?",
                        "1. Сфотографировать следы по правилам масштабной съёмки: оптическая ось "
                        "перпендикулярна плоскости следа, рядом — масштабная линейка.\n"
                        "2. Описать в протоколе осмотра местоположение, форму, размеры и "
                        "особенности рельефа подошвы.\n"
                        "3. Удалить случайные предметы, не нарушая рельефа, и изготовить "
                        "гипсовый слепок.\n"
                        "4. Упаковать и опечатать слепок, получить обувь проверяемых лиц для "
                        "сравнения.\n"
                        "5. Назначить трасологическую экспертизу: не оставлен ли след обувью, "
                        "изъятой у подозреваемого?",
                    ),
                ),
                "осмотр места происшествия",
                "case",
            ),
        ),
        DatasetKind(
            key="instructions",
            title="Инструкции и задания",
            icon="📋",
            summary="«Составь план…», «Перечисли признаки…» → образцовое выполнение",
            suits="методические рекомендации, наставления, инструкции, памятки, "
            "руководства по производству экспертиз",
            fragment_hint="порядок действий, алгоритмы, требования, перечни, правила оформления",
            prompt=KindPrompt(
                name="Instructions and tasks",
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
            ),
            default_items=4,
            item_chars=800,
            item_model=InstructionItem,
            convert=lambda item: Example(
                ((item.instruction, item.response),), item.category, "instruction"
            ),
            sample=Example(
                (
                    (
                        "Составь перечень вопросов эксперту при назначении дактилоскопической "
                        "экспертизы.",
                        "1. Имеются ли на представленных предметах следы рук и пригодны ли они "
                        "для идентификации личности?\n"
                        "2. Какими пальцами или участками ладони оставлены следы?\n"
                        "3. Не оставлены ли следы лицами, дактилокарты которых представлены на "
                        "экспертизу?\n"
                        "4. Не оставлены ли все следы одним и тем же лицом?",
                    ),
                ),
                "дактилоскопия",
                "instruction",
            ),
        ),
        DatasetKind(
            key="reasoning",
            title="С рассуждением (CoT)",
            icon="🧠",
            summary="вопрос → пошаговое рассуждение эксперта → вывод (для reasoning-моделей)",
            suits="монографии, экспертные заключения, аналитические и дискуссионные статьи, "
            "обзоры практики",
            fragment_hint=(
                "аргументация, анализ признаков, сравнение версий или методов, обоснование выводов"
            ),
            prompt=KindPrompt(
                name="Reasoning (chain of thought)",
                summary="a question → the expert's step-by-step reasoning → a conclusion",
                suits="monographs, expert reports, analytical and discussion papers, "
                "casework reviews",
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
            ),
            default_items=3,
            item_chars=1200,
            item_model=ReasoningItem,
            convert=lambda item: Example(
                ((item.question, item.answer),), item.category, "reasoning", item.reasoning
            ),
            sample=Example(
                (
                    (
                        "На месте происшествия обнаружена дорожка следов: шаг укорочен, угол "
                        "разворота стоп увеличен, линия ходьбы ломаная. Что можно предположить "
                        "о человеке?",
                        "Человек двигался медленно и неустойчиво — вероятно, нёс тяжёлый груз "
                        "или находился в состоянии опьянения либо утомления. Выбрать версию "
                        "помогут глубина отпечатков и сопутствующие следы.",
                    ),
                ),
                "трасология",
                "reasoning",
                "1. Укороченный шаг означает замедленное или затруднённое движение.\n"
                "2. Увеличенный угол разворота стоп и ломаная линия ходьбы указывают на "
                "неустойчивость.\n"
                "3. Такая совокупность признаков характерна для переноски груза, опьянения "
                "или утомления.\n"
                "4. Различить версии можно по вдавленности следов и следам волочения.",
            ),
        ),
        DatasetKind(
            key="dialogue",
            title="Диалоги",
            icon="💬",
            summary="многоходовая беседа: уточняющие вопросы пользователя и ответы ассистента",
            suits="обзорные и научно-популярные тексты, лекции, учебные пособия с "
            "последовательным изложением темы",
            fragment_hint=(
                "последовательное объяснение темы, которое естественно раскрывать уточняющими "
                "вопросами"
            ),
            prompt=KindPrompt(
                name="Dialogues",
                summary="a multi-turn conversation: the user's follow-up questions and the "
                "assistant's answers",
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
            ),
            default_items=2,
            item_chars=1800,
            item_model=DialogueItem,
            convert=lambda item: Example(
                tuple((turn.user, turn.assistant) for turn in item.turns), item.category, "dialogue"
            ),
            sample=Example(
                (
                    (
                        "Чем объёмный след отличается от поверхностного?",
                        "Объёмный след возникает, когда объект деформирует мягкую поверхность — "
                        "грунт, снег, пластилин — и оставляет трёхмерный отпечаток. "
                        "Поверхностный остаётся на твёрдой поверхности за счёт наслоения или "
                        "отслоения вещества.",
                    ),
                    (
                        "А как изъять объёмный след обуви на снегу?",
                        "После масштабной фотосъёмки след опыляют тальком или тонким слоем "
                        "сухого гипса, чтобы снизить нагрев при схватывании, заливают гипсовым "
                        "раствором и после затвердения извлекают слепок.",
                    ),
                ),
                "трасология",
                "dialogue",
            ),
        ),
    )
}

AUTO_TITLE = "Авто — по документу"
AUTO_ICON = "🔎"
AUTO_SUMMARY = (
    "Один датасет. Модель прочитает фрагменты из начала, середины и конца документа, "
    "определит его жанр (статья, исследование, учебник, методичка…) и выберет один тип "
    "датасета для всего документа."
)

MIXED_TITLE = "Авто — по фрагментам"
MIXED_ICON = "🔀"
MIXED_SUMMARY = (
    "Несколько датасетов. Для каждого фрагмента модель сама решает, какие типы примеров ему "
    "подходят (определения → вопрос-ответ, порядок действий → инструкции, практика → "
    "ситуационные задачи…), и раскладывает примеры по отдельным файлам — по одному на тип."
)
MIXED_DEFAULT_ITEMS = 5
MIXED_MAX_KINDS = 2
# Поле selected идёт первым: модель сначала выбирает типы, потом пишет примеры. Схема
# ограничивает выбор двумя типами, а примеры невыбранных типов программа отбрасывает.
MIXED_RESULT = create_model(
    "MixedResult",
    # maxItems — только подсказка для грамматики Ollama; лишнее обрезаем сами, а не
    # роняем ответ валидацией.
    selected=(
        list[Literal[tuple(KINDS)]],  # type: ignore[valid-type]
        Field(json_schema_extra={"maxItems": MIXED_MAX_KINDS}),
    ),
    **{key: (list[kind.item_model], ...) for key, kind in KINDS.items()},  # type: ignore[call-overload]
)


def mixed_examples(result: BaseModel, limit: int) -> list[Example]:
    """Примеры выбранных моделью типов, по очереди из каждого типа, не больше limit."""
    selected = set(result.selected[:MIXED_MAX_KINDS])  # type: ignore[attr-defined]
    per_kind = [
        [kind.make_example(item) for item in getattr(result, key)]
        for key, kind in KINDS.items()
        if key in selected
    ]
    interleaved = chain.from_iterable(zip_longest(*per_kind))
    return [example for example in interleaved if example is not None][:limit]


def run_title(run: str) -> str:
    """Название режима: конкретного типа или одного из авто-режимов."""
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
    """Сколько текста примерно сгенерирует модель на один фрагмент."""
    if run in KINDS:
        item_chars = KINDS[run].item_chars
    else:
        item_chars = sum(kind.item_chars for kind in KINDS.values()) // len(KINDS)
    # Модель не всегда выдаёт максимум примеров — в среднем около 80%.
    return int(item_chars * max_items * 0.8)
