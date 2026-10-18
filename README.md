# Dataset Generator — forensic science datasets

The program turns documents on forensic science into training datasets for an AI assistant for
criminalistics and forensic examination. The source can be anything: research articles, studies,
dissertations and theses, reports, expert reports, casework reviews, guidelines, textbooks,
books — in PDF, DOCX, EPUB, FB2, DjVu, TXT, HTML and other formats.

The text is processed by **DeepSeek V4.1-Flash through the API**: a 300-page book takes about
10–20 minutes and costs ~$0.1–0.3.

Features:

- a terminal interface (TUI) that shows in real time how the model "thinks";
- several dataset types to choose from, or the model picks the type itself;
- a separate dataset type for analysing research studies;
- the dataset language follows the language of the source;
- a source card for every file (genre, methods, sample, findings, limitations, suitability)
  and quality labels on every example: what its knowledge rests on, and whether its supporting
  quote really is in the source;
- reference lists and running heads (journal name, DOI, page numbers) are removed
  automatically, so the model spends no time or money on them;
- parallel API requests, time and cost estimates, and spending tracking;
- resuming from where processing stopped, protection against duplicates;
- `chatml`, `alpaca` and `sharegpt` record formats in JSON Lines.

## Installation

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # project dependencies
```

Sign up at [platform.deepseek.com](https://platform.deepseek.com), top up your balance (prepaid;
a book costs cents) and create an API key. Store it in an environment variable — for example,
add this line to `~/.zshrc`:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

DjVu files additionally need `djvulibre` (`sudo dnf install djvulibre`).

## Quick start

1. Put your documents into the `input/` folder.
2. Run `uv run dataset-generator`.
3. Type a file name (or a path to a file or folder) and press Enter.
4. Choose a dataset type in the window that opens and press "Start".
5. The finished datasets appear in the `datasets/` folder.

## Model and cost

**The model is `deepseek-flash` (V4.1-Flash).** It is more than enough for examples based on a
given text, while `deepseek-v4-pro` costs about 4 times as much (`-m deepseek-v4-pro` if you need
maximum accuracy). Thinking is on with `low` depth: it costs very little, and the model picks
types more accurately and makes fewer mistakes. Fragments are sent 16 at a time. The source text
is sent to DeepSeek's servers.

**DeepSeek pricing** (per 1M tokens, peak / off-peak): Flash — input $0.30 / $0.15, cached input
$0.006 / $0.003, output $1.20 / $0.60. Peak hours are Mon–Fri 01:00–04:00 and 06:00–10:00 UTC
(04:00–07:00 and 09:00–13:00 Moscow time); at all other times and on weekends everything is half
price. The price table is in `pricing.py`; update it if DeepSeek changes its prices.

**Using model outputs for training.** As far as is known, DeepSeek's terms allow using model
outputs to train other models. Check the current terms before publishing the dataset or using it
commercially.

## Interface

```
┌ input field: file name or path ──────────────────────────────────────────┐
├ Settings ──────────┬ 🧠 Model thinking (live stream) ────────────────────┤
├ Queue ─────────────┤                                                     │
├ progress bar ──────┼ 📝 Model response (JSON) ───────────────────────────┤
├ Progress and time ─┤                                                     │
├ Model and cost ────┼ 📚 Dataset and log: finished examples, events ──────┤
└────────────────────┴─────────────────────────────────────────────────────┘
```

The thinking and response streams are shown for one fragment ("+15 in parallel" is how many more
are being processed in the background); finished examples of all fragments go to the log. The
"Model and cost" panel shows the model, the current tariff (peak or off-peak) and how much has
been spent in this session.

| Key | Action |
|---|---|
| Enter | add a file or folder to the queue (an empty Enter restarts the queue) |
| → | accept path autocompletion |
| ↑ ↓ / Tab / Enter / Esc | choose the type and format in the picker |
| F2 | turn model thinking on/off (from the next fragment) |
| F3 | stop processing (progress is saved) |
| F4 | clear the panels |
| Ctrl+Q | quit |

Files can be added while processing is running — they join the queue, each with its own type.

## Dataset types

The picker shows an example for each type on the right, along with what a dataset line looks
like.

| Type | Key | What you get | Best for |
|---|---|---|---|
| 🔀 **Auto — per fragment** (recommended) | `mixed` | for each fragment the model picks 1–2 suitable types; examples are sorted into separate files, one per type | any documents, especially mixed ones |
| 🔎 Auto — per document | `auto` | the model reads the beginning, middle and end of the document (the whole text for a short article), identifies its genre and picks one type for the whole document | uniform documents |
| ❓ Question–answer | `qa` | definitions, explanations, classifications, procedures, legal questions | textbooks, lectures, reference works, review articles |
| 🔬 Research analysis | `research` | aim, methods and sample, results, limitations and practical value of a study | research articles, dissertations, experiment reports |
| 🧩 Case problems | `cases` | an incident scenario → analysis of the investigator's and expert's actions | investigation methodology, casework |
| 📋 Instructions and tasks | `instructions` | "Draw up a plan…", "List the features…" → an exemplary response | guidelines, manuals, SOPs |
| 🧠 Reasoning (CoT) | `reasoning` | a question → step-by-step reasoning in `<think>…</think>` → a conclusion | monographs, expert reports, analytical papers |
| 💬 Dialogues | `dialogue` | a conversation of 2–4 exchanges with follow-up questions | review texts, study guides |

Before the fragments, the model reads the beginning, middle and end of every document (the whole
text of a short one) and fills in its [source card](#source-card). "Auto — per document" takes
the dataset type the card recommends. The card is reused: the document is not analysed again on
the next run.

**Language.** Prompts for the model are written in English. The language of the examples is
detected automatically from the source text: an English article gives an English dataset, a
Russian one gives a Russian dataset. The system prompt in the examples is always in English
(`--no-system` leaves it out). To set the language manually: `--language en`.

**Research materials.** Findings of a specific study go into the dataset as findings of that
study — with its object, sample and method — rather than as established facts. An author's
position is presented as one viewpoint, and citation markers like `[12]` are not carried over.
Abstracts, keywords, author information and acknowledgements are skipped by the model.

**Response checks.** DeepSeek has no strict JSON schema: the schema is passed in the prompt, and
the program checks every response and retries the request on errors (the last attempt runs
without thinking).

## Time and cost

Before the first fragment, the log shows a time and cost estimate based on the amount of text,
the tariff (including peak hours) and the number of parallel requests. The "Progress and time"
panel shows how much has been spent so far ("Cost: $0.04 of ≈ $0.21"), including prompt caching
and failed attempts; "Time left" is refined from the actual time per fragment.

## Output

All files are written to `datasets/` in JSON Lines format (one line is one example):

| Mode | Files |
|---|---|
| a specific type or "Auto — per document" | `<file>.<type>.jsonl`, e.g. `article.pdf.research.jsonl` |
| "Auto — per fragment" | `<file>.mixed.<type>.jsonl` — one file per type the model picked |

Next to them are `<file>.source.json`, the [source card](#source-card), and `*.state.json`, a
service file used for resuming. With parallel processing, examples are written in the order
fragments finish; the fragment index is in `metadata.chunk`.

The record format is chosen in the picker or with `--format`. A `chatml` example (the default):

```json
{"messages": [
  {"role": "system", "content": "You are an AI assistant for forensic science…"},
  {"role": "user", "content": "How are three-dimensional footwear impressions in snow recovered?"},
  {"role": "assistant", "content": "1. Photograph the impression with a scale…"}
 ],
 "metadata": {"source": "snow.pdf", "source_id": "3f9a1c0d2b7e4a65", "chunk": 3, "kind": "qa",
              "category": "footwear impressions", "type": "procedure",
              "evidence": "empirical", "support": "stated",
              "quote": "Impressions in fresh snow were cast after spraying with snow wax…",
              "quote_match": "exact", "language": "en", "model": "deepseek-flash"}}
```

- `alpaca` — `instruction`, `input`, `output`, `system`, plus `history` for dialogues;
- `sharegpt` — `conversations` made of `system` / `human` / `gpt` turns.

These files work with Hugging Face `datasets`, TRL, Unsloth, Axolotl and LLaMA-Factory:

```bash
cat datasets/*.jsonl > forensics.jsonl          # everything together
cat datasets/*.qa.jsonl > forensics-qa.jsonl    # only question–answer from all files
```

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="forensics.jsonl", split="train")
```

### Quality labels

Only `messages` is trained on. `metadata` says where each example comes from and how far it can
be trusted, so the dataset can be filtered and balanced:

| Field | Values |
|---|---|
| `evidence` | what the knowledge rests on: `empirical` (results of a specific study), `review` (a synthesis of studies), `case` (one case from casework), `theory` (a hypothesis, model or typology not presented as tested), `expert_opinion` (the author's position or experience without data), `established` (definitions, textbook knowledge), `normative` (law, standards, official procedures) |
| `support` | how the example relates to the fragment: `stated` (everything is in the text), `inferred` (the model combined or generalised what the text says), `constructed` (a hypothetical scenario that applies the material, as in case problems) |
| `quote` | the passage of the fragment that supports the example, copied by the model |
| `quote_match` | the program's check of that passage against the fragment: `exact`, `partial` (lightly edited) or `none` (not in the source: the example may be ungrounded) |

The model sets `evidence` and `support`, so they are its judgement; `quote_match` is computed by
the program and does not depend on the model. Unknown label values are stored as `""`.
`model` only says who wrote the example, not that it is correct.

The [fine-tuning notebook](#fine-tuning) skips examples with `quote_match: none` by default. To
pick examples yourself, for instance only empirical findings with a found quote:

```bash
jq -c 'select(.metadata.evidence == "empirical" and .metadata.quote_match != "none")' \
  datasets/*.jsonl > forensics-empirical.jsonl
```

### Source card

`<file>.source.json` describes the whole source; examples link to it by `source_id`. The model
fills in the description once per file:

- `title`, `authors`, `year`, `genre` (`research_article`, `review`, `meta_analysis`, `thesis`,
  `monograph`, `textbook`, `guideline`, `report`, `expert_report`, `case_report`, `legal_text`,
  `other`) and `topic`;
- `evidence_basis` — what the content rests on, with the same values as `evidence` above;
- `objectives`, `methods`, `sample` (data, participants, cases and their size), `setting`
  (jurisdiction and period);
- `key_findings` and `limitations` (stated by the authors and evident ones: small sample, no
  control group, outdated data);
- `applications` — forensic tasks the source supports (offender profiling, crime scene
  assessment…);
- `suitability` (`high` / `medium` / `low`) and `suitability_reason` — how reliable and useful
  the source is as training material;
- `recommended_kind` and `kind_reason` — the dataset type that suits it best.

`analysis.based_on` says what the description rests on: the whole text or fragments of it. The
program adds `datasets`: for every dataset made from the file, the number of examples, the counts
of each `evidence`, `support` and `quote_match` value, and the top categories. It is updated at
the end of every run.

**Resuming.** If processing was interrupted (F3, Ctrl+Q, Ctrl+C, the PC shut down, the balance
ran out), add the same file with the same type again — processing continues where it stopped,
and fragments that are already paid for are not sent again. If the file itself, the chunking,
format or language settings, or the record layout of a newer program version changed, the old
dataset is not deleted but renamed to `*.jsonl.bak-<date>`.

## Command line

```
uv run dataset-generator [FILES ...] [OPTIONS]
```

`FILES` are file names (looked up in the current folder, then in `input/`) or folders (all
supported files inside are used). Without files, an empty TUI opens where you type a file name.
Without `-k`, the dataset type is asked for: in the TUI with the picker, in `--no-tui` mode with a
terminal menu (with no input available, for example in a script, `mixed` is used). Put paths with
spaces and parentheses in single quotes: `'…/Book (1995).pdf'`.

Examples:

```bash
uv run dataset-generator                                # TUI, everything by hand
uv run dataset-generator article.pdf                    # TUI, asks for the dataset type
uv run dataset-generator -k mixed input/                # whole folder, the model picks the types
uv run dataset-generator -k research input/articles/    # research analysis from articles
uv run dataset-generator -k qa -f alpaca book.fb2.zip   # a specific type and format
uv run dataset-generator --no-tui -k mixed input/       # no TUI: per-fragment results in the terminal
uv run dataset-generator --no-think -k mixed book.pdf   # no thinking: even cheaper and faster
uv run dataset-generator -m deepseek-v4-pro book.pdf    # DeepSeek's strongest model
uv run dataset-generator --help                         # help on all options
```

| Option | Default | Purpose |
|---|---|---|
| `-m, --model` | `deepseek-flash` | DeepSeek model: `deepseek-flash` or `deepseek-v4-pro` |
| `-k, --kind` | ask | `mixed`, `auto`, `qa`, `research`, `cases`, `instructions`, `reasoning`, `dialogue` |
| `-f, --format` | `chatml` | record format: `chatml`, `alpaca`, `sharegpt` |
| `--language` | `auto` | dataset language: `auto` (same as the source) or a code (`ru`, `en`, `uk`…) |
| `--no-system` | — | do not add a system prompt to the examples |
| `--think` / `--no-think` | on | model thinking mode (`--no-think` is cheaper and faster) |
| `--effort` | `low` | thinking depth: `low`, `high`, `max` (more expensive and slower) |
| `-j, --parallel` | `16` | how many fragments to send to the API at once |
| `--api-url` | `https://api.deepseek.com` | API address (any OpenAI-compatible API works) |
| `-i, --input-dir` | `input` | where to look for files by name |
| `-o, --output-dir` | `datasets` | where to write datasets |
| `--chunk-size` | `3000` | text fragment size, characters |
| `--overlap` | `300` | overlap between neighbouring fragments, characters |
| `-n, --max-items` | depends on the type | max examples per fragment (qa — 5, research and instructions — 4, mixed, cases and reasoning — 3, dialogue — 2) |
| `--num-predict` | `8192` | generation limit per fragment (doubled with thinking) |
| `--retries` | `2` | retries on errors (the last attempt runs without thinking) |
| `--no-tui` | — | plain terminal output instead of the TUI (handy for background runs) |

The API key is read from the `DEEPSEEK_API_KEY` environment variable.

## Supported files

PDF, EPUB, MOBI, FB2 (including `.fb2.zip`), DjVu, DOCX, TXT/MD (the encoding is detected
automatically, including cp1251), HTML (for example, saved web pages of articles), XPS, CBZ.
Scans without a text layer must be OCR'd first, for example with `ocrmypdf scan.pdf out.pdf`.

Text extraction removes page numbers, hyphenation, repeated PDF running heads and reference
lists (the TUI log shows how many entries were removed).

## Tips

- run outside peak hours — everything is half price;
- `--no-think` is even cheaper and faster, with slightly lower quality on difficult fragments;
- if the API answers with 429 errors (too many requests), lower `-j`;
- fine-tuning a model needs several thousand examples — process several sources and combine the
  datasets.

## Statistical data (NIBRS)

Data files (SPSS, Stata, SAS, R, delimited) are tables of codes, not text, so the generator does
not read them. For the [NIBRS extract files](https://www.icpsr.umich.edu/web/NACJD/series/128)
from ICPSR there is a converter that writes a text report of **offender profiles** — base rates
for profiling: for victims of an offense with a given feature, what the offenders were like. It
computes the figures with pandas, and the generator then processes the report like an article,
so the model only phrases the examples and cannot get a figure wrong; the quotes are checked
against the report.

Download the collection in **Stata** (or SPSS) format — they store the labels of variables and
codes — and pass the downloaded archives as they are. Several years are pooled into one report,
which gives enough cases for rare combinations; 2021 onwards is the best choice, since then all
agencies report through NIBRS:

```bash
uv run nibrs-report ~/Downloads/Nibrs/*.zip --inspect       # which variables were found
uv run nibrs-report ~/Downloads/Nibrs/*.zip                 # → input/nibrs-2021-2024.txt
uv run dataset-generator input/nibrs-2021-2024.txt -k mixed -n 4
```

For each archive only the victim-level file is unpacked (found by the archive's manifest, also
when it is a `.gz` inside the archive), next to the archive, and it is deleted once it has been
read; one year takes about two minutes. Unpacked download folders and the data files themselves
work too.

The report covers individual victims of murder, rape, sodomy and sexual assault with an object,
fondling (criminal sexual contact), aggravated assault, robbery and kidnapping. For each offense
it gives an overview (victims, clearance, the victim's sex and age, weapons, locations, time of
day, circumstances) and then offender profiles — the relationship to the victim, the offender's
sex and age, two or more offenders, suspected alcohol or drug use — for all victims and by:

- the victim's sex and age;
- the weapon (firearm, knife, blunt object, hands and feet, asphyxiation…);
- the location (residence, street, parking lot, outdoors, hotel, bar, store, school…);
- the time of day and, for murder and aggravated assault, the circumstances;
- for murder and sexual assault, also the victim's sex combined with the location and with the
  weapon.

Every paragraph names the data and says for how many victims each characteristic was recorded.
The report begins with the definitions, the method, and the biases: offender information comes
mostly from cases with an identified suspect, so strangers are under-represented. Race and
ethnicity are left out on purpose, so that the model does not learn to profile by race.
Variables are recognised by their labels and renamed to the same names for every year, so
`--inspect` shows what each one was taken for. The years in the text are taken from the data;
`--name` sets the name by hand.

## Fine-tuning

[`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) fine-tunes Qwen3-8B on the
datasets with QLoRA (Unsloth) on a free Google Colab T4. In Colab: *File → Upload notebook*,
select the T4 GPU, put the `*.jsonl` files into `MyDrive/forensics-llm/data/` and run all cells.
Checkpoints, the trained LoRA adapter and an optional GGUF export for Ollama are saved to Google
Drive; to continue training later, set `CONTINUE_FROM` to the previous run's name. Examples
whose supporting quote was not found in the source are skipped (`DROP_UNVERIFIED`).

## Project layout

```
src/dataset_generator/
├── cli.py          command-line arguments
├── tui.py          Textual interface;  kind_picker.py — dataset type picker
├── console.py      mode without the TUI
├── pipeline.py     pipeline: file → language and type → fragments (in parallel) → model → dataset
├── backends.py     DeepSeek API: streaming requests, JSON checks, errors
├── llm.py          example generation and source analysis on top of the backend
├── quality.py      quality labels and the check of supporting quotes against the source
├── nibrs.py        NIBRS data files → a text report of offender profiles
├── source_card.py  source cards: description of the source, statistics of its datasets
├── kinds.py        dataset types: response schemas, prompt tasks, examples
├── prompts.py      forensic expert prompts
├── pricing.py      DeepSeek pricing, peak hours, cost calculation
├── estimate.py     time and cost estimates, remaining time
├── extract.py      text extraction;  language.py — language detection
├── chunking.py     splitting into fragments, removing reference lists
└── dataset.py      record formats, writing JSONL, resuming
```

Code checks: `uv run ruff check .` and `uv run ruff format .`
