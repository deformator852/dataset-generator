"""NIBRS extract files (ICPSR) → a text report of crime statistics for the generator.

The numbers are computed here, with pandas, from the victim-level file; the model only
turns the report into training examples, so it cannot get a figure wrong. Variables are
found by their labels, which Stata and SPSS files store, rather than by names, which may
differ between years of the extract files: `--inspect` shows what was found.

    uv run nibrs-report ~/Downloads/ICPSR_39270-V2.zip
    uv run nibrs-report ~/Datasets/NIBRS-2023 --inspect     # an unpacked download works too
"""

import argparse
import gzip
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat

READERS: dict[str, Callable[..., Any]] = {
    ".dta": pyreadstat.read_dta,
    ".sav": pyreadstat.read_sav,
    ".zsav": pyreadstat.read_sav,
    ".por": pyreadstat.read_por,
    ".sas7bdat": pyreadstat.read_sas7bdat,
}
_CHUNK_ROWS = 1_000_000
_MIN_GROUP = 30  # fewer victims than this: no paragraph, the shares would be noise
_TOP = 8  # categories listed per paragraph; the rest are summed up as "other"


@dataclass(frozen=True, slots=True)
class Concept:
    key: str
    label: str  # regex on the variable label
    multi: bool = False  # numbered variables (injury 1…5): all of them, not just the first


# Race and ethnicity are left out on purpose: an assistant for investigators must not learn
# to profile suspects by race.
VICTIM_CONCEPTS = (
    Concept("offense", r"^ucr offense code \d+$", multi=True),  # offenses against this victim
    Concept("victim_type", r"^type of victim$"),
    Concept("victim_age", r"^age of victim$"),
    Concept("victim_sex", r"^sex of victim$"),
    Concept("circumstance", r"homicide circumstance \d", multi=True),
    Concept("injury", r"^injury \d", multi=True),
    Concept("relationship", r"^relationship vic(tim)? to off(ender)? 1$"),
    Concept("hour", r"incident date hour"),
    Concept("offenders", r"^total offender segments$"),
    Concept("arrestees", r"^total arrestee segments$"),
    Concept("cleared_exceptionally", r"^cleared exceptionally$"),
    Concept("offender_number", r"^offender sequence number - 1$"),
    Concept("offender_age", r"^age of offender - 1$"),
    Concept("offender_sex", r"^sex of offender - 1$"),
)
# The incident's offense records (up to three in this file), "… - 1" to "… - 3": weapons,
# location and completion are taken from the record of the offense against the victim.
SEGMENT_CONCEPTS = (
    Concept("segment_offense", r"^ucr offense code - (\d)$"),
    Concept("completed", r"attempted ?/ ?completed - (\d)$"),
    Concept("location", r"^location type - (\d)$"),
    Concept("weapon", r"^weapon ?/ ?force \d - (\d)$", multi=True),
    Concept("using", r"suspected of using \d - (\d)$", multi=True),
)

OFFENSES = (
    ("murder and nonnegligent manslaughter", r"murder"),
    ("negligent manslaughter", r"(?<!non)negligent manslaughter"),
    ("rape", r"(?<!statutory )\brape\b"),
    ("sodomy", r"sodomy"),
    ("sexual assault with an object", r"with (an )?object"),
    ("fondling", r"fondling"),
    ("aggravated assault", r"aggravated assault"),
    ("robbery", r"robbery"),
    ("kidnapping or abduction", r"kidnap|abduct"),
    ("human trafficking for commercial sex acts", r"trafficking.{0,3}commercial"),
    ("human trafficking for involuntary servitude", r"trafficking.{0,3}involuntary"),
)
_HOMICIDES = {"murder and nonnegligent manslaughter", "negligent manslaughter"}
_WITH_CIRCUMSTANCES = _HOMICIDES | {"aggravated assault"}
# Offenses where the victim's age decides most (child abuse vs adult victims).
_BY_AGE = {
    "rape",
    "sodomy",
    "sexual assault with an object",
    "fondling",
    "kidnapping or abduction",
    "human trafficking for commercial sex acts",
}

# Order matters: "child of boyfriend" before partners, "boyfriend" before "friend",
# "cohabitant (non-intimate)" before anything intimate.
_RELATIONSHIPS = (
    ("the victim was the offender", r"victim was (the )?offender"),
    ("acquaintance or otherwise known", r"child of (a )?(boy|girl)friend|cohabitant"),
    ("intimate partner", r"spouse|common.law|boyfriend|girlfriend|homosexual|ex.?relationship"),
    ("family member", r"parent|child|sibling|grand|in.law|step|family"),
    ("acquaintance or otherwise known", r"acquaint|friend|neighbo|babysit|employ|known"),
    ("stranger", r"stranger"),
)
_AGE_BANDS = (
    (0, 11, "under 12"),
    (12, 17, "12–17"),
    (18, 24, "18–24"),
    (25, 34, "25–34"),
    (35, 49, "35–49"),
    (50, 64, "50–64"),
    (65, 200, "65 and over"),
)
# Hour 00 is left out: see the introduction of the report.
_HOUR_BANDS = (
    (1, 5, "night (01:00–05:59)"),
    (6, 11, "morning (06:00–11:59)"),
    (12, 17, "afternoon (12:00–17:59)"),
    (18, 23, "evening (18:00–23:59)"),
)
_FIREARM = re.compile(r"firearm|handgun|rifle|shotgun", re.I)
_UNKNOWN = re.compile(
    r"unknown|undetermined|missing|not (reported|applicable|specified)|^na |^\s*$|^nan$|^<na>$",
    re.I,
)


@dataclass(frozen=True, slots=True)
class Columns:
    victim: dict[str, list[str]]  # concept → columns
    segment: dict[str, dict[int, list[str]]]  # concept → offense record number → columns

    def names(self) -> list[str]:
        names = [name for names in self.victim.values() for name in names]
        names += [name for slots in self.segment.values() for ns in slots.values() for name in ns]
        return sorted(set(names))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nibrs-report",
        description="Turns the NIBRS extract files (ICPSR) into a text report of crime "
        "statistics that dataset-generator can process like an article.",
    )
    parser.add_argument("source", type=Path, help="the ICPSR download: a .zip or its folder")
    parser.add_argument("--victims", type=Path, help="the victim-level file, if not found")
    parser.add_argument("-o", "--output", type=Path, help="default: input/<name>.txt")
    parser.add_argument(
        "--name", help='data name in the report text; default: "NIBRS <year of the data>"'
    )
    parser.add_argument(
        "--inspect", action="store_true", help="list the variables and what they were taken for"
    )
    args = parser.parse_args(argv)

    if args.victims:
        return _run(args.victims, args)
    if args.source.suffix.lower() == ".zip":
        with extracted_victim_file(args.source) as victims:
            return _run(victims, args) if victims else 1
    victims = find_victim_file(args.source)
    if victims is None:
        print(f"No victim-level file in {args.source}; use --victims")
        return 1
    return _run(victims, args)


def _run(victims: Path, args: argparse.Namespace) -> int:
    if args.inspect:
        inspect(victims)
        return 0
    columns = find_columns(victims)
    if "offense" not in columns.victim:
        print(f"No offense code variables in {victims.name}; check with --inspect")
        return 1
    found = set(columns.victim) | set(columns.segment)
    missing = [c.key for c in (*VICTIM_CONCEPTS, *SEGMENT_CONCEPTS) if c.key not in found]
    if missing:
        print(f"Not found, left out of the report: {', '.join(missing)}")

    year = data_year(victims)
    if not args.name and not year:
        print('The year of the data was not found; set it with --name "NIBRS <year>"')
        return 1
    name = args.name or f"NIBRS {year}"
    print(f"Reading {victims.name} ({name})…")
    frame, total, midnight = load_victims(victims, columns)
    report = build_report(frame, columns, name=name, total=total, midnight=midnight)
    output = args.output or Path("input") / f"{_slug(name)}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"{len(report):,} characters → {output}")
    print(f"Next: uv run dataset-generator {output} -k mixed")
    return 0


@contextmanager
def extracted_victim_file(archive_path: Path) -> Iterator[Path | None]:
    """Unpacks only the victim-level file of an ICPSR download, and deletes it afterwards.

    The file is found by the download's manifest ("DS0004 Victim-Level File"). It is
    unpacked next to the archive, not to /tmp, which is often in memory and too small.
    """
    with zipfile.ZipFile(archive_path) as archive:
        member = _victim_member(archive)
        if member is None:
            if _is_fbi_master_file(archive):
                print(
                    f"{archive_path.name} is the FBI's raw NIBRS master file (fixed-width "
                    "segments without labels), which this script does not read. Download the "
                    "ICPSR extract files of the same year in Stata format instead: "
                    'search ICPSR for "National Incident-Based Reporting System, <year>: '
                    'Extract Files".'
                )
            else:
                print(f"No victim-level file in the manifest of {archive_path.name}; unpack it")
            yield None
            return
        needed = member.file_size * (5 if member.filename.endswith(".gz") else 1)
        free = shutil.disk_usage(archive_path.parent).free
        if free < needed * 1.1:
            print(f"Not enough disk space next to the archive: {needed / 1e9:.1f} GB needed")
            yield None
            return
        with tempfile.TemporaryDirectory(dir=archive_path.parent, prefix=".nibrs-") as tmp:
            name = Path(member.filename).name.removesuffix(".gz")
            target = Path(tmp) / name
            print(f"Unpacking {name} ({member.file_size / 1e9:.1f} GB) — deleted afterwards…")
            with archive.open(member) as source, target.open("wb") as file:
                stream = (
                    gzip.GzipFile(fileobj=source) if member.filename.endswith(".gz") else source
                )
                shutil.copyfileobj(stream, file, 16 << 20)
            yield target


def _is_fbi_master_file(archive: zipfile.ZipFile) -> bool:
    """The FBI download: text records starting with segment codes, batch headers first."""
    texts = [m for m in archive.infolist() if m.filename.lower().endswith(".txt")]
    if len(texts) != 1:
        return False
    with archive.open(texts[0]) as file:
        return file.read(2) == b"BH"


def _victim_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo | None:
    manifest = next((m for m in archive.namelist() if m.endswith("-manifest.txt")), None)
    if manifest is None:
        return None
    text = archive.read(manifest).decode("utf-8", errors="replace")
    match = re.search(r"^DS(\d{4})\s+Victim-Level", text, re.M | re.I)
    if match is None:
        return None
    number = match.group(1)
    candidates = [m for m in archive.infolist() if f"-{number}-Data." in m.filename]
    # Formats with the labels of variables and codes, in order of preference.
    for suffix in (".dta", ".dta.gz", ".sav", ".sav.gz", ".zsav", ".por"):
        for member in candidates:
            if member.filename.lower().endswith(suffix):
                return member
    return None


def find_victim_file(folder: Path) -> Path | None:
    """The victim-level file, recognised by its variables rather than its dataset number.

    The number differs between years: DS2 in the 1991–2016 extract files, DS4 since 2017.
    """
    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in READERS):
        try:
            columns = find_columns(path)
        except Exception:  # not a readable data file
            continue
        if {"offense", "victim_type", "victim_age"} <= set(columns.victim):
            return path
    return None


def data_year(path: Path) -> int | None:
    """The year of the data, from the master file year or else the incident dates."""
    labels = _metadata(path).column_names_to_labels
    for pattern in (r"master file year", r"^incident date$"):
        name = next(
            (n for n, label in labels.items() if label and re.search(pattern, label, re.I)), None
        )
        if name is None:
            continue
        frame, _ = READERS[path.suffix.lower()](path, usecols=[name], row_limit=1000)
        years = pd.Series(frame[name]).dropna().map(_year).dropna()
        if len(years):
            return int(years.mode().iloc[0])
    return None


def _year(value: object) -> int | None:
    if hasattr(value, "year"):
        return int(value.year)  # type: ignore[attr-defined]
    match = re.search(r"(19|20)\d\d", str(value))
    return int(match.group()) if match else None


def _metadata(path: Path) -> Any:
    _, meta = READERS[path.suffix.lower()](path, metadataonly=True)
    return meta


def find_columns(path: Path) -> Columns:
    labels = _metadata(path).column_names_to_labels
    victim: dict[str, list[str]] = {}
    for concept in VICTIM_CONCEPTS:
        pattern = re.compile(concept.label, re.I)
        matched = [name for name, label in labels.items() if label and pattern.search(label)]
        if matched:
            victim[concept.key] = matched if concept.multi else matched[:1]
    segment: dict[str, dict[int, list[str]]] = {}
    for concept in SEGMENT_CONCEPTS:
        pattern = re.compile(concept.label, re.I)
        slots: dict[int, list[str]] = {}
        for name, label in labels.items():
            if label and (match := pattern.search(label)):
                slots.setdefault(int(match.group(1)), []).append(name)
        if slots:
            segment[concept.key] = {
                number: names if concept.multi else names[:1] for number, names in slots.items()
            }
    return Columns(victim, segment)


def inspect(path: Path) -> None:
    meta = _metadata(path)
    columns = find_columns(path)
    taken = {name: key for key, names in columns.victim.items() for name in names}
    for key, slots in columns.segment.items():
        taken |= {
            name: f"{key}, offense record {n}" for n, names in slots.items() for name in names
        }
    print(f"{path}: {meta.number_rows or '?'} rows, {len(meta.column_names)} variables\n")
    for name in meta.column_names:
        label = meta.column_names_to_labels.get(name) or ""
        mark = f"  ← {taken[name]}" if name in taken else ""
        print(f"{name:<14} {label}{mark}")


def load_victims(path: Path, columns: Columns) -> tuple[pd.DataFrame, int, float]:
    """Individual victims of the offenses in the report.

    The file is read in chunks and only those victims are kept, so it fits in memory.
    Codes are replaced with their labels (131 → "Aggravated Assault"). Also returns the
    number of victim records of all kinds and the share of them recorded at hour 00.
    """
    patterns = [re.compile(pattern, re.I) for _, pattern in OFFENSES]
    hour = columns.victim.get("hour", [None])[0]
    kept: list[pd.DataFrame] = []
    total = midnight = 0
    for chunk, _ in pyreadstat.read_file_in_chunks(
        READERS[path.suffix.lower()],
        path,
        chunksize=_CHUNK_ROWS,
        usecols=columns.names(),
        apply_value_formats=True,
        formats_as_category=True,
    ):
        total += len(chunk)
        if hour:
            midnight += int((pd.to_numeric(chunk[hour], errors="coerce") == 0).sum())
        mask = pd.Series(False, index=chunk.index)
        for pattern in patterns:
            mask |= _matches(chunk, columns.victim["offense"], pattern)
        if "victim_type" in columns.victim:
            victim_type = chunk[columns.victim["victim_type"][0]].astype(str)
            mask &= victim_type.str.contains("individual", case=False)
        kept.append(chunk[mask].astype(str))
        print(f"  {total:,} victim records read", end="\r", flush=True)
    print()
    return pd.concat(kept, ignore_index=True), total, midnight / total if total else 0.0


def _matches(frame: pd.DataFrame, names: Iterable[str], pattern: re.Pattern[str]) -> pd.Series:
    """Rows where any of the columns has a value matching the pattern."""
    mask = pd.Series(False, index=frame.index)
    for name in names:
        values = frame[name]
        hits = [value for value in values.dropna().unique() if pattern.search(str(value))]
        mask |= values.isin(hits)
    return mask


# --- the report ----------------------------------------------------------------------


def build_report(
    frame: pd.DataFrame, columns: Columns, *, name: str, total: int, midnight: float
) -> str:
    # Every paragraph names the data: the generator cuts the report into fragments, and
    # each fragment must say where its figures come from.
    prefix = f"In {name} data (US agencies participating in NIBRS, not nationally representative),"
    sections = [_introduction(name, total, midnight)]
    for offense, pattern in OFFENSES:
        regex = re.compile(pattern, re.I)
        victims = frame[_matches(frame, columns.victim["offense"], regex)]
        if len(victims) < _MIN_GROUP:
            continue
        paragraphs = list(_offense_paragraphs(offense, regex, victims, columns, prefix))
        sections.append(f"{offense.capitalize()}\n\n" + "\n\n".join(paragraphs))
    return "\n\n\n".join(sections) + "\n"


def _introduction(name: str, total: int, midnight: float) -> str:
    return f"""\
Crime statistics from {name}: victims of violent and sexual offenses

The National Incident-Based Reporting System (NIBRS) is part of the FBI's Uniform Crime \
Reporting Program. Participating law enforcement agencies report every criminal incident \
and arrest: for each Group A offense, they record the offense, the location, weapons or \
force used, and the victims, offenders and arrestees. \
An incident is one or more offenses committed by the same offender, or group of offenders \
acting in concert, at the same time and place.

This report was computed from the victim-level file of {name} ({total:,} victim records of \
all kinds). It covers victims who are individuals (not businesses, society or the \
government) and against whom one of the offenses below was committed; a victim of several \
such offenses appears in several sections. Shares are given among victims with a known \
value, and the share of unknown or unreported values is stated separately. Weapons, the \
location and whether the offense was completed come from the incident's record of that \
offense. The relationship refers to the first offender linked to the victim, and the age \
and sex of the offender to the first offender of the incident. The number of offenders \
and the clearance are counted for the incident. Relationships are grouped as follows: \
intimate partner (spouse, common-law spouse, ex-spouse, boyfriend or girlfriend, \
ex-relationship, same-sex relationship), family member (parent, child, sibling, \
grandparent, grandchild, in-law, step-relatives, foster relatives, other family), \
acquaintance or otherwise known (acquaintance, friend, neighbor, babysittee, employee, \
employer, non-intimate cohabitant, child of a boyfriend or girlfriend, otherwise known) \
and stranger.

The time of day leaves out the hour 00:00–00:59: it holds {midnight:.1%} of all victim \
records, several times more than the neighbouring hours, so it evidently also contains \
incidents whose time was unknown. A case is counted as cleared by arrest if the incident \
has at least one arrestee in the data, and as cleared exceptionally if the agency reported \
an exceptional clearance (for example, the offender died or prosecution was declined).

Limitations: the data are not a nationally representative sample of crime in the United \
States, because agency participation varies between states and cities. The figures \
describe crimes known to and recorded by the police, classified when they were reported, \
not court findings; crimes that were never reported, such as many sexual offenses, are \
missing. Offender information is often unknown, especially when no suspect was \
identified."""


def _offense_paragraphs(
    offense: str,
    regex: re.Pattern[str],
    victims: pd.DataFrame,
    columns: Columns,
    prefix: str,
) -> Iterable[str]:
    has = columns.victim.__contains__

    def values(key: str, convert: Callable[[str], str | None] = _known) -> pd.Series:
        return _convert(victims[columns.victim[key][0]], convert)

    base = f"{prefix} {offense} —"
    ages = _band_order(_AGE_BANDS)
    sex = values("victim_sex", _sex) if has("victim_sex") else None
    age = values("victim_age", _age_band) if has("victim_age") else None
    record = _offense_record(victims, columns, regex)

    if sex is not None:
        yield _shares(base, "sex of the victim", sex)
    if age is not None:
        yield _shares(base, "age of the victim", age, order=ages)
    if has("relationship"):
        relationship = values("relationship", _relationship)
        yield _shares(base, "relationship of the victim to the offender", relationship)
        yield _shares(
            base, "detailed relationship of the victim to the offender", values("relationship")
        )
        groups: list[tuple[str, pd.Series]] = []
        if sex is not None:
            groups += [(f"{value} victims", sex == value) for value in ("female", "male")]
        if age is not None and offense in _BY_AGE:
            groups += [
                ("victims under 12", age == "under 12"),
                ("victims aged 12–17", age == "12–17"),
                ("adult victims (18 and over)", age.notna() & ~age.isin(["under 12", "12–17"])),
            ]
        for label, mask in groups:
            if mask.sum() >= _MIN_GROUP:
                aspect = f"relationship to the offender for {label}"
                yield _shares(base, aspect, relationship[mask])
    if has("circumstance") and offense in _WITH_CIRCUMSTANCES:
        circumstances = victims[columns.victim["circumstance"]]
        yield _multi_shares(base, "circumstances of the incident", circumstances)
    if has("injury") and offense not in _HOMICIDES:
        injuries = _convert_cells(victims[columns.victim["injury"]], _injury)
        yield _multi_shares(base, "injury of the victim", injuries)
    if "weapon" in record:
        weapons = _convert_cells(record["weapon"], _weapon)
        firearm = weapons.apply(lambda column: column.str.contains(_FIREARM, na=False))
        firearm = firearm.any(axis=1) & weapons.notna().any(axis=1)
        known = weapons.notna().any(axis=1)
        sentence = _multi_shares(base, "weapon or force involved", weapons)
        if known.sum():
            sentence += (
                f" A firearm of any type was involved for {_pct(firearm.sum(), known.sum())}."
            )
        yield sentence
    if "location" in record:
        yield _shares(base, "location of the offense", _convert(record["location"][0], _known))
    completed = _convert(record["completed"][0], _known) if "completed" in record else None
    if completed is not None and completed.nunique() > 1:  # a murder is always completed
        yield _shares(base, "offense completed or attempted", completed)
    if "using" in record:
        yield _using(base, record["using"])
    if has("hour"):
        yield _time_of_day(base, victims[columns.victim["hour"][0]])
    if has("offenders") and has("offender_number"):
        yield _shares(
            base, "number of offenders in the incident", _offender_count(victims, columns)
        )
    if has("offender_sex"):
        yield _shares(base, "sex of the offender", values("offender_sex", _sex))
    if has("offender_age"):
        yield _shares(base, "age of the offender", values("offender_age", _age_band), order=ages)
    if has("arrestees"):
        yield _clearance(base, victims, columns)


def _offense_record(
    victims: pd.DataFrame, columns: Columns, regex: re.Pattern[str]
) -> dict[str, pd.DataFrame]:
    """Columns of the incident's record of this offense: concept → a frame of its values.

    A victim whose offense is not among the incident's first records gets no values.
    """
    record_number = pd.Series(0, index=victims.index)
    for number, names in sorted(columns.segment.get("segment_offense", {}).items()):
        hit = _matches(victims, names, regex) & (record_number == 0)
        record_number[hit] = number
    result: dict[str, pd.DataFrame] = {}
    for key, slots in columns.segment.items():
        if key == "segment_offense":
            continue
        width = max(len(names) for names in slots.values())
        frame = {}
        for position in range(width):
            column = pd.Series(None, index=victims.index, dtype=object)
            for number, names in slots.items():
                if position < len(names):
                    column = column.where(record_number != number, victims[names[position]])
            frame[position] = column
        result[key] = pd.DataFrame(frame)
    return result


def _shares(base: str, aspect: str, values: pd.Series, order: list[str] | None = None) -> str:
    """One value per victim: unknown share first, then the shares among known values."""
    known = values.dropna()
    counts = known.value_counts()
    if order:
        counts = counts.reindex([band for band in order if band in counts.index])
    return _sentence(base, aspect, len(values), len(known), counts)


def _multi_shares(base: str, aspect: str, cells: pd.DataFrame) -> str:
    """Several values per victim (weapon 1…3): the share of victims with each value."""
    cells = _convert_cells(cells, _known)
    known = cells.notna().any(axis=1)
    counts = _victims_per_value(cells)
    sentence = _sentence(base, aspect, len(cells), int(known.sum()), counts)
    if len(counts) > 1:
        sentence += " A victim can have several values, so the shares can add up to more than 100%."
    return sentence


def _sentence(base: str, aspect: str, total: int, known: int, counts: pd.Series) -> str:
    unknown = total - known
    text = f"{base} {aspect}, {total:,} victims."
    if known == 0:
        return f"{text} It was unknown or not reported for all of them."
    if unknown:
        text += (
            f" It was unknown or not reported for {_pct(unknown, total)}; among the "
            f"{known:,} victims where it was known:"
        )
    top = counts.iloc[:_TOP]
    parts = [f"{_label(value)} {_pct(count, known)}" for value, count in top.items()]
    rest = int(counts.iloc[_TOP:].sum())
    if rest:
        parts.append(f"other {_pct(rest, known)}")
    return f"{text} {', '.join(parts)}."


def _time_of_day(base: str, hours: pd.Series) -> str:
    """Hour 00 also holds incidents of unknown time, so it is counted apart and left out."""
    numbers = pd.to_numeric(hours, errors="coerce")
    midnight = int((numbers == 0).sum())
    bands = _convert(hours, _hour_band)
    known = bands.dropna()
    order = [band for band in _band_order(_HOUR_BANDS) if band in set(known)]
    counts = known.value_counts().reindex(order)
    parts = [f"{band} {_pct(count, len(known))}" for band, count in counts.items()]
    return (
        f"{base} time of day, {len(hours):,} victims. {_pct(midnight, len(hours))} of them "
        f"were recorded at hour 00, which is left out because it also holds incidents of "
        f"unknown time; among the other {len(known):,} victims: {', '.join(parts)}."
    )


def _using(base: str, cells: pd.DataFrame) -> str:
    """Suspected use is only recorded when it was suspected, so shares are of all victims."""
    cells = _convert_cells(cells, _known)
    total = len(cells)
    counts = _victims_per_value(cells)
    none = int((~cells.notna().any(axis=1)).sum())
    parts = [f"{value.lower()} {_pct(count, total)}" for value, count in counts.items()]
    used = f": {', '.join(parts)}" if parts else ""
    return (
        f"{base} offenders suspected of using alcohol, drugs or computer equipment, {total:,} "
        f"victims. Use was suspected for {_pct(total - none, total)} of the victims{used}; for "
        f"the other {_pct(none, total)} no use was recorded."
    )


_CLEARANCE_REASONS = (
    (r"death", "death of the offender"),
    (r"prosecution", "prosecution declined"),
    (r"custody of other", "offender in custody of another jurisdiction"),
    (r"refused", "victim refused to cooperate"),
    (r"juvenile", "juvenile offender, no custody"),
)


def _clearance_reason(value: str) -> str:
    return next(
        (text for pattern, text in _CLEARANCE_REASONS if re.search(pattern, value, re.I)),
        value.lower(),
    )


def _offender_count(victims: pd.DataFrame, columns: Columns) -> pd.Series:
    unknown = _convert(
        victims[columns.victim["offender_number"][0]],
        lambda value: _UNKNOWN.search(str(value)) is not None or str(value) in {"0", "0.0"},
    ).astype(bool)
    bands = _convert(
        pd.to_numeric(victims[columns.victim["offenders"][0]], errors="coerce"),
        lambda n: None if pd.isna(n) else "one" if n == 1 else "two" if n == 2 else "three or more",
    )
    return bands.where(~unknown, None)


def _clearance(base: str, victims: pd.DataFrame, columns: Columns) -> str:
    arrests = pd.to_numeric(victims[columns.victim["arrestees"][0]], errors="coerce").fillna(0)
    status = pd.Series("not cleared", index=victims.index)
    if "cleared_exceptionally" in columns.victim:
        reason = _convert(victims[columns.victim["cleared_exceptionally"][0]], _known)
        exceptional = reason.notna()
        reason = _convert(reason[exceptional], _clearance_reason)
        status[exceptional] = "cleared exceptionally (" + reason + ")"
    status[arrests > 0] = "cleared by arrest"
    counts = status.value_counts()
    parts = [f"{value} {_pct(count, len(status))}" for value, count in counts.items()]
    return f"{base} clearance of the incident, {len(status):,} victims: {', '.join(parts)}."


def _pct(part: float, whole: float) -> str:
    share = 100 * part / whole
    return "<0.1%" if 0 < share < 0.05 else f"{share:.1f}%"


def _label(value: object) -> str:
    """A category as it reads in a sentence: "Victim was Spouse" → "spouse"."""
    return re.sub(r"^victim was ", "", str(value), flags=re.I).lower()


def _band_order(bands: Iterable[tuple[int, int, str]]) -> list[str]:
    return [label for _, _, label in bands]


# --- value conversion ----------------------------------------------------------------


def _convert(values: pd.Series, convert: Callable[[Any], Any]) -> pd.Series:
    """Converts each distinct value once: a column has millions of cells but few values."""
    mapping = {value: convert(value) for value in values.unique()}
    return values.map(mapping)


def _convert_cells(cells: pd.DataFrame, convert: Callable[[Any], Any]) -> pd.DataFrame:
    return cells.apply(lambda column: _convert(column, convert))


def _victims_per_value(cells: pd.DataFrame) -> pd.Series:
    """How many victims have each value at least once among their columns."""
    pairs = cells.stack().dropna().rename("value").reset_index()
    return pairs.drop_duplicates([pairs.columns[0], "value"])["value"].value_counts()


def _known(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if _UNKNOWN.search(text) else text


def _weapon(value: object) -> str | None:
    known = _known(value)
    return "no weapon (none)" if known and known.lower() == "none" else known


def _injury(value: object) -> str | None:
    known = _known(value)
    return "no injury" if known and known.lower() == "none" else known


def _sex(value: object) -> str | None:
    text = str(value).strip().lower()
    if text in {"f", "female"}:
        return "female"
    if text in {"m", "male"}:
        return "male"
    return None


def _relationship(value: object) -> str | None:
    known = _known(value)
    if known is None:
        return None
    lowered = known.lower()
    for group, pattern in _RELATIONSHIPS:
        if re.search(pattern, lowered):
            return group
    return "other"


def _age(value: object) -> float | None:
    """Age in years; infants are stored as fractions (0.1 under 24 hours … 0.5 under a year)."""
    text = str(value).strip().lower()
    if re.search(r"neonate|newborn|baby|under 24 hours|days? old", text):
        return 0.0
    if "over 98" in text:
        return 99.0
    try:
        age = float(text)
    except ValueError:
        return None
    return age if 0 < age < 120 else None  # 0 is the code for an unknown age


def _age_band(value: object) -> str | None:
    age = _age(value)
    return None if age is None else _band(int(age), _AGE_BANDS)


def _hour_band(value: object) -> str | None:
    try:
        hour = int(float(str(value)))
    except ValueError:
        return None
    return _band(hour, _HOUR_BANDS)


def _band(number: int, bands: Iterable[tuple[int, int, str]]) -> str | None:
    return next((label for low, high, label in bands if low <= number <= high), None)


def _slug(name: str) -> str:
    return re.sub(r"[^\w]+", "-", name.lower()).strip("-")


if __name__ == "__main__":
    sys.exit(main())
