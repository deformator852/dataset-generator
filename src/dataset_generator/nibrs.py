"""NIBRS extract files (ICPSR) → a text report of offender profiles for the generator.

The report gives base rates for profiling: for victims of an offense with a given feature
(the victim's sex and age, the weapon, the location, the time, the circumstances), what the
offenders were like (relationship to the victim, sex, age, number, alcohol or drugs). The
numbers are computed here, with pandas, from the victim-level file; the model only turns
the report into training examples, so it cannot get a figure wrong.

Several years can be pooled into one report, which gives enough cases for rare
combinations. Variables are found by their labels, which Stata and SPSS files store, and
renamed to the same names for every year: `--inspect` shows what was found.

    uv run nibrs-report ~/Downloads/Nibrs/NIBRS-2024.zip
    uv run nibrs-report ~/Downloads/Nibrs/*.zip                 # several years, pooled
    uv run nibrs-report ~/Datasets/NIBRS-2023 --inspect         # an unpacked download too
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

import numpy as np
import pandas as pd
import pyreadstat
from pandas.api.types import union_categoricals

READERS: dict[str, Callable[..., Any]] = {
    ".dta": pyreadstat.read_dta,
    ".sav": pyreadstat.read_sav,
    ".zsav": pyreadstat.read_sav,
    ".por": pyreadstat.read_por,
    ".sas7bdat": pyreadstat.read_sas7bdat,
}
_CHUNK_ROWS = 1_000_000
_MIN_GROUP = 30  # fewer known values than this: the shares would be noise
_MIN_PROFILE = 100  # fewer victims than this: no offender profile
_TOP = 8  # categories listed per paragraph; the rest are summed up as "other"


@dataclass(frozen=True, slots=True)
class Concept:
    key: str
    label: str  # regex on the variable label
    multi: bool = False  # numbered variables (offense 1…10): all of them, not just the first


# Race and ethnicity are left out on purpose: an assistant for investigators must not learn
# to profile suspects by race.
VICTIM_CONCEPTS = (
    Concept("offense", r"^ucr offense code \d+$", multi=True),  # offenses against this victim
    Concept("victim_type", r"^type of victim$"),
    Concept("victim_age", r"^age of victim$"),
    Concept("victim_sex", r"^sex of victim$"),
    Concept("circumstance", r"homicide circumstance \d", multi=True),
    Concept("relationship", r"^relationship vic(tim)? to off(ender)? 1$"),
    Concept("hour", r"incident date hour"),
    Concept("offenders", r"^total offender segments$"),
    Concept("arrestees", r"^total arrestee segments$"),
    Concept("cleared_exceptionally", r"^cleared exceptionally$"),
    # Offender 1 of the victim-level file is the offender the victim is linked to (checked
    # on the 2024 file: the same offender for 99.9% of victims with a relationship).
    Concept("offender_number", r"^offender sequence number - 1$"),
    Concept("offender_age", r"^age of offender - 1$"),
    Concept("offender_sex", r"^sex of offender - 1$"),
)
# The incident's offense records (up to three in this file), "… - 1" to "… - 3": weapons,
# location and alcohol or drug use are taken from the record of the offense against the
# victim.
SEGMENT_CONCEPTS = (
    Concept("segment_offense", r"^ucr offense code - (\d)$"),
    Concept("location", r"^location type - (\d)$"),
    Concept("weapon", r"^weapon ?/ ?force \d - (\d)$", multi=True),
    Concept("using", r"suspected of using \d - (\d)$", multi=True),
)
_MULTI = {concept.key: concept.multi for concept in VICTIM_CONCEPTS}

OFFENSES = (
    ("murder and nonnegligent manslaughter", r"murder"),
    (
        "rape, sodomy or sexual assault with an object",
        r"(?<!statutory )\brape\b|sodomy|with (an )?object",
    ),
    # 11D was renamed from "Fondling" to "Criminal Sexual Contact" in the 2024 data.
    ("fondling or criminal sexual contact", r"fondling|criminal sexual contact"),
    ("aggravated assault", r"aggravated assault"),
    ("robbery", r"robbery"),
    ("kidnapping or abduction", r"kidnap|abduct"),
)
_WITH_CIRCUMSTANCES = {"murder and nonnegligent manslaughter", "aggravated assault"}
# Offender profiles by the victim's sex combined with the location and with the weapon.
_DETAILED = {
    "murder and nonnegligent manslaughter",
    "rape, sodomy or sexual assault with an object",
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
# Condition phrases complete "offender profile for offenses …"; order matters, the first
# match wins ("parking" before "park").
_WEAPONS = (
    ("involving a firearm", r"firearm|handgun|rifle|shotgun"),
    ("involving a knife or cutting instrument", r"knife|cutting"),
    ("involving a blunt object", r"blunt"),
    ("involving hands, fists, feet or other personal weapons", r"personal weapons"),
    ("involving asphyxiation", r"asphyx"),
    ("involving a motor vehicle", r"motor vehicle"),
    ("involving fire or an incendiary device", r"incendiary"),
    ("involving poison or drugs", r"poison|drugs/narcotics|sleeping"),
    ("in which no weapon was used", r"^no weapon"),
)
_LOCATIONS = (
    ("at a residence or home", r"residence|home"),
    ("in a parking lot or garage", r"parking"),
    ("on a highway, road, alley, street or sidewalk", r"highway|road|alley|street|sidewalk"),
    (
        "outdoors in a field, woods, at a lake or beach, or in a park",
        r"field|woods|lake|waterway|beach|\bpark\b|playground|camp",
    ),
    ("at a hotel or motel", r"hotel|motel"),
    ("at a bar or nightclub", r"bar/nightclub|nightclub"),
    (
        "at a store, restaurant, gas station or bank",
        r"store|grocery|supermarket|mall|restaurant|liquor|gas station|bank|atm",
    ),
    ("at a school, college or daycare", r"school|college|daycare"),
    ("in cyberspace", r"cyberspace"),
)
_CIRCUMSTANCES = (
    ("that arose from an argument", r"argument"),
    ("involving domestic violence or a lovers' quarrel", r"lovers|domestic"),
    ("linked to drug dealing", r"drug dealing"),
    ("linked to gang activity", r"gang"),
    ("committed during another felony", r"other felony"),
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
_OFFENDER_AGE_BANDS = (
    (0, 17, "under 18"),
    (18, 24, "18–24"),
    (25, 34, "25–34"),
    (35, 49, "35–49"),
    (50, 200, "50 and over"),
)
# Hour 00 is left out: see the introduction of the report.
_HOUR_BANDS = (
    (1, 5, "night (01:00–05:59)"),
    (6, 11, "morning (06:00–11:59)"),
    (12, 17, "afternoon (12:00–17:59)"),
    (18, 23, "evening (18:00–23:59)"),
)
_HOUR_PHRASES = {
    "night (01:00–05:59)": "committed at night (01:00–05:59)",
    "morning (06:00–11:59)": "committed in the morning (06:00–11:59)",
    "afternoon (12:00–17:59)": "committed in the afternoon (12:00–17:59)",
    "evening (18:00–23:59)": "committed in the evening (18:00–23:59)",
}
_FIREARM = re.compile(r"firearm|handgun|rifle|shotgun", re.I)
_ALCOHOL_OR_DRUGS = re.compile(r"alcohol|drug", re.I)
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

    def canonical(self) -> tuple[dict[str, str], "Columns"]:
        """The same names in every year's file: offense_1…, weapon_2_1 (offense record 2)…"""
        rename: dict[str, str] = {}
        victim: dict[str, list[str]] = {}
        for key, names in self.victim.items():
            new = [f"{key}_{i}" for i in range(1, len(names) + 1)] if _MULTI[key] else [key]
            rename |= dict(zip(names, new, strict=True))
            victim[key] = new
        segment: dict[str, dict[int, list[str]]] = {}
        for key, slots in self.segment.items():
            segment[key] = {}
            for number, names in slots.items():
                new = [f"{key}_{number}_{i}" for i in range(1, len(names) + 1)]
                rename |= dict(zip(names, new, strict=True))
                segment[key][number] = new
        return rename, Columns(victim, segment)


@dataclass(frozen=True, slots=True)
class Year:
    year: int
    records: int  # victim records of all kinds
    midnight: int  # of them at hour 00


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nibrs-report",
        description="Turns the NIBRS extract files (ICPSR) into a text report of offender "
        "profiles that dataset-generator can process like an article.",
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="ICPSR downloads: .zip archives, their folders or victim-level data files; "
        "several years are pooled into one report",
    )
    parser.add_argument("-o", "--output", type=Path, help="default: input/<name>.txt")
    parser.add_argument("--name", help='data name in the report; default: "NIBRS <years>"')
    parser.add_argument(
        "--inspect", action="store_true", help="list the variables and what they were taken for"
    )
    args = parser.parse_args(argv)

    frames: list[pd.DataFrame] = []
    columns: list[Columns] = []
    years: list[Year] = []
    for source in args.sources:
        with victim_file(source) as path:
            if path is None:
                return 1
            if args.inspect:
                inspect(path)
                continue
            found = find_columns(path)
            if "offense" not in found.victim:
                print(f"No offense code variables in {path.name}; check with --inspect")
                return 1
            have = set(found.victim) | set(found.segment)
            missing = [c.key for c in (*VICTIM_CONCEPTS, *SEGMENT_CONCEPTS) if c.key not in have]
            if missing:
                print(f"{source.name}: not found, left out: {', '.join(missing)}")
            year = data_year(path)
            if year is None:
                print(f"{source.name}: the year of the data was not found")
                return 1
            if year in {y.year for y in years}:
                print(f"{source.name}: {year} is already among the sources")
                return 1
            print(f"Reading {path.name} ({year})…")
            frame, records, midnight = load_victims(path, found)
            frames.append(frame)
            columns.append(found.canonical()[1])
            years.append(Year(year, records, midnight))
    if args.inspect:
        return 0

    name = args.name or f"NIBRS {_years_label(sorted(y.year for y in years))}"
    report = build_report(_concat(frames), _merge(columns), name=name, years=years)
    output = args.output or Path("input") / f"{_slug(name)}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"{len(report):,} characters → {output}")
    print(f"Next: uv run dataset-generator {output} -k mixed -n 4")
    return 0


@contextmanager
def victim_file(source: Path) -> Iterator[Path | None]:
    if source.suffix.lower() == ".zip":
        with extracted_victim_file(source) as path:
            yield path
        return
    if source.suffix.lower() in READERS:
        yield source
        return
    path = find_victim_file(source) if source.is_dir() else None
    if path is None:
        print(f"No victim-level file in {source}")
    yield path


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


def load_victims(path: Path, columns: Columns) -> tuple[pd.DataFrame, int, int]:
    """Individual victims of the offenses in the report, with the canonical column names.

    The file is read in chunks and only those victims are kept, as categories, so even
    several pooled years fit in memory. Codes are replaced with their labels (131 →
    "Aggravated Assault"). Also returns the number of victim records of all kinds and how
    many of them were recorded at hour 00.
    """
    rename, canonical = columns.canonical()
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
        chunk = chunk.rename(columns=rename)
        mask = pd.Series(False, index=chunk.index)
        for pattern in patterns:
            mask |= _matches(chunk, canonical.victim["offense"], pattern)
        if "victim_type" in canonical.victim:
            victim_type = chunk["victim_type"].astype(str)
            mask &= victim_type.str.contains("individual", case=False)
        kept.append(chunk[mask].astype(str).astype("category"))
        print(f"  {total:,} victim records read", end="\r", flush=True)
    print()
    return _concat(kept), total, midnight


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Joins categorical frames without turning them into millions of Python strings."""
    names = list(dict.fromkeys(name for frame in frames for name in frame.columns))
    data = {}
    for name in names:
        parts = [
            frame[name].array
            if name in frame
            else pd.Categorical.from_codes(np.zeros(len(frame), dtype=np.int8), ["nan"])
            for frame in frames
        ]
        data[name] = union_categoricals(parts, ignore_order=True)
    return pd.DataFrame(data)


def _merge(columns: list[Columns]) -> Columns:
    """Canonical columns of several years: the widest set of each concept."""
    victim: dict[str, list[str]] = {}
    segment: dict[str, dict[int, list[str]]] = {}
    for each in columns:
        for key, names in each.victim.items():
            victim[key] = max(victim.get(key, []), names, key=len)
        for key, slots in each.segment.items():
            for number, names in slots.items():
                current = segment.setdefault(key, {}).get(number, [])
                segment[key][number] = max(current, names, key=len)
    return Columns(victim, segment)


def _matches(frame: pd.DataFrame, names: Iterable[str], pattern: re.Pattern[str]) -> pd.Series:
    """Rows where any of the columns has a value matching the pattern."""
    mask = pd.Series(False, index=frame.index)
    for name in names:
        if name not in frame:
            continue
        values = frame[name]
        hits = [value for value in values.dropna().unique() if pattern.search(str(value))]
        mask |= values.isin(hits)
    return mask


# --- the report ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Offense:
    """The values of one offense's victims, converted once and shared by its paragraphs."""

    victims: int
    victim_sex: pd.Series | None
    victim_age: pd.Series | None
    weapon_labels: pd.DataFrame | None  # as recorded, several per victim
    weapons: pd.DataFrame | None  # weapon groups of the profiles
    firearm: pd.Series | None
    location_label: pd.Series | None  # as recorded
    location: pd.Series | None  # location groups of the profiles
    hour: pd.Series | None  # time of day bands
    circumstances: pd.DataFrame | None  # circumstance groups, several per victim
    known: pd.Series  # offender information was recorded
    relationship: pd.Series | None
    offender_sex: pd.Series | None
    offender_age: pd.Series | None
    offenders: pd.Series | None  # one, two, three or more
    using: pd.Series | None  # alcohol or drugs suspected


def build_report(frame: pd.DataFrame, columns: Columns, *, name: str, years: list[Year]) -> str:
    # Every paragraph names the data: the generator cuts the report into fragments, and
    # each fragment must say where its figures come from.
    # The country is in the sentence itself, not in the brackets: models retelling the
    # report tend to drop what is in brackets.
    scope = "agencies participating in NIBRS, not nationally representative"
    if len(years) > 1:
        scope = f"pooled, {scope}"
    prefix = f"In {name} data from the United States ({scope}),"
    sections = [_introduction(name, years, frame, columns)]
    for offense, pattern in OFFENSES:
        regex = re.compile(pattern, re.I)
        victims = frame[_matches(frame, columns.victim["offense"], regex)].reset_index(drop=True)
        if len(victims) < _MIN_PROFILE:
            continue
        data = _offense_data(victims, columns, regex)
        base = f"{prefix} {offense} —"
        paragraphs = [*_overview(base, offense, victims, columns, data)]
        paragraphs += [p for p in _profiles(base, offense, data) if p]
        sections.append(f"{offense.capitalize()}\n\n" + "\n\n".join(paragraphs))
    return "\n\n\n".join(sections) + "\n"


def _introduction(name: str, years: list[Year], frame: pd.DataFrame, columns: Columns) -> str:
    records = sum(y.records for y in years)
    midnight = sum(y.midnight for y in years) / records if records else 0.0
    if len(years) == 1:
        source = f"the victim-level file of {name} ({records:,} victim records of all kinds)"
        pooling = ""
    else:
        per_year = ", ".join(
            f"{y.year}: {y.records:,}" for y in sorted(years, key=lambda y: y.year)
        )
        source = (
            f"the victim-level files of {len(years)} years pooled together ({records:,} victim "
            f"records of all kinds; {per_year})"
        )
        pooling = (
            " The years are pooled to have enough cases for rare combinations; agency "
            "participation and some classifications differ between years, so the pooled "
            "shares are averages over the agencies that reported in each year."
        )
    return f"""\
Offender profiles from {name} (United States): what the circumstances of violent and \
sexual offenses say about the offender

The National Incident-Based Reporting System (NIBRS) is part of the FBI's Uniform Crime \
Reporting Program. Participating law enforcement agencies report every criminal incident \
and arrest: for each Group A offense, they record the offense, the location, weapons or \
force used, and the victims, offenders and arrestees. An incident is one or more offenses \
committed by the same offender, or group of offenders acting in concert, at the same time \
and place.

This report was computed from {source}. For each offense it first describes the victims \
and the offense, and then gives offender profiles: among victims with a given feature — \
the victim's sex and age, the weapon, the location, the time of day or the circumstances \
— how the offenders were distributed by their relationship to the victim, sex and age, \
how often there were several offenders, and how often the offender was suspected of using \
alcohol or drugs. These are base rates for forming and ranking investigative hypotheses; \
they do not predict who committed a particular crime and must not be used to exclude a \
suspect.

Method: the report covers victims who are individuals (not businesses, society or the \
government) and against whom the offense was committed; a victim of several offenses \
appears in several sections. The offender is the first offender linked to the victim. \
Offender characteristics are given as shares among victims where the characteristic was \
recorded, and that share is stated. The weapon, the location and suspected alcohol or \
drug use come from the incident's record of that offense; the number of offenders and the \
clearance refer to the incident. Relationships are grouped as follows: intimate partner \
(spouse, common-law spouse, ex-spouse, boyfriend or girlfriend, ex-relationship, same-sex \
relationship), family member (parent, child, sibling, grandparent, grandchild, in-law, \
step-relatives, foster relatives, other family), acquaintance or otherwise known \
(acquaintance, friend, neighbor, babysittee, employee, employer, non-intimate cohabitant, \
child of a boyfriend or girlfriend, otherwise known) and stranger. Race and ethnicity were \
left out of this report.{pooling}

Bias: offender information is recorded mostly when a suspect was identified. Offenders \
who are strangers to the victim are identified less often than partners, relatives and \
acquaintances, so among recorded offenders strangers are under-represented. The profiles \
describe the offenders of crimes recorded by the police with offender information, not \
all offenders. The time of day leaves out the hour 00:00–00:59: it holds {midnight:.1%} \
of all victim records, several times more than the neighbouring hours, so it evidently \
also contains incidents whose time was unknown.

Limitations: the data are not a nationally representative sample of crime in the United \
States, because agency participation varies between states and cities. The figures \
describe crimes known to and recorded by the police, classified when they were reported, \
not court findings; crimes that were never reported, such as many sexual offenses, are \
missing. A case is counted as cleared by arrest if the incident has at least one arrestee \
in the data."""


def _offense_data(victims: pd.DataFrame, columns: Columns, regex: re.Pattern[str]) -> Offense:
    def values(key: str, convert: Callable[[Any], Any] = _known) -> pd.Series | None:
        if key not in columns.victim:
            return None
        return _convert(victims[columns.victim[key][0]], convert)

    record = _offense_record(victims, columns, regex)
    weapon_labels = weapons = firearm = location_label = location = None
    using = circumstances = None
    if "weapon" in record:
        weapon_labels = _convert_cells(record["weapon"], _weapon)
        firearm = weapon_labels.apply(lambda column: column.str.contains(_FIREARM, na=False))
        firearm = firearm.any(axis=1)
        weapons = _convert_cells(weapon_labels, lambda value: _group(value, _WEAPONS))
    if "location" in record:
        location_label = _convert(record["location"][0], _known)
        location = _convert(location_label, lambda value: _group(value, _LOCATIONS))
    if "using" in record:
        cells = _convert_cells(record["using"], _known)
        using = cells.apply(lambda column: column.str.contains(_ALCOHOL_OR_DRUGS, na=False))
        using = using.any(axis=1)
    if "circumstance" in columns.victim:
        cells = victims[[n for n in columns.victim["circumstance"] if n in victims]]
        circumstances = _convert_cells(cells, lambda value: _group(value, _CIRCUMSTANCES))
    if "offender_number" in columns.victim:
        known = ~_convert(victims["offender_number"], _offender_unknown).astype(bool)
    else:
        known = pd.Series(True, index=victims.index)
    offenders = None
    if "offenders" in columns.victim:
        count = _numeric(victims["offenders"])
        offenders = _convert(count, _count_band).where(known, None)
    return Offense(
        victims=len(victims),
        victim_sex=values("victim_sex", _sex),
        victim_age=values("victim_age", _age_band),
        weapon_labels=weapon_labels,
        weapons=weapons,
        firearm=firearm,
        location_label=location_label,
        location=location,
        hour=values("hour", _hour_band),
        circumstances=circumstances,
        known=known,
        relationship=values("relationship", _relationship),
        offender_sex=values("offender_sex", _sex),
        offender_age=values("offender_age", _offender_age_band),
        offenders=offenders,
        using=using,
    )


def _overview(
    base: str, offense: str, victims: pd.DataFrame, columns: Columns, data: Offense
) -> Iterable[str]:
    text = (
        f"{base} overview, {data.victims:,} victims. Offender information was recorded for "
        f"{_pct(data.known.sum(), data.victims)} of them."
    )
    if "arrestees" in columns.victim:
        text += " " + _clearance(victims, columns)
    yield text
    if data.victim_sex is not None:
        yield _shares(base, "sex of the victim", data.victim_sex)
    if data.victim_age is not None:
        yield _shares(base, "age of the victim", data.victim_age, order=_band_order(_AGE_BANDS))
    if data.weapon_labels is not None and data.firearm is not None:
        sentence = _multi_shares(base, "weapon or force involved", data.weapon_labels)
        known = data.weapon_labels.notna().any(axis=1)
        if known.sum():
            sentence += (
                f" A firearm of any type was involved for "
                f"{_pct((data.firearm & known).sum(), known.sum())}."
            )
        yield sentence
    if data.location_label is not None:
        yield _shares(base, "location of the offense", data.location_label)
    if data.hour is not None and "hour" in columns.victim:
        yield _time_of_day(base, victims["hour"])
    if "circumstance" in columns.victim and offense in _WITH_CIRCUMSTANCES:
        cells = victims[[n for n in columns.victim["circumstance"] if n in victims]]
        yield _multi_shares(base, "circumstances of the incident", cells)


def _profiles(base: str, offense: str, data: Offense) -> Iterable[str | None]:
    everyone = pd.Series(True, index=data.known.index)
    yield _profile(base, "all victims", everyone, data)
    if data.victim_sex is not None and data.victim_age is not None:
        for sex in ("female", "male"):
            for _, _, band in _AGE_BANDS:
                rows = (data.victim_sex == sex) & (data.victim_age == band)
                yield _profile(base, f"{sex} victims {_aged(band)}", rows, data)
    conditions: list[tuple[str, pd.Series]] = []
    if data.weapons is not None:
        conditions += [
            (f"offenses {phrase}", (data.weapons == phrase).any(axis=1)) for phrase, _ in _WEAPONS
        ]
    if data.location is not None:
        conditions += [(f"offenses {phrase}", data.location == phrase) for phrase, _ in _LOCATIONS]
    if data.hour is not None:
        conditions += [
            (f"offenses {_HOUR_PHRASES[band]}", data.hour == band) for _, _, band in _HOUR_BANDS
        ]
    if data.circumstances is not None and offense in _WITH_CIRCUMSTANCES:
        conditions += [
            (f"offenses {phrase}", (data.circumstances == phrase).any(axis=1))
            for phrase, _ in _CIRCUMSTANCES
        ]
    for label, rows in conditions:
        yield _profile(base, label, rows, data)
    if offense in _DETAILED and data.victim_sex is not None:
        for sex in ("female", "male"):
            victims_of_sex = data.victim_sex == sex
            if data.location is not None:
                for phrase, _ in _LOCATIONS:
                    rows = victims_of_sex & (data.location == phrase)
                    yield _profile(base, f"{sex} victims of offenses {phrase}", rows, data)
            if data.weapons is not None:
                for phrase, _ in _WEAPONS:
                    rows = victims_of_sex & (data.weapons == phrase).any(axis=1)
                    yield _profile(base, f"{sex} victims of offenses {phrase}", rows, data)


def _profile(base: str, condition: str, rows: pd.Series, data: Offense) -> str | None:
    """The offenders of the victims in `rows`, or None if there are too few of them."""
    victims = int(rows.sum())
    known = int((data.known & rows).sum())
    if victims < _MIN_PROFILE or known < _MIN_GROUP:
        return None
    parts = [
        f"{base} offender profile for {condition}: {victims:,} victims; offender information "
        f"was recorded for {_pct(known, victims)}."
    ]
    for label, values, order in (
        ("Relationship of the victim to the offender", data.relationship, None),
        ("Offender sex", data.offender_sex, None),
        ("Offender age", data.offender_age, _band_order(_OFFENDER_AGE_BANDS)),
    ):
        if values is not None and (part := _distribution(label, values[rows], order)):
            parts.append(part)
    if data.offenders is not None:
        count = data.offenders[rows].dropna()
        if len(count) >= _MIN_GROUP:
            several = int((count != "one").sum())
            parts.append(f"Two or more offenders: {_pct(several, len(count))}.")
    if data.using is not None:
        using = _pct(data.using[rows].sum(), victims)
        parts.append(f"Offender suspected of using alcohol or drugs: {using}.")
    return " ".join(parts)


def _distribution(label: str, values: pd.Series, order: list[str] | None) -> str | None:
    known = values.dropna()
    if len(known) < _MIN_GROUP:
        return None
    counts = known.value_counts()
    counts = counts[counts > 0]
    if order:
        counts = counts.reindex([value for value in order if value in counts.index])
    shares = ", ".join(
        f"{_label(value)} {_pct(count, len(known))}" for value, count in counts.items()
    )
    return f"{label} (recorded for {_pct(len(known), len(values))}): {shares}."


def _aged(band: str) -> str:
    return band if band.startswith("under") else f"aged {band}"


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
                if position < len(names) and names[position] in victims:
                    values = victims[names[position]].astype(object)
                    column = column.where(record_number != number, values)
            frame[position] = column
        result[key] = pd.DataFrame(frame)
    return result


def _shares(base: str, aspect: str, values: pd.Series, order: list[str] | None = None) -> str:
    """One value per victim: unknown share first, then the shares among known values."""
    known = values.dropna()
    counts = known.value_counts()
    counts = counts[counts > 0]
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
    midnight = int((_numeric(hours) == 0).sum())
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


def _clearance(victims: pd.DataFrame, columns: Columns) -> str:
    arrests = _numeric(victims["arrestees"]).fillna(0)
    status = pd.Series("not cleared", index=victims.index)
    if "cleared_exceptionally" in columns.victim:
        reason = _convert(victims["cleared_exceptionally"], _known)
        status[reason.notna()] = "cleared exceptionally"
    status[arrests > 0] = "cleared by arrest"
    counts = status.value_counts()
    parts = [f"{value} {_pct(count, len(status))}" for value, count in counts.items()]
    return f"Clearance of the incident: {', '.join(parts)}."


def _pct(part: float, whole: float) -> str:
    share = 100 * part / whole
    return "<0.1%" if 0 < share < 0.05 else f"{share:.1f}%"


def _label(value: object) -> str:
    """A category as it reads in a sentence: "Victim was Spouse" → "spouse"."""
    return re.sub(r"^victim was ", "", str(value), flags=re.I).lower()


def _band_order(bands: Iterable[tuple[int, int, str]]) -> list[str]:
    return [label for _, _, label in bands]


def _years_label(years: list[int]) -> str:
    contiguous = years == list(range(years[0], years[-1] + 1))
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}–{years[-1]}" if contiguous else ", ".join(map(str, years))


# --- value conversion ----------------------------------------------------------------


def _convert(values: pd.Series, convert: Callable[[Any], Any]) -> pd.Series:
    """Converts each distinct value once: a column has millions of cells but few values."""
    mapping = {value: convert(value) for value in values.unique()}
    return pd.Series(values.map(mapping), index=values.index, dtype=object)


def _convert_cells(cells: pd.DataFrame, convert: Callable[[Any], Any]) -> pd.DataFrame:
    return cells.apply(lambda column: _convert(column, convert))


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str), errors="coerce")


def _victims_per_value(cells: pd.DataFrame) -> pd.Series:
    """How many victims have each value at least once among their columns."""
    pairs = cells.stack().dropna().rename("value").reset_index()
    return pairs.drop_duplicates([pairs.columns[0], "value"])["value"].value_counts()


def _known(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if _UNKNOWN.search(text) else text


def _group(value: object, groups: Iterable[tuple[str, str]]) -> str | None:
    known = _known(value)
    if known is None:
        return None
    return next((name for name, pattern in groups if re.search(pattern, known, re.I)), None)


def _weapon(value: object) -> str | None:
    known = _known(value)
    return "no weapon (none)" if known and known.lower() == "none" else known


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


def _offender_unknown(value: object) -> bool:
    return _UNKNOWN.search(str(value)) is not None or str(value).strip() in {"0", "0.0"}


def _count_band(count: float) -> str | None:
    if pd.isna(count):
        return None
    return "one" if count == 1 else "two" if count == 2 else "three or more"


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


def _offender_age_band(value: object) -> str | None:
    age = _age(value)
    return None if age is None else _band(int(age), _OFFENDER_AGE_BANDS)


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
