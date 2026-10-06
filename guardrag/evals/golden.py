"""The Golden Set (CONTEXT.md): questions tagged answerable, with where the answer lives, or
Out-of-Corpus. The file is evals/golden_set.yaml.

Targets name a document and section rather than Passage ids, so they survive re-chunking.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args

import yaml

from guardrag.retrieval import RetrievedPassage

GOLDEN_SET_PATH = Path(__file__).parents[2] / "evals" / "golden_set.yaml"

Language = Literal["en", "fil", "taglish"]
QuestionKind = Literal["answerable", "out_of_corpus"]


class GoldenSetError(ValueError):
    pass


@dataclass(frozen=True)
class Location:
    """A document, by its manifest title, and optionally one of its sections."""

    document: str
    section: str | None = None

    def holds(self, passage: RetrievedPassage) -> bool:
        return passage.document_title == self.document and (
            self.section is None or passage.section == self.section
        )


@dataclass(frozen=True)
class Target:
    """One part of the answer, found in any of `locations`."""

    locations: tuple[Location, ...]

    def met_by(self, passage: RetrievedPassage) -> bool:
        return any(location.holds(passage) for location in self.locations)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    language: Language
    kind: QuestionKind
    # Every Target is needed for a full answer: a question spanning two Agencies has two.
    expected: tuple[Target, ...] = ()
    tags: tuple[str, ...] = ()
    # For an Out-of-Corpus Question, the Agency the Refusal should point to.
    agency: str | None = None
    note: str | None = field(default=None, compare=False)


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = [_question(entry) for entry in raw["questions"]]
    duplicates = [id for id, n in Counter(q.id for q in questions).items() if n > 1]
    if duplicates:
        raise GoldenSetError(f"duplicate ids: {duplicates}")
    return questions


def _question(entry: dict) -> GoldenQuestion:
    id = entry.get("id")
    language, kind = entry.get("language"), entry.get("kind")
    if language not in get_args(Language):
        raise GoldenSetError(f"{id}: language must be one of {get_args(Language)}")
    if kind not in get_args(QuestionKind):
        raise GoldenSetError(f"{id}: kind must be one of {get_args(QuestionKind)}")
    expected = tuple(_target(item) for item in entry.get("expected") or [])
    if kind == "answerable" and not expected:
        raise GoldenSetError(f"{id}: an answerable question needs at least one expected target")
    if kind == "out_of_corpus" and expected:
        raise GoldenSetError(f"{id}: an Out-of-Corpus Question has no expected targets")
    return GoldenQuestion(
        id=id,
        question=entry["question"],
        language=language,
        kind=kind,
        expected=expected,
        tags=tuple(entry.get("tags") or ()),
        agency=entry.get("agency"),
        note=entry.get("note"),
    )


def _target(item: dict) -> Target:
    locations = item["any_of"] if "any_of" in item else [item]
    return Target(tuple(Location(loc["document"], loc.get("section")) for loc in locations))
