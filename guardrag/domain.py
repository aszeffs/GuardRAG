"""Core types shared by ingest, retrieval and answering. Names follow CONTEXT.md."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

DocumentKind = Literal["service", "statute"]


@dataclass(frozen=True)
class Section:
    """A headed block of a document's text, as extracted from HTML or PDF."""

    heading: str | None
    text: str
    page: int | None = None


@dataclass(frozen=True)
class SourceDocument:
    url: str
    title: str
    agency: str
    kind: DocumentKind
    fetched_at: date
    sections: list[Section] = field(default_factory=list)
    effective_date: date | None = None

    @property
    def as_of(self) -> date:
        return self.effective_date or self.fetched_at


@dataclass(frozen=True)
class Passage:
    """A short excerpt of one document within a single section; the unit retrieved and cited."""

    ordinal: int
    text: str
    section: str | None
    page: int | None
