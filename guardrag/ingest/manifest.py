from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from guardrag.domain import DocumentKind

DEFAULT_MANIFEST = Path(__file__).with_name("sources.yaml")


@dataclass(frozen=True)
class SourceEntry:
    url: str
    agency: str
    kind: DocumentKind
    title: str
    effective_date: date | None = None


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[SourceEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [SourceEntry(**entry) for entry in raw["documents"]]
