"""Scripted stand-ins for the `/ask` dependencies. Each records how it was called."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

from guardrag.answer import DraftAnswer
from guardrag.retrieval import RetrievedPassage


@dataclass
class StaticRetriever:
    """Returns the same ranked Passages for every query, cut to `k`."""

    results: list[RetrievedPassage]
    queries: list[tuple[str, int]] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        self.queries.append((query, k))
        return self.results[:k]


@dataclass
class ScriptedRetriever:
    """Returns a preset ranking per query (none for an unknown query), cut to `k`."""

    rankings: dict[str, list[RetrievedPassage]]
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        self.queries.append(query)
        return self.rankings.get(query, [])[:k]


@dataclass
class ScriptedLLM:
    """Returns preset drafts in order, including ones citing Passages it wasn't given."""

    drafts: list[DraftAnswer]
    calls: list[tuple[str, list[RetrievedPassage]]] = field(default_factory=list)

    def draft(self, question: str, passages: Sequence[RetrievedPassage]) -> DraftAnswer:
        self.calls.append((question, list(passages)))
        return self.drafts[len(self.calls) - 1]


@dataclass
class FakeClassifier:
    """Flags text containing any of `triggers`; raises `error` instead, if set."""

    triggers: tuple[str, ...] = ()
    error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def is_injection(self, text: str) -> bool:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return any(t.lower() in text.lower() for t in self.triggers)


@dataclass
class FakeGroqClient:
    """Mimics `groq.Groq().chat.completions.create`, replying with preset message contents.

    Raises `error` instead, if set.
    """

    replies: list[str | None]
    error: Exception | None = None
    requests: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.replies[len(self.requests) - 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
