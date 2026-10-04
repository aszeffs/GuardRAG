"""Find the Passages most likely to answer a question.

Owned by the author: see issues #5 (vector) and #6 (keyword, hybrid) and tests/test_retrieval.py.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from guardrag.embedder import Embedder

RetrieverMode = Literal["vector", "keyword", "hybrid"]


@dataclass(frozen=True)
class RetrievedPassage:
    """A Passage returned by a Retriever, with the document metadata a Citation needs."""

    passage_id: int
    text: str
    section: str | None
    page: int | None
    document_title: str
    agency: str
    url: str
    as_of: date
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        """Return at most `k` Passages for `query`, best first.

        Scores are comparable only within one Retriever, and never increase down the list. An
        empty list means nothing matched.
        """
        ...


class VectorRetriever:
    """Cosine similarity between the embedded query and Passage embeddings (HNSW index)."""

    def __init__(self, embedder: Embedder, db_url: str | None = None) -> None:
        self.embedder = embedder
        self.db_url = db_url

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        raise NotImplementedError


class KeywordRetriever:
    """Postgres full-text search over Passages, OR-combining the query's terms."""

    def __init__(self, db_url: str | None = None) -> None:
        self.db_url = db_url

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        raise NotImplementedError


class HybridRetriever:
    """Reciprocal Rank Fusion of two rankings: score(p) = sum of 1 / (rrf_k + rank(p))."""

    def __init__(self, vector: Retriever, keyword: Retriever, rrf_k: int = 60) -> None:
        self.vector = vector
        self.keyword = keyword
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[RetrievedPassage]:
        raise NotImplementedError


def build_retriever(
    mode: RetrieverMode, embedder: Embedder, db_url: str | None = None
) -> Retriever:
    """The Retriever named by the `RETRIEVER` setting, so the three can be compared (#9)."""
    match mode:
        case "vector":
            return VectorRetriever(embedder, db_url)
        case "keyword":
            return KeywordRetriever(db_url)
        case "hybrid":
            return HybridRetriever(VectorRetriever(embedder, db_url), KeywordRetriever(db_url))
