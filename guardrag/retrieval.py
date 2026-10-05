"""Find the Passages most likely to answer a question.

Vector search is #5; keyword and hybrid are #6. See tests/test_retrieval.py.
"""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from psycopg.rows import class_row

from guardrag.config import RetrieverMode
from guardrag.db import connect
from guardrag.embedder import Embedder

# pgvector's default hnsw.ef_search: an HNSW scan yields at most this many rows.
HNSW_EF_SEARCH_DEFAULT = 40

# Ordering by the bare `<=>` distance with a LIMIT is what lets the planner use the HNSW
# `vector_cosine_ops` index on passages.embedding. Score is cosine similarity.
VECTOR_SEARCH_SQL = """
    SELECT p.id AS passage_id, p.text, p.section, p.page, d.title AS document_title, d.agency,
           d.url, coalesce(d.effective_date, d.fetched_at) AS as_of,
           1 - (p.embedding <=> %(query)s::vector) AS score
    FROM passages p JOIN documents d ON d.id = p.document_id
    ORDER BY p.embedding <=> %(query)s::vector
    LIMIT %(k)s
"""


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
        [embedding] = self.embedder.embed([query])
        with connect(self.db_url) as conn:
            ef_search = max(HNSW_EF_SEARCH_DEFAULT, k)
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            with conn.cursor(row_factory=class_row(RetrievedPassage)) as cur:
                return cur.execute(VECTOR_SEARCH_SQL, {"query": embedding, "k": k}).fetchall()


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
