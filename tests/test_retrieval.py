"""Acceptance tests for retrieval (seam 2 in issue #1): `Retriever.search` against the seeded test
database (tests/seed.py) with the real embedder.

The queries were checked against the real embedder: for "BIR Form 1904" vector search ranks the
Form 1902 Passage first, so keyword search has to find the exact term on its own.
"""

from datetime import date

import pytest
from fakes import StaticRetriever
from seed import (
    CERTIFICATE_REPLACEMENT,
    LOAN_APPLICATION,
    LOAN_ELIGIBILITY,
    LOAN_INTEREST,
    ONE_TIME_TAXPAYER,
    PASSAGES,
    retrieved,
)

from guardrag.db import connect
from guardrag.retrieval import (
    VECTOR_SEARCH_SQL,
    HybridRetriever,
    KeywordRetriever,
    Retriever,
    VectorRetriever,
    build_retriever,
)

FILIPINO_PARAPHRASE = ("Paano mag-apply ng salary loan sa SSS?", LOAN_APPLICATION.id)
ENGLISH_PARAPHRASE = ("I lost my certificate of registration", CERTIFICATE_REPLACEMENT.id)
EXACT_TERM = ("BIR Form 1904", ONE_TIME_TAXPAYER.id)


@pytest.fixture
def vector(embedder, seeded_db_url) -> VectorRetriever:
    return VectorRetriever(embedder, seeded_db_url)


@pytest.fixture
def keyword(seeded_db_url) -> KeywordRetriever:
    return KeywordRetriever(seeded_db_url)


@pytest.fixture
def hybrid(embedder, seeded_db_url) -> Retriever:
    return build_retriever("hybrid", embedder, seeded_db_url)


def ids(results) -> list[int]:
    return [r.passage_id for r in results]


def assert_ranked_best_first(results) -> None:
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# --- vector (#5) --------------------------------------------------------------------------------


@pytest.mark.db
class TestVector:
    @pytest.mark.parametrize("query, expected", [FILIPINO_PARAPHRASE, ENGLISH_PARAPHRASE])
    def test_a_paraphrase_finds_the_matching_passage_first(self, vector, query, expected) -> None:
        assert ids(vector.search(query, k=3))[0] == expected

    @pytest.mark.parametrize("k", [1, 3, len(PASSAGES)])
    def test_k_is_respected(self, vector, k) -> None:
        assert len(vector.search("salary loan", k=k)) == k

    def test_results_are_ranked_best_first(self, vector) -> None:
        assert_ranked_best_first(vector.search("salary loan interest", k=len(PASSAGES)))

    def test_results_carry_document_metadata(self, vector) -> None:
        [top] = vector.search(ENGLISH_PARAPHRASE[0], k=1)

        assert top == retrieved(CERTIFICATE_REPLACEMENT, top.score)
        assert top.as_of == date(2025, 3, 1)  # the printed effective date

    def test_as_of_falls_back_to_the_fetch_date(self, vector) -> None:
        [top] = vector.search(FILIPINO_PARAPHRASE[0], k=1)

        assert top.agency == "SSS"
        assert top.as_of == date(2026, 10, 1)

    def test_search_can_use_the_hnsw_cosine_index(self, embedder, seeded_db_url) -> None:
        """Six rows are cheaper to scan, so sequential scans are disabled to see whether the
        planner can serve the query from the index at all."""
        [query] = embedder.embed(["salary loan"])
        with connect(seeded_db_url) as conn:
            conn.execute("SET enable_seqscan = off")
            plan = conn.execute("EXPLAIN " + VECTOR_SEARCH_SQL, {"query": query, "k": 3}).fetchall()

        assert any("Index Scan using passages_embedding_idx" in line for (line,) in plan)


# --- keyword (#6) -------------------------------------------------------------------------------


@pytest.mark.db
class TestKeyword:
    def test_an_exact_form_number_is_found_first(self, keyword) -> None:
        query, expected = EXACT_TERM
        assert ids(keyword.search(query, k=3))[0] == expected

    def test_an_exact_peso_amount_is_found_first(self, keyword) -> None:
        assert ids(keyword.search("P100", k=3)) == [CERTIFICATE_REPLACEMENT.id]

    def test_query_terms_are_or_combined(self, keyword) -> None:
        """No Passage holds every term; AND-ing them would find nothing."""
        results = keyword.search("how much interest does the salary loan charge per year", k=3)

        assert ids(results)[0] == LOAN_INTEREST.id

    def test_k_is_respected_and_results_ranked_best_first(self, keyword) -> None:
        results = keyword.search("salary loan", k=2)

        assert len(results) == 2
        assert set(ids(results)) <= {LOAN_ELIGIBILITY.id, LOAN_APPLICATION.id, LOAN_INTEREST.id}
        assert_ranked_best_first(results)

    def test_no_matching_term_finds_nothing(self, keyword) -> None:
        assert keyword.search("xyzzy plugh", k=3) == []

    def test_results_carry_document_metadata(self, keyword) -> None:
        top = keyword.search(EXACT_TERM[0], k=1)[0]

        assert top == retrieved(ONE_TIME_TAXPAYER, top.score)


# --- hybrid (#6) --------------------------------------------------------------------------------


def test_rrf_favours_a_passage_both_lists_rank_well() -> None:
    """With k=60: ranked 2nd by both lists scores 2/62 ≈ 0.0323, beating 1st by one list
    (1/61 ≈ 0.0164)."""
    a, b, c, d, e = (retrieved(p, 0.5) for p in PASSAGES[:5])
    hybrid = HybridRetriever(StaticRetriever([a, b, c]), StaticRetriever([d, b, e]))

    results = hybrid.search("anything", k=5)

    assert ids(results)[0] == b.passage_id
    assert results[0].score == pytest.approx(2 / 62)
    assert set(ids(results)) == {p.passage_id for p in (a, b, c, d, e)}
    assert_ranked_best_first(results)


def test_rrf_k_respected() -> None:
    a, b, c, d = (retrieved(p, 0.5) for p in PASSAGES[:4])
    hybrid = HybridRetriever(StaticRetriever([a, b]), StaticRetriever([c, d]))

    assert len(hybrid.search("anything", k=3)) == 3


@pytest.mark.db
class TestHybrid:
    def test_an_exact_form_number_is_found(self, hybrid) -> None:
        """Vector ranks it 2nd and keyword 1st, so RRF ties it with the Form 1902 Passage."""
        query, expected = EXACT_TERM
        assert expected in ids(hybrid.search(query, k=2))

    @pytest.mark.parametrize("query, expected", [FILIPINO_PARAPHRASE, ENGLISH_PARAPHRASE])
    def test_a_paraphrase_is_found(self, hybrid, query, expected) -> None:
        assert expected in ids(hybrid.search(query, k=3))

    def test_results_carry_document_metadata(self, hybrid) -> None:
        [top] = hybrid.search("affidavit of loss", k=1)

        assert top == retrieved(CERTIFICATE_REPLACEMENT, top.score)


@pytest.mark.parametrize(
    "mode, kind",
    [("vector", VectorRetriever), ("keyword", KeywordRetriever), ("hybrid", HybridRetriever)],
)
def test_the_retriever_is_chosen_by_configuration(embedder, mode, kind) -> None:
    assert isinstance(build_retriever(mode, embedder), kind)
