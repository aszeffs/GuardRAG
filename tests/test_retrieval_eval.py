"""The retrieval eval (#9): recall@k and MRR over the Golden Set, through `Retriever.search`.

Scoring runs against scripted retrievers built from the seed Passages, so it needs no database.
"""

from pathlib import Path

import pytest
from fakes import ScriptedRetriever
from seed import (
    CERTIFICATE_REPLACEMENT,
    LOAN_APPLICATION,
    LOAN_ELIGIBILITY,
    LOAN_INTEREST,
    ONE_TIME_TAXPAYER,
    TIN_FOR_EMPLOYEES,
    SeedPassage,
    retrieved,
)

from guardrag.evals.golden import (
    GoldenQuestion,
    GoldenSetError,
    Location,
    Target,
    load_golden_set,
)
from guardrag.evals.retrieval import evaluate, passes_gate
from guardrag.ingest.manifest import load_manifest

ROOT = Path(__file__).parents[1]


def at(passage: SeedPassage) -> Target:
    """A Target met only by `passage`'s own document and section."""
    return Target((Location(passage.document.title, passage.section),))


def question(
    id: str, *expected: Target, language: str = "en", kind: str = "answerable"
) -> GoldenQuestion:
    return GoldenQuestion(
        id=id, question=f"question {id}", language=language, kind=kind, expected=expected
    )


def ranking(*passages: SeedPassage):
    return [retrieved(p, score=1.0 - i / 10) for i, p in enumerate(passages)]


def test_a_target_in_the_top_k_is_recalled_and_one_below_it_is_not() -> None:
    q = question("q", at(LOAN_ELIGIBILITY))
    retriever = ScriptedRetriever(
        {q.question: ranking(LOAN_APPLICATION, LOAN_INTEREST, LOAN_ELIGIBILITY)}
    )

    assert evaluate([q], retriever, k=3).overall.recall == 1.0
    assert evaluate([q], retriever, k=2).overall.recall == 0.0


def test_reciprocal_rank_comes_from_the_first_passage_meeting_any_target() -> None:
    q = question("q", at(LOAN_INTEREST), at(LOAN_APPLICATION))
    retriever = ScriptedRetriever(
        {q.question: ranking(TIN_FOR_EMPLOYEES, LOAN_APPLICATION, LOAN_INTEREST)}
    )

    assert evaluate([q], retriever, k=5).overall.mrr == 0.5


def test_recall_is_the_share_of_a_questions_targets_found() -> None:
    q = question("two-agencies", at(TIN_FOR_EMPLOYEES), at(LOAN_ELIGIBILITY))
    retriever = ScriptedRetriever({q.question: ranking(LOAN_ELIGIBILITY, LOAN_INTEREST)})

    scores = evaluate([q], retriever, k=5)

    assert scores.overall.recall == 0.5
    assert scores.misses == ["two-agencies"]


def test_a_target_without_a_section_is_met_by_any_passage_of_its_document() -> None:
    q = question("q", Target((Location(LOAN_INTEREST.document.title),)))
    retriever = ScriptedRetriever({q.question: ranking(TIN_FOR_EMPLOYEES, LOAN_INTEREST)})

    assert evaluate([q], retriever, k=5).overall.mrr == 0.5


def test_a_target_is_met_by_any_of_its_locations() -> None:
    either = Target(
        (
            Location(ONE_TIME_TAXPAYER.document.title, ONE_TIME_TAXPAYER.section),
            Location(LOAN_INTEREST.document.title, LOAN_INTEREST.section),
        )
    )
    q = question("q", either)
    retriever = ScriptedRetriever({q.question: ranking(LOAN_INTEREST)})

    assert evaluate([q], retriever, k=5).overall.recall == 1.0


def test_the_same_section_in_another_document_does_not_count() -> None:
    q = question("q", Target((Location("Some other document", LOAN_INTEREST.section),)))
    retriever = ScriptedRetriever({q.question: ranking(LOAN_INTEREST)})

    assert evaluate([q], retriever, k=5).overall.recall == 0.0


def test_scores_average_over_answerable_questions_and_split_by_language() -> None:
    found = question("found", at(LOAN_ELIGIBILITY))
    missed = question("missed", at(CERTIFICATE_REPLACEMENT), language="fil")
    out_of_corpus = question("ooc", kind="out_of_corpus")
    retriever = ScriptedRetriever(
        {
            found.question: ranking(LOAN_ELIGIBILITY),
            missed.question: ranking(LOAN_INTEREST),
            out_of_corpus.question: ranking(LOAN_INTEREST),
        }
    )

    scores = evaluate([found, missed, out_of_corpus], retriever, k=5)

    assert (scores.overall.questions, scores.overall.recall, scores.overall.mrr) == (2, 0.5, 0.5)
    assert scores.by_language["en"].recall == 1.0
    assert scores.by_language["fil"].recall == 0.0
    assert out_of_corpus.question not in retriever.queries


def test_the_gate_fails_below_the_threshold_only() -> None:
    q = question("q", at(LOAN_ELIGIBILITY), at(LOAN_INTEREST))
    scores = evaluate([q], ScriptedRetriever({q.question: ranking(LOAN_ELIGIBILITY)}), k=5)

    assert passes_gate(scores, min_recall=0.5)
    assert not passes_gate(scores, min_recall=0.51)


GOOD_ENTRY = """
  - id: salary-loan
    question: How do I apply for an SSS salary loan?
    language: en
    kind: answerable
    expected:
      - document: "SSS: Salary Loan"
        section: How to Apply
"""


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(GOOD_ENTRY + GOOD_ENTRY, id="duplicate id"),
        pytest.param(GOOD_ENTRY.replace("language: en", "language: es"), id="unknown language"),
        pytest.param(
            GOOD_ENTRY.replace("kind: answerable", "kind: out_of_corpus"), id="ooc with target"
        ),
        pytest.param(
            GOOD_ENTRY.split("    expected:")[0] + "    expected: []\n", id="answerable no target"
        ),
    ],
)
def test_a_malformed_golden_set_is_rejected(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "golden.yaml"
    path.write_text("questions:" + bad, encoding="utf-8")

    with pytest.raises(GoldenSetError):
        load_golden_set(path)


def test_a_golden_set_target_may_list_alternative_locations(tmp_path: Path) -> None:
    path = tmp_path / "golden.yaml"
    path.write_text(
        "questions:"
        + GOOD_ENTRY
        + "      - any_of:\n"
        + "          - document: A\n"
        + "          - document: B\n"
        + "            section: S\n",
        encoding="utf-8",
    )

    [q] = load_golden_set(path)

    assert q.expected == (
        Target((Location("SSS: Salary Loan", "How to Apply"),)),
        Target((Location("A"), Location("B", "S"))),
    )


@pytest.fixture(scope="module")
def golden() -> list[GoldenQuestion]:
    return load_golden_set()


class TestTheGoldenSet:
    """The checked-in Golden Set meets the shape agreed in #9."""

    def test_it_has_40_to_50_questions(self, golden) -> None:
        assert 40 <= len(golden) <= 50

    def test_about_a_fifth_is_filipino_or_taglish(self, golden) -> None:
        share = sum(q.language != "en" for q in golden) / len(golden)
        assert 0.15 <= share <= 0.3

    def test_it_includes_out_of_corpus_trick_and_cross_agency_questions(self, golden) -> None:
        assert sum(q.kind == "out_of_corpus" for q in golden) >= 5
        assert sum("trick" in q.tags for q in golden) >= 5
        assert any(len(q.expected) > 1 for q in golden)

    def test_every_target_names_a_corpus_document(self, golden) -> None:
        titles = {entry.title for entry in load_manifest()}
        named = {loc.document for q in golden for t in q.expected for loc in t.locations}
        assert named - titles == set()

    def test_it_is_labelled_as_llm_drafted_and_author_reviewed(self) -> None:
        text = (ROOT / "evals" / "golden_set.yaml").read_text(encoding="utf-8")
        assert "LLM-drafted from the source documents, reviewed by the author" in text
