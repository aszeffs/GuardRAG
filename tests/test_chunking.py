"""Acceptance tests for chunk_document (seam 3 in issue #1).

Most tests count tokens as whitespace-separated words, and every word in a section is tagged with
that section's index ("s1w7") so a test can tell which section a Passage's text came from.
"""

import re
from datetime import date

import pytest

from guardrag.chunking import chunk_document
from guardrag.domain import Passage, Section, SourceDocument

_TAGGED_WORD = re.compile(r"s(\d+)w(\d+)")


def count_words(text: str) -> int:
    return len(text.split())


def count_chars(text: str) -> int:
    return len(text)


def section_text(index: int, n_words: int) -> str:
    """n_words tagged words, with a full stop every eight words so sentences exist."""
    words = [f"s{index}w{i}" + ("." if i % 8 == 7 else "") for i in range(n_words)]
    return " ".join(words)


def make_doc(*sections: Section) -> SourceDocument:
    return SourceDocument(
        url="https://example.gov.ph/charter.pdf",
        title="Example Citizen's Charter",
        agency="BIR",
        kind="service",
        fetched_at=date(2026, 10, 1),
        sections=list(sections),
    )


def tagged_words(text: str) -> list[tuple[int, int]]:
    """(section index, word index) for each tagged word in a Passage."""
    return [(int(s), int(w)) for s, w in _TAGGED_WORD.findall(text)]


def _span(text: str, passages: list[Passage], i: int) -> tuple[int, int]:
    """Where Passage i sits in text, searching after the start of Passage i - 1."""
    after = _span(text, passages, i - 1)[0] + 1 if i else 0
    start = text.find(passages[i].text, after)
    assert start >= 0, f"Passage is not verbatim section text: {passages[i].text!r}"
    return start, start + len(passages[i].text)


def section_of(passage: Passage) -> int:
    """Index of the section a Passage's first word came from."""
    return tagged_words(passage.text)[0][0]


LONG_DOC = make_doc(
    Section("1. Issuance of TIN", section_text(0, 130), page=3),
    Section("2. Certificate of Registration", section_text(1, 9), page=4),
    Section(None, section_text(2, 75), page=4),
    Section("Section 4. Coverage", section_text(3, 41)),
)
MAX_TOKENS, OVERLAP_TOKENS = 20, 5


@pytest.fixture
def passages():
    return chunk_document(
        LONG_DOC, count_words, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS
    )


def test_every_passage_fits_max_tokens(passages) -> None:
    assert passages
    assert all(count_words(p.text) <= MAX_TOKENS for p in passages)


def test_ordinals_are_contiguous_from_zero(passages) -> None:
    assert [p.ordinal for p in passages] == list(range(len(passages)))


def test_no_passage_crosses_a_section(passages) -> None:
    for p in passages:
        assert len({s for s, _ in tagged_words(p.text)}) == 1, p.text


def test_heading_and_page_carried(passages) -> None:
    for p in passages:
        source = LONG_DOC.sections[section_of(p)]
        assert p.section == source.heading
        assert p.page == source.page


def test_no_text_is_dropped(passages) -> None:
    for index, sec in enumerate(LONG_DOC.sections):
        covered = {w for p in passages for s, w in tagged_words(p.text) if s == index}
        assert covered == set(range(count_words(sec.text)))


def test_short_section_becomes_one_passage(passages) -> None:
    short = [p for p in passages if p.section == "2. Certificate of Registration"]
    assert len(short) == 1
    assert [w for _, w in tagged_words(short[0].text)] == list(range(9))


def test_consecutive_passages_in_a_section_overlap(passages) -> None:
    pairs = 0
    for prev, nxt in zip(passages, passages[1:], strict=False):
        if section_of(prev) != section_of(nxt):
            continue
        pairs += 1
        shared = set(tagged_words(prev.text)) & set(tagged_words(nxt.text))
        assert shared, f"no overlap between ordinals {prev.ordinal} and {nxt.ordinal}"
    assert pairs > 0


@pytest.mark.parametrize("overlap", [MAX_TOKENS, MAX_TOKENS + 1])
def test_overlap_not_below_max_tokens_raises(overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_document(LONG_DOC, count_words, max_tokens=MAX_TOKENS, overlap_tokens=overlap)


def test_word_longer_than_max_tokens_is_split_without_loss() -> None:
    # A token-dense run with no whitespace (a long URL, a PDF table squashed into one string).
    path = "-".join(str(i) for i in range(30))  # no repeated run, so each Passage is findable
    text = f"See https://example.gov.ph/{path} for details."
    doc = make_doc(Section("Links", text))

    passages = chunk_document(doc, count_chars, max_tokens=20, overlap_tokens=5)

    assert all(count_chars(p.text) <= 20 for p in passages)
    covered: set[int] = set()
    for i in range(len(passages)):
        covered.update(range(*_span(text, passages, i)))
    assert {i for i, ch in enumerate(text) if not ch.isspace()} <= covered


def test_blank_sections_produce_no_passages() -> None:
    doc = make_doc(Section("Empty", "  \n "), Section("Full", section_text(0, 3)))

    passages = chunk_document(doc, count_words, max_tokens=20, overlap_tokens=5)

    assert [p.section for p in passages] == ["Full"]
    assert passages[0].ordinal == 0


def test_each_passage_reaches_further_than_the_last() -> None:
    # A long unbroken run right after the overlap tail cannot fit beside it.
    path = "-".join(str(i) for i in range(15))
    text = f"w1 w2 w3 w4 w5 w6 w7 w8 w9 {path}"
    doc = make_doc(Section("Links", text))

    passages = chunk_document(doc, count_chars, max_tokens=20, overlap_tokens=8)

    ends = [_span(text, passages, i)[1] for i in range(len(passages))]
    assert ends == sorted(set(ends)), [p.text for p in passages]
