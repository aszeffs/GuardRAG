"""Split a SourceDocument into Passages.

Each section is cut into windows of whole words that fit `max_tokens`, measured with the
embedder's own tokenizer (ADR 0004). A window is pulled back to the last sentence or line end in
its second half when there is one, and the next window starts `overlap_tokens` before the previous
one ended, so a requirement split at a boundary still appears whole in one Passage. Windows never
cross a section, so every Passage keeps a single heading and page for its Citation.
"""

import re
from collections.abc import Callable

from guardrag.domain import Passage, Section, SourceDocument
from guardrag.embedder import MAX_PASSAGE_TOKENS

CountTokens = Callable[[str], int]

_UNIT = re.compile(r"\S+\s*")
# A word that ends a sentence (closing quotes or brackets allowed), or one followed by a line break.
_BOUNDARY = re.compile(r"""[.!?]["')\]]*\s*$|\n""")


def chunk_document(
    doc: SourceDocument,
    count_tokens: CountTokens,
    *,
    max_tokens: int = MAX_PASSAGE_TOKENS,
    overlap_tokens: int = 20,
) -> list[Passage]:
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(f"need 0 <= overlap_tokens < max_tokens, got {overlap_tokens=}")
    passages: list[Passage] = []
    for section in doc.sections:
        for text in _chunk_section(section, count_tokens, max_tokens, overlap_tokens):
            passages.append(Passage(len(passages), text, section.heading, section.page))
    return passages


def _chunk_section(
    section: Section, count_tokens: CountTokens, max_tokens: int, overlap_tokens: int
) -> list[str]:
    units = _units(section.text, count_tokens, max_tokens)

    def count(start: int, end: int) -> int:
        return count_tokens(_join(units, start, end))

    texts: list[str] = []
    start = prev_end = 0
    while start < len(units):
        end = _longest_fit(start, len(units), lambda e, start=start: count(start, e) <= max_tokens)
        if end <= prev_end:
            # The next word is too long to sit beside the overlap, so start it without one.
            start = prev_end
            continue
        if end < len(units):
            # Give up at most half the window, and always add text the last Passage lacked.
            floor = max(start + max(1, (end - start) // 2), prev_end + 1)
            end = _back_off_to_boundary(units, floor, end)
        texts.append(_join(units, start, end))
        if end == len(units):
            break
        prev_end = end
        # The earliest start after the current one whose tail up to `end` fits the overlap.
        start = _earliest_fit(start + 1, end, lambda s, end=end: count(s, end) <= overlap_tokens)
    return texts


def _units(text: str, count_tokens: CountTokens, max_tokens: int) -> list[str]:
    """Words with their trailing whitespace; a word too long for one Passage is cut into pieces."""
    units: list[str] = []
    for word in _UNIT.findall(text):
        while count_tokens(word.strip()) > max_tokens:
            cut = _longest_fit(
                0, len(word), lambda k, w=word: count_tokens(w[:k].strip()) <= max_tokens
            )
            units.append(word[:cut])
            word = word[cut:]
        units.append(word)
    return units


def _back_off_to_boundary(units: list[str], floor: int, end: int) -> int:
    """The last sentence or line end in units[floor - 1 : end], else `end` unchanged."""
    for b in range(end, floor - 1, -1):
        if _BOUNDARY.search(units[b - 1]):
            return b
    return end


def _longest_fit(lo: int, hi: int, fits: Callable[[int], bool]) -> int:
    """Largest n in [lo + 1, hi] with fits(n), assuming fits holds up to some point then stops.

    lo + 1 is returned without being checked, so a single unit always makes progress.
    """
    lo += 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _earliest_fit(lo: int, hi: int, fits: Callable[[int], bool]) -> int:
    """Smallest n in [lo, hi] with fits(n), assuming fits(hi) and that fits stays true once true."""
    while lo < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _join(units: list[str], start: int, end: int) -> str:
    return "".join(units[start:end]).strip()
