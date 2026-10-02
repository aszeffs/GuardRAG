"""Split a SourceDocument into Passages.

Owned by the author: see the GitHub issue "Implement chunk_document" and tests/test_chunking.py.
"""

from collections.abc import Callable

from guardrag.domain import Passage, SourceDocument
from guardrag.embedder import MAX_PASSAGE_TOKENS


def chunk_document(
    doc: SourceDocument,
    count_tokens: Callable[[str], int],
    *,
    max_tokens: int = MAX_PASSAGE_TOKENS,
    overlap_tokens: int = 20,
) -> list[Passage]:
    raise NotImplementedError
