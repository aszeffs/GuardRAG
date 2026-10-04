"""The system prompt and structured output for the answer model.

Owned by the author: see issue #7 and tests/test_prompt.py.
"""

from collections.abc import Sequence

from guardrag.answer import DraftAnswer
from guardrag.retrieval import RetrievedPassage


def build_messages(question: str, passages: Sequence[RetrievedPassage]) -> list[dict[str, str]]:
    """Chat messages asking for a DraftAnswer as JSON.

    Each Passage goes in with its `passage_id`, fenced off as data rather than instructions, and
    the model is told to cite only those ids and to reply in the question's Answer Language.
    """
    raise NotImplementedError


def parse_draft(content: str | None) -> DraftAnswer:
    """Validate the model's reply. Anything malformed becomes a draft with no Citations."""
    raise NotImplementedError
