"""Enforce grounding in code (ADR 0002): turn a draft into the response a citizen sees.

Owned by the author: see issue #8 and tests/test_grounding.py.
"""

from collections.abc import Callable, Sequence

from guardrag.answer import AskResponse, DraftAnswer
from guardrag.retrieval import RetrievedPassage

Grounder = Callable[[DraftAnswer, Sequence[RetrievedPassage]], AskResponse]


def ground(draft: DraftAnswer, retrieved: Sequence[RetrievedPassage]) -> AskResponse:
    """Check `draft` against the Passages retrieved for this request.

    - Citations to ids not in `retrieved` are dropped.
    - `draft.out_of_scope` becomes an `out_of_scope` Refusal with a polite decline.
    - No surviving Citation becomes an `out_of_corpus` Refusal: "I don't know", pointing to the
      likely Agency.
    - Otherwise Confidence is `high` or `low`, derived from the evidence by the rules in the
      README, never from the model.
    """
    raise NotImplementedError
