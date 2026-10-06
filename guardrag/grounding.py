"""Enforce grounding in code (ADR 0002): turn a draft into the response a citizen sees.

The Confidence rules are written down in the README; see tests/test_grounding.py.
"""

from collections.abc import Callable, Sequence

from guardrag.answer import AskResponse, Citation, Confidence, DraftAnswer, RefusalKind
from guardrag.retrieval import RetrievedPassage

Grounder = Callable[[DraftAnswer, Sequence[RetrievedPassage]], AskResponse]

DONT_KNOW_PREFIX = "I don't know. The documents I answer from don't cover this."
OUT_OF_SCOPE_MESSAGE = "Sorry, I can only help with questions about Philippine government services."


def ground(draft: DraftAnswer, retrieved: Sequence[RetrievedPassage]) -> AskResponse:
    """Check `draft` against the Passages retrieved for this request.

    - Citations to ids not in `retrieved` are dropped.
    - `draft.out_of_scope` becomes an `out_of_scope` Refusal with a polite decline.
    - No surviving Citation becomes an `out_of_corpus` Refusal: "I don't know", pointing to the
      likely Agency.
    - Otherwise Confidence is `high` or `low`, derived from the evidence by the rules in the
      README, never from the model.
    """
    if draft.out_of_scope:
        return out_of_scope()
    by_id = {p.passage_id: p for p in retrieved}
    claimed = list(dict.fromkeys(draft.citations))
    cited = [by_id[pid] for pid in claimed if pid in by_id]
    if not cited or not draft.answer.strip():
        return _refusal("out_of_corpus", _dont_know(retrieved))
    return AskResponse(
        answer=draft.answer,
        citations=[Citation.model_validate(p, from_attributes=True) for p in cited],
        confidence=_confidence(claimed, cited, retrieved),
        refusal=None,
    )


def _confidence(
    claimed: Sequence[int], cited: Sequence[RetrievedPassage], retrieved: Sequence[RetrievedPassage]
) -> Confidence:
    """The README's rules: `high` needs the best retrieved Passage cited and nothing dropped."""
    nothing_dropped = len(cited) == len(claimed)
    return "high" if nothing_dropped and retrieved[0] in cited else "low"


def out_of_scope() -> AskResponse:
    """The polite decline for an Out-of-Scope Request."""
    return _refusal("out_of_scope", OUT_OF_SCOPE_MESSAGE)


def _refusal(kind: RefusalKind, message: str) -> AskResponse:
    return AskResponse(answer=message, citations=[], confidence="none", refusal=kind)


def _dont_know(retrieved: Sequence[RetrievedPassage]) -> str:
    """An "I don't know" pointing to the Agency of the best retrieved Passage, if any."""
    if not retrieved:
        return f"{DONT_KNOW_PREFIX} Please ask the government agency that handles this service."
    return f"{DONT_KNOW_PREFIX} The {retrieved[0].agency} is the agency most likely to help."
