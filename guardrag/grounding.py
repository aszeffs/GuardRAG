"""Enforce grounding in code (ADR 0002): turn a draft into the response a citizen sees.

The Confidence rules are written down in the README; see tests/test_grounding.py.
"""

from collections.abc import Callable, Sequence

from guardrag.answer import AskResponse, Citation, Confidence, DraftAnswer, RefusalKind
from guardrag.retrieval import RetrievedPassage

Grounder = Callable[[DraftAnswer, Sequence[RetrievedPassage]], AskResponse]

DONT_KNOW = "I don't know. The documents I answer from don't cover this."
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
        return _refusal("out_of_scope", OUT_OF_SCOPE_MESSAGE)
    by_id = {p.passage_id: p for p in retrieved}
    claimed = list(dict.fromkeys(draft.citations))
    cited = [by_id[i] for i in claimed if i in by_id]
    if not cited:
        return _refusal("out_of_corpus", _dont_know(retrieved))
    return AskResponse(
        answer=draft.answer,
        citations=[_citation(p) for p in cited],
        confidence=_confidence(cited, retrieved, all_valid=len(cited) == len(claimed)),
        refusal=None,
    )


def _confidence(
    cited: Sequence[RetrievedPassage], retrieved: Sequence[RetrievedPassage], *, all_valid: bool
) -> Confidence:
    """`high` when the best retrieved Passage is cited and no Citation had to be dropped.

    Rank, not score, because scores are comparable only within one Retriever. A dropped
    Citation means part of the answer may rest on something the model was never given.
    """
    return "high" if all_valid and retrieved[0] in cited else "low"


def _refusal(kind: RefusalKind, message: str) -> AskResponse:
    return AskResponse(answer=message, citations=[], confidence="none", refusal=kind)


def _dont_know(retrieved: Sequence[RetrievedPassage]) -> str:
    """An "I don't know" pointing to the Agency of the best retrieved Passage, if any."""
    if not retrieved:
        return f"{DONT_KNOW} Please ask the government agency that handles this service."
    return f"{DONT_KNOW} The {retrieved[0].agency} is the agency most likely to help."


def _citation(p: RetrievedPassage) -> Citation:
    return Citation(
        passage_id=p.passage_id,
        document_title=p.document_title,
        section=p.section,
        page=p.page,
        url=p.url,
        as_of=p.as_of,
    )
