"""The `/ask` contract: what a citizen gets back. Names follow CONTEXT.md."""

from datetime import date
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field

from guardrag.guards import MAX_QUESTION_CHARS, check_question

Confidence = Literal["high", "low", "none"]
RefusalKind = Literal["out_of_corpus", "out_of_scope"]


class AskRequest(BaseModel):
    """Oversized or malformed input fails validation, so it gets a 422 before any model call."""

    question: Annotated[
        str, Field(min_length=1, max_length=MAX_QUESTION_CHARS), AfterValidator(check_question)
    ]


class DraftAnswer(BaseModel):
    """What the model proposes, before grounding checks it (ADR 0002).

    This is also the JSON shape the model is asked to reply in.
    """

    answer: str
    citations: list[int] = Field(
        default_factory=list, description="passage_id of each Passage the answer relies on"
    )
    out_of_scope: bool = False


class Citation(BaseModel):
    """Points to the exact Passage that supports an answer, as shown to the citizen."""

    passage_id: int
    document_title: str
    section: str | None
    page: int | None
    url: str
    as_of: date


class AskResponse(BaseModel):
    """The reply to one question.

    Invariants (ADR 0002): a response with `refusal` set has `confidence: none`, and a response
    without one has at least one Citation and `confidence` of `high` or `low`.
    """

    answer: str
    citations: list[Citation]
    confidence: Confidence
    refusal: RefusalKind | None
