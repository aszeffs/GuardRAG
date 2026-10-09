"""The answer model: turns a question and its retrieved Passages into a draft answer."""

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Protocol

from guardrag.answer import DraftAnswer
from guardrag.config import ANSWER_MODEL
from guardrag.prompt import build_messages, parse_draft
from guardrag.retrieval import RetrievedPassage


class LLM(Protocol):
    def draft(self, question: str, passages: Sequence[RetrievedPassage]) -> DraftAnswer:
        """Draft an answer to `question` from `passages` only.

        `citations` should hold ids from `passages`, but callers must not trust that: grounding
        drops any other id. Malformed model output must come back as a draft with no Citations
        (which grounding turns into an Out-of-Corpus Refusal), never as an exception.
        """
        ...


class GroqLLM:
    """The answer model on Groq, through its OpenAI-compatible chat completions API."""

    def __init__(self, api_key: str, model: str = ANSWER_MODEL, client: Any = None) -> None:
        self.api_key = api_key
        self.model = model
        if client is not None:
            self._client = client

    @cached_property
    def _client(self) -> Any:
        # Built on first use, so the app starts (and /health answers) without GROQ_API_KEY.
        from groq import Groq

        return Groq(api_key=self.api_key)

    def draft(self, question: str, passages: Sequence[RetrievedPassage]) -> DraftAnswer:
        from groq import BadRequestError

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=build_messages(question, passages),
                response_format={"type": "json_object"},
                temperature=0,
                # gpt-oss reasons at medium effort by default, and at medium it often gave a
                # sample computation's figure as a rule (#36). Non-reasoning models reject this.
                reasoning_effort="high",
            )
        except BadRequestError as e:
            # In JSON mode Groq rejects a reply that isn't valid JSON instead of returning it.
            if _error_code(e.body) != "json_validate_failed":
                raise
            return parse_draft(None)
        return parse_draft(completion.choices[0].message.content)


def _error_code(body: object) -> str | None:
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            return error.get("code")
    return None
