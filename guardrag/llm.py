"""The answer model: turns a question and its retrieved Passages into a draft answer."""

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Protocol

from guardrag.answer import DraftAnswer
from guardrag.prompt import build_messages, parse_draft
from guardrag.retrieval import RetrievedPassage

ANSWER_MODEL = "llama-3.3-70b-versatile"


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
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=build_messages(question, passages),
            response_format={"type": "json_object"},
            temperature=0,
        )
        return parse_draft(completion.choices[0].message.content)
