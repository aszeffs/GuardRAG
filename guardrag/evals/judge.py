"""The Ragas judge for the answer eval: needs the eval extra (pip install -e ".[eval]").

Ragas 0.4 metrics, run on the judge model through Groq's OpenAI-compatible endpoint:

- faithfulness: the share of the answer's claims the given Passages support.
- answer_relevancy: how closely questions generated back from the answer match the question
  (cosine similarity under the Corpus embedder), 0 for a noncommittal answer.
- context_precision: whether the Passages the judge finds useful for the answer are ranked high.
  This is the variant without a reference answer, because the Golden Set has none.
"""

import asyncio
import sys
from collections.abc import Sequence
from typing import Any

from openai import APIError
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)
from ragas.metrics.collections.faithfulness.util import NLIStatementOutput

from guardrag.config import JUDGE_MODEL
from guardrag.embedder import Embedder
from guardrag.evals.answers import Metric
from guardrag.evals.clients import judge_client

# Groq's free tier rejects any request to the judge model that asks for more than 1,000 output
# tokens (Ragas asks for 1,024), so ask for the most it allows.
MAX_OUTPUT_TOKENS = 1000
# Faithfulness asks for a verdict and a reason per claim, which for a long answer outgrows that
# limit in one reply, so each request to the judge carries only a few claims.
CLAIMS_PER_REQUEST = 4


class _CorpusEmbedding(BaseRagasEmbedding):
    """The Corpus embedder, which is multilingual, so a Filipino question compares fairly."""

    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        self.embedder = embedder

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embedder.embed([text])[0]

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text)


class BatchedFaithfulness(Faithfulness):
    """Ragas faithfulness, with the claims judged in batches of CLAIMS_PER_REQUEST."""

    async def _create_verdicts(self, statements: list[str], context: str) -> NLIStatementOutput:
        verdicts = []
        for start in range(0, len(statements), CLAIMS_PER_REQUEST):
            batch = statements[start : start + CLAIMS_PER_REQUEST]
            verdicts += (await super()._create_verdicts(batch, context)).statements
        return NLIStatementOutput(statements=verdicts)


class RagasJudge:
    def __init__(
        self,
        api_key: str,
        embedder: Embedder,
        metrics: Sequence[Metric],
        model: str = JUDGE_MODEL,
    ) -> None:
        llm = llm_factory(
            model,
            client=judge_client(api_key),
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        available = {
            "faithfulness": lambda: BatchedFaithfulness(llm=llm),
            "answer_relevancy": lambda: AnswerRelevancy(
                llm=llm, embeddings=_CorpusEmbedding(embedder)
            ),
            "context_precision": lambda: ContextPrecisionWithoutReference(llm=llm),
        }
        self.metrics = {name: available[name]() for name in metrics}
        # One loop for the whole run: the async client's connections are bound to it.
        self._loop = asyncio.new_event_loop()

    def score(self, question: str, answer: str, passages: Sequence[str]) -> dict[Metric, float]:
        return {
            name: self._loop.run_until_complete(self._score(name, question, answer, passages))
            for name in self.metrics
        }

    async def _score(
        self, name: Metric, question: str, answer: str, passages: Sequence[str]
    ) -> float:
        inputs = {"user_input": question, "response": answer}
        if name != "answer_relevancy":
            inputs["retrieved_contexts"] = list(passages)
        try:
            return float((await self.metrics[name].ascore(**inputs)).value)
        except APIError:
            # Groq refused even after retries (a daily limit, a bad key): the run can't go on.
            raise
        except Exception as e:  # a malformed judge reply costs one score, not the run
            print(f"{name}: no score for {question!r}: {e!r}", file=sys.stderr)
            return float("nan")
