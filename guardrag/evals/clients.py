"""Groq clients for the evals, which retry rate-limited calls instead of failing the run.

Groq's free tier allows 8,000 tokens a minute per model, which a run exceeds, so a 429 is routine.
Both SDKs back off exponentially and honour Groq's `retry-after`. A 429 from the daily request
limit still fails the run once the retries run out.
"""

from typing import Any

import httpx
from groq import Groq

GROQ_OPENAI_URL = "https://api.groq.com/openai/v1"
MAX_RETRIES = 10


def retrying_groq(api_key: str, http_client: httpx.Client | None = None) -> Groq:
    """For the answer model and Prompt Guard, in place of their fail-fast production clients."""
    return Groq(api_key=api_key, max_retries=MAX_RETRIES, http_client=http_client)


def judge_client(api_key: str, http_client: httpx.AsyncClient | None = None) -> Any:
    """An async OpenAI client on Groq's compatible endpoint, which is what Ragas drives.

    Ragas gives a native Groq client instructor's tool-calling mode, and it gives an OpenAI client
    JSON mode, which works with every Ragas output model.
    """
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=api_key,
        base_url=GROQ_OPENAI_URL,
        max_retries=MAX_RETRIES,
        http_client=http_client,
    )
