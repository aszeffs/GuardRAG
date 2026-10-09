"""Settings. Imports nothing from guardrag, so any module can read config without a cycle."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Must be a reasoning model: GroqLLM sends reasoning_effort, which other models reject.
ANSWER_MODEL = "openai/gpt-oss-120b"
# Ragas judge (#12): a different family from the answer model, so it isn't grading its own work.
# Qwen is the only other chat family Groq serves as of October 2026.
JUDGE_MODEL = "qwen/qwen3.8-27b"
PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
RetrieverMode = Literal["vector", "keyword", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://guardrag:guardrag@127.0.0.1:5432/guardrag"
    groq_api_key: str = ""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    data_dir: str = "data"
    answer_model: str = ANSWER_MODEL
    judge_model: str = JUDGE_MODEL
    retriever: RetrieverMode = "hybrid"
    retrieval_k: int = 5
    prompt_guard_model: str = PROMPT_GUARD_MODEL
    rate_limit_requests: int = 20
    rate_limit_window_seconds: float = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
