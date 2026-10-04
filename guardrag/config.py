from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://guardrag:guardrag@127.0.0.1:5432/guardrag"
    groq_api_key: str = ""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    data_dir: str = "data"
    answer_model: str = "llama-3.3-70b-versatile"
    retriever: Literal["vector", "keyword", "hybrid"] = "hybrid"
    retrieval_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
