from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://guardrag:guardrag@localhost:5432/guardrag"
    groq_api_key: str = ""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    data_dir: str = "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
