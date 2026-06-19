from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: Path = Path(__file__).resolve().parent.parent.parent.parent / "gai_model_v5_max.pkl"
    embedding_model: str = "intfloat/multilingual-e5-small"
    vector_db_path: Path = Path("data/vector_store.index")
    llm_api_key: str = ""
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o-mini"
    use_local_model: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    max_tokens: int = 256
    temperature: float = 0.6
    top_k: int = 30
    rag_top_k: int = 3
    chunk_size: int = 256
    chunk_overlap: int = 32


settings = Settings()
