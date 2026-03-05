from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from pathlib import Path

class Config(BaseSettings):
    chroma_persist_directory: Path = Path("./chroma")
    chroma_collection_name: str = "documents"

    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_device: str = "cpu"

    telegram_bot_token: SecretStr | None = None

    openrouter_api_key: SecretStr
    openrouter_model: str
    openrouter_temperature: float = 0.0

    splitter_add_start_index: bool = True
    splitter_chunk_size: int = 1000
    splitter_chunk_overlap: int = 200

    retriever_k: int = 3
    retriever_score_threshold: float | None = None

    indexer_batch_size: int = 50

    agent_system_prompt: str = (
        "You are an assistant with access to a knowledge base. "
        "Use the retrieve_context tool to search for information before answering. "
        "If the knowledge base does not contain the answer, say so honestly."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )