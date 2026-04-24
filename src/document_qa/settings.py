from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HuggingFaceEmbedderSettings(BaseModel):
    provider: Literal["huggingface"] = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 1
    documents_encode_prompt_name: str | None = None
    query_encode_prompt_name: str | None = None


EmbedderSettings = HuggingFaceEmbedderSettings


class LanceKnowledgeBaseSettings(BaseModel):
    provider: Literal["lance"] = "lance"
    table_name: str = "documents"
    path: Path = Path(".data/lance")


KnowledgeBaseSettings = LanceKnowledgeBaseSettings


class ChunkingSettings(BaseModel):
    max_tokens: int = 500


class RetrievalSettings(BaseModel):
    k: int = 10
    vector_weight: float = 0.5
    fts_weight: float = 0.5


class TelegramSettings(BaseModel):
    bot_token: SecretStr | None = None


class OpenRouterChatModelSettings(BaseModel):
    provider: Literal["openrouter"] = "openrouter"
    model_name: str = ""
    temperature: float = 0.2
    api_key: SecretStr | None = None


ChatModelSettings = OpenRouterChatModelSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DQA_",
        env_nested_delimiter="__",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        nested_model_default_partial_update=True,
        extra="ignore",
    )

    embedder: EmbedderSettings = Field(default_factory=HuggingFaceEmbedderSettings)
    knowledge_base: KnowledgeBaseSettings = Field(
        default_factory=LanceKnowledgeBaseSettings
    )
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chatmodel: ChatModelSettings = Field(default_factory=OpenRouterChatModelSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
