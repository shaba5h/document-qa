import os

from pydantic import SecretStr

from document_qa.infrastructure.langchain.factories import build_langchain_chat_model
from document_qa.settings import OpenRouterChatModelSettings
from document_qa.settings import Settings


def clear_dqa_environment(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("DQA_"):
            monkeypatch.delenv(name, raising=False)


def test_settings_reads_nested_dqa_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)
    monkeypatch.setenv("DQA_CHATMODEL__API_KEY", "test-key")
    monkeypatch.setenv("DQA_CHATMODEL__MODEL_NAME", "openai/gpt-4o-mini")
    monkeypatch.setenv("DQA_RETRIEVAL__K", "4")
    monkeypatch.setenv("DQA_KNOWLEDGE_BASE__PATH", ".data/test-lance")
    monkeypatch.setenv("DQA_CHATMODEL__TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("DQA_CHATMODEL__MAX_RETRIES", "1")
    monkeypatch.setenv("DQA_TELEGRAM__ALLOWED_USER_IDS", "[123, 456]")
    monkeypatch.setenv("DQA_TELEGRAM__MAX_QUESTION_CHARS", "900")
    monkeypatch.setenv("DQA_TELEGRAM__RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("DQA_TELEGRAM__MAX_CONCURRENT_REQUESTS", "2")

    settings = Settings()

    assert settings.chatmodel.api_key is not None
    assert settings.chatmodel.api_key.get_secret_value() == "test-key"
    assert settings.chatmodel.model_name == "openai/gpt-4o-mini"
    assert settings.retrieval.k == 4
    assert settings.knowledge_base.path.as_posix() == ".data/test-lance"
    assert settings.chatmodel.timeout_seconds == 45
    assert settings.chatmodel.max_retries == 1
    assert settings.telegram.allowed_user_ids == {123, 456}
    assert settings.telegram.max_question_chars == 900
    assert settings.telegram.rate_limit_requests == 3
    assert settings.telegram.max_concurrent_requests == 2


def test_chat_model_factory_applies_timeout_and_retry_guardrails() -> None:
    model = build_langchain_chat_model(
        OpenRouterChatModelSettings(
            model_name="test/model",
            api_key=SecretStr("test-key"),
            timeout_seconds=45,
            max_retries=1,
        )
    )

    assert getattr(model, "request_timeout") == 45_000
    assert getattr(model, "max_retries") == 1
    assert getattr(model, "client").sdk_configuration.timeout_ms == 45_000
    retry_config = getattr(model, "client").sdk_configuration.retry_config
    assert retry_config.backoff.max_elapsed_time == 150_000


def test_chat_model_factory_explicitly_disables_sdk_default_retries() -> None:
    model = build_langchain_chat_model(
        OpenRouterChatModelSettings(
            model_name="test/model",
            api_key=SecretStr("test-key"),
            max_retries=0,
        )
    )

    assert getattr(model, "client").sdk_configuration.retry_config is None
