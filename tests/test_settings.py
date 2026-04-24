import os

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

    settings = Settings()

    assert settings.chatmodel.api_key is not None
    assert settings.chatmodel.api_key.get_secret_value() == "test-key"
    assert settings.chatmodel.model_name == "openai/gpt-4o-mini"
    assert settings.retrieval.k == 4
    assert settings.knowledge_base.path.as_posix() == ".data/test-lance"
