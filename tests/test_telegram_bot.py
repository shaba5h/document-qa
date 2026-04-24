import os

from typer.testing import CliRunner

from document_qa.entrypoints.telegram_bot import (
    TELEGRAM_MESSAGE_LIMIT,
    _split_telegram_text,
    app,
)


runner = CliRunner()


def clear_dqa_environment(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("DQA_"):
            monkeypatch.delenv(name, raising=False)


def test_telegram_help_does_not_start_bot() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Start polling Telegram" in result.output


def test_telegram_without_token_fails_before_building_pipeline(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "DQA_TELEGRAM__BOT_TOKEN is required" in result.output


def test_split_telegram_text_keeps_chunks_within_message_limit() -> None:
    text = ("word " * TELEGRAM_MESSAGE_LIMIT).strip()

    chunks = _split_telegram_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= TELEGRAM_MESSAGE_LIMIT for chunk in chunks)
    assert " ".join(chunks) == text
