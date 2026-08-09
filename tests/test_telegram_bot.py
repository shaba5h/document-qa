import asyncio
import os
from pathlib import Path
from typing import cast

from aiogram.types import Message
from typer.testing import CliRunner

from document_qa.application.ask import AskUseCase
from document_qa.domain.models import Document, QAResponse
from document_qa.entrypoints.telegram_bot import (
    TELEGRAM_MESSAGE_LIMIT,
    _answer_question,
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


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.answers: list[str] = []

    async def answer(self, text: str) -> None:
        self.answers.append(text)


class FakeAskUseCase:
    def __init__(self, response: QAResponse) -> None:
        self.response = response
        self.questions: list[str] = []

    def execute(self, question: str) -> QAResponse:
        self.questions.append(question)
        return self.response


def test_answer_question_sends_grounded_answer_with_cited_source() -> None:
    document = Document(
        id="refunds",
        text="Refunds are available for 30 days.",
        source_path=Path("/knowledge/policy.pdf"),
        source_filename="policy.pdf",
        source_hash="hash",
        section_path=["Refunds"],
    )
    use_case = FakeAskUseCase(
        QAResponse(
            answer="Refunds are available for 30 days [1].",
            retrieved_documents=[(document, 0.9)],
        )
    )
    message = FakeMessage("  What is the refund window?  ")

    asyncio.run(
        _answer_question(
            cast(Message, message),
            cast(AskUseCase, use_case),
        )
    )

    assert use_case.questions == ["What is the refund window?"]
    assert message.answers == [
        "Refunds are available for 30 days [1].\n\n"
        "Sources:\n- [1] policy.pdf - Refunds"
    ]


def test_answer_question_hides_pipeline_errors() -> None:
    class FailingAskUseCase:
        def execute(self, question: str) -> QAResponse:
            raise RuntimeError("secret provider details")

    message = FakeMessage("What is indexed?")

    asyncio.run(
        _answer_question(
            cast(Message, message),
            cast(AskUseCase, FailingAskUseCase()),
        )
    )

    assert message.answers == ["Could not answer the question."]
