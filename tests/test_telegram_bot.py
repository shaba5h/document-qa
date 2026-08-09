import asyncio
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import cast

import pytest
from aiogram.types import Message
from typer.testing import CliRunner

from document_qa.application.ask import AskUseCase
from document_qa.domain.models import Document, QAResponse
from document_qa.entrypoints.telegram_bot import (
    TELEGRAM_MESSAGE_LIMIT,
    TelegramGuard,
    _answer_question,
    _split_telegram_text,
    app,
)
from document_qa.settings import TelegramSettings


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
    def __init__(self, text: str, *, user_id: int = 1) -> None:
        self.text = text
        self.answers: list[str] = []
        self.from_user = SimpleNamespace(id=user_id)
        self.chat = SimpleNamespace(id=user_id)

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
    guard = TelegramGuard(TelegramSettings())

    asyncio.run(
        _answer_question(
            cast(Message, message),
            cast(AskUseCase, use_case),
            guard,
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
    guard = TelegramGuard(TelegramSettings())

    asyncio.run(
        _answer_question(
            cast(Message, message),
            cast(AskUseCase, FailingAskUseCase()),
            guard,
        )
    )

    assert message.answers == ["Could not answer the question."]


def test_answer_question_rejects_user_outside_allowlist_without_rag_call() -> None:
    use_case = FakeAskUseCase(QAResponse(answer="unused", retrieved_documents=[]))
    message = FakeMessage("What is indexed?", user_id=2)
    guard = TelegramGuard(TelegramSettings(allowed_user_ids={1}))

    asyncio.run(
        _answer_question(
            cast(Message, message),
            cast(AskUseCase, use_case),
            guard,
        )
    )

    assert use_case.questions == []
    assert message.answers == ["This bot is private."]


def test_answer_question_rejects_oversized_question_without_rag_call() -> None:
    use_case = FakeAskUseCase(QAResponse(answer="unused", retrieved_documents=[]))
    message = FakeMessage("123456")
    guard = TelegramGuard(TelegramSettings(max_question_chars=5))

    asyncio.run(
        _answer_question(
            cast(Message, message),
            cast(AskUseCase, use_case),
            guard,
        )
    )

    assert use_case.questions == []
    assert message.answers == [
        "Question is too long. Maximum length is 5 characters."
    ]


def test_guard_rejects_parallel_request_from_same_user() -> None:
    guard = TelegramGuard(TelegramSettings())

    assert guard.begin(1) is None
    assert guard.begin(1) == "Your previous question is still being processed."

    guard.finish(1)
    assert guard.begin(1) is None


def test_guard_rate_limits_completed_requests(monkeypatch) -> None:
    monkeypatch.setattr(
        "document_qa.entrypoints.telegram_bot.monotonic",
        lambda: 100.0,
    )
    guard = TelegramGuard(
        TelegramSettings(
            rate_limit_requests=1,
            rate_limit_window_seconds=60,
            rate_limit_tracked_users=1,
        )
    )

    assert guard.begin(1) is None
    guard.finish(1)

    assert guard.begin(1) == "Too many requests. Try again in 60 seconds."


def test_guard_does_not_exceed_cache_when_all_tracked_users_are_active() -> None:
    guard = TelegramGuard(TelegramSettings(rate_limit_tracked_users=1))

    assert guard.begin(1) is None
    assert guard.begin(2) == "The bot is busy. Try again shortly."

    guard.finish(1)


def test_guard_does_not_evict_unexpired_rate_history_for_new_user() -> None:
    guard = TelegramGuard(
        TelegramSettings(
            rate_limit_requests=1,
            rate_limit_window_seconds=60,
            rate_limit_tracked_users=1,
        )
    )

    assert guard.begin(1) is None
    guard.finish(1)
    assert guard.begin(2) == "The bot is busy. Try again shortly."
    assert guard.begin(1) == "Too many requests. Try again in 60 seconds."


def test_guard_rejects_when_global_rag_capacity_is_full() -> None:
    guard = TelegramGuard(TelegramSettings(max_concurrent_requests=1))

    async def exercise_guard() -> None:
        assert await guard.acquire_slot() is True
        assert await guard.acquire_slot() is False
        guard.release_slot()
        assert await guard.acquire_slot() is True
        guard.release_slot()

    asyncio.run(exercise_guard())


def test_answer_question_rejects_busy_bot_without_rag_call() -> None:
    use_case = FakeAskUseCase(QAResponse(answer="unused", retrieved_documents=[]))
    message = FakeMessage("What is indexed?")
    guard = TelegramGuard(TelegramSettings(max_concurrent_requests=1))

    async def exercise_busy_bot() -> None:
        assert await guard.acquire_slot() is True
        try:
            await _answer_question(
                cast(Message, message),
                cast(AskUseCase, use_case),
                guard,
            )
        finally:
            guard.release_slot()

    asyncio.run(exercise_busy_bot())

    assert use_case.questions == []
    assert message.answers == ["The bot is busy. Try again shortly."]


def test_cancelled_handler_holds_user_and_global_slots_until_worker_finishes() -> None:
    started = Event()
    finished = Event()
    release_worker = Event()

    class BlockingAskUseCase:
        def execute(self, question: str) -> QAResponse:
            started.set()
            try:
                if not release_worker.wait(timeout=2):
                    raise TimeoutError("test worker was not released")
                return QAResponse(
                    answer="[NO_EVIDENCE] No evidence.",
                    retrieved_documents=[],
                )
            finally:
                finished.set()

    message = FakeMessage("What is indexed?")
    guard = TelegramGuard(TelegramSettings(max_concurrent_requests=1))

    async def exercise_cancellation() -> None:
        task = asyncio.create_task(
            _answer_question(
                cast(Message, message),
                cast(AskUseCase, BlockingAskUseCase()),
                guard,
            )
        )
        assert await asyncio.to_thread(started.wait, 1)

        task.cancel()
        await asyncio.sleep(0)

        assert guard.begin(1) == "Your previous question is still being processed."
        assert await guard.acquire_slot() is False

        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(finished.wait, 1)
        await asyncio.sleep(0)

        assert guard.begin(1) is None
        guard.finish(1)

    asyncio.run(exercise_cancellation())
