from __future__ import annotations

import asyncio
import logging
import sys
from collections import deque
from math import ceil
from time import monotonic
from typing import NoReturn

import typer
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from document_qa.application.ask import AskUseCase
from document_qa.application.citations import format_answer_with_citations
from document_qa.bootstrap import build_ask_use_case
from document_qa.settings import Settings, TelegramSettings

TELEGRAM_MESSAGE_LIMIT = 4096
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Run the Telegram bot interface for document-qa.",
)


class TelegramGuard:
    def __init__(self, settings: TelegramSettings) -> None:
        self._settings = settings
        self._requests: dict[int, deque[float]] = {}
        self._active_users: set[int] = set()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    @property
    def max_question_chars(self) -> int:
        return self._settings.max_question_chars

    def begin(self, user_id: int) -> str | None:
        allowed_user_ids = self._settings.allowed_user_ids
        if allowed_user_ids and user_id not in allowed_user_ids:
            return "This bot is private."
        if user_id in self._active_users:
            return "Your previous question is still being processed."

        now = monotonic()
        if not self._prepare_rate_entry(user_id, now):
            return "The bot is busy. Try again shortly."
        requests = self._requests.setdefault(user_id, deque())
        window_start = now - self._settings.rate_limit_window_seconds
        while requests and requests[0] <= window_start:
            requests.popleft()

        if len(requests) >= self._settings.rate_limit_requests:
            retry_after = ceil(
                requests[0]
                + self._settings.rate_limit_window_seconds
                - now
            )
            return f"Too many requests. Try again in {max(1, retry_after)} seconds."

        requests.append(now)
        self._active_users.add(user_id)
        return None

    def finish(self, user_id: int) -> None:
        self._active_users.discard(user_id)

    async def acquire_slot(self) -> bool:
        if self._semaphore.locked():
            return False
        await self._semaphore.acquire()
        return True

    def release_slot(self) -> None:
        self._semaphore.release()

    def _prepare_rate_entry(self, user_id: int, now: float) -> bool:
        if user_id in self._requests:
            return True
        if len(self._requests) < self._settings.rate_limit_tracked_users:
            return True

        cutoff = now - self._settings.rate_limit_window_seconds
        for user_id, requests in list(self._requests.items()):
            if user_id not in self._active_users and (
                not requests or requests[-1] <= cutoff
            ):
                del self._requests[user_id]

        if len(self._requests) < self._settings.rate_limit_tracked_users:
            return True
        return False


def _exit_with_error(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _split_telegram_text(text: str) -> list[str]:
    if len(text) <= TELEGRAM_MESSAGE_LIMIT:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > TELEGRAM_MESSAGE_LIMIT:
        split_at = remaining.rfind("\n", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at <= 0:
            split_at = remaining.rfind(" ", 0, TELEGRAM_MESSAGE_LIMIT)
        if split_at <= 0:
            split_at = TELEGRAM_MESSAGE_LIMIT

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _requester_id(message: Message) -> int:
    if message.from_user is not None:
        return message.from_user.id
    return message.chat.id


async def _answer_question(
    message: Message,
    ask_use_case: AskUseCase,
    guard: TelegramGuard,
) -> None:
    question = message.text.strip() if message.text else ""
    if not question:
        await message.answer("Send a text question.")
        return

    user_id = _requester_id(message)
    rejection = guard.begin(user_id)
    if rejection is not None:
        await message.answer(rejection)
        return

    slot_acquired = False
    cleanup_deferred = False
    try:
        if len(question) > guard.max_question_chars:
            await message.answer(
                "Question is too long. Maximum length is "
                f"{guard.max_question_chars} characters."
            )
            return

        slot_acquired = await guard.acquire_slot()
        if not slot_acquired:
            await message.answer("The bot is busy. Try again shortly.")
            return

        try:
            worker = asyncio.create_task(
                asyncio.to_thread(ask_use_case.execute, question)
            )
            try:
                response = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cleanup_deferred = True
                worker.add_done_callback(
                    lambda task: _finish_cancelled_request(
                        task,
                        guard,
                        user_id,
                    )
                )
                raise
            answer = format_answer_with_citations(response) or "No answer."
        except Exception:
            logger.exception("Failed to answer Telegram question")
            await message.answer("Could not answer the question.")
            return

        for chunk in _split_telegram_text(answer):
            await message.answer(chunk)
    finally:
        if not cleanup_deferred:
            if slot_acquired:
                guard.release_slot()
            guard.finish(user_id)


def _finish_cancelled_request(
    task: asyncio.Task,
    guard: TelegramGuard,
    user_id: int,
) -> None:
    if not task.cancelled():
        task.exception()
    guard.release_slot()
    guard.finish(user_id)


def create_router(ask_use_case: AskUseCase, guard: TelegramGuard) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def handle_start(message: Message) -> None:
        await message.answer("Send a text question. Use /id to show your user ID.")

    @router.message(Command("help"))
    async def handle_help(message: Message) -> None:
        await message.answer("Send a text question. Use /id to show your user ID.")

    @router.message(Command("id"))
    async def handle_id(message: Message) -> None:
        await message.answer(f"Your user ID: {_requester_id(message)}")

    @router.message(F.text.startswith("/"))
    async def handle_command(message: Message) -> None:
        await message.answer("Only text questions are supported.")

    @router.message(F.text)
    async def handle_text_question(message: Message) -> None:
        await _answer_question(message, ask_use_case, guard)

    @router.message()
    async def handle_unsupported_message(message: Message) -> None:
        await message.answer("Only text questions are supported.")

    return router


async def run_bot(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    token = settings.telegram.bot_token
    if token is None:
        raise RuntimeError("DQA_TELEGRAM__BOT_TOKEN is required.")
    if not settings.chatmodel.model_name.strip():
        raise RuntimeError("DQA_CHATMODEL__MODEL_NAME is required.")
    if settings.chatmodel.api_key is None:
        raise RuntimeError("DQA_CHATMODEL__API_KEY is required.")

    ask_use_case = build_ask_use_case(settings)
    guard = TelegramGuard(settings.telegram)

    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(ask_use_case, guard))

    bot = Bot(
        token=token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=None),
    )

    await dispatcher.start_polling(bot)


@app.command(help="Start polling Telegram for document-qa questions.")
def run() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    try:
        asyncio.run(run_bot())
    except RuntimeError as exc:
        _exit_with_error(str(exc))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
