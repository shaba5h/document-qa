from __future__ import annotations

import asyncio
import logging
import sys
from typing import NoReturn

import typer
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from document_qa.application.ask import AskUseCase
from document_qa.bootstrap import build_ask_use_case
from document_qa.settings import Settings

TELEGRAM_MESSAGE_LIMIT = 4096
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Run the Telegram bot interface for document-qa.",
)


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


async def _answer_question(message: Message, ask_use_case: AskUseCase) -> None:
    question = message.text.strip() if message.text else ""
    if not question:
        await message.answer("Send a text question.")
        return

    try:
        response = await asyncio.to_thread(ask_use_case.execute, question)
    except Exception:
        logger.exception("Failed to answer Telegram question")
        await message.answer("Could not answer the question.")
        return

    answer = response.answer.strip() or "No answer."

    for chunk in _split_telegram_text(answer):
        await message.answer(chunk)


def create_router(ask_use_case: AskUseCase) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def handle_start(message: Message) -> None:
        await message.answer("Send a text question.")

    @router.message(Command("help"))
    async def handle_help(message: Message) -> None:
        await message.answer("Send a text question.")

    @router.message(F.text.startswith("/"))
    async def handle_command(message: Message) -> None:
        await message.answer("Only text questions are supported.")

    @router.message(F.text)
    async def handle_text_question(message: Message) -> None:
        await _answer_question(message, ask_use_case)

    @router.message()
    async def handle_unsupported_message(message: Message) -> None:
        await message.answer("Only text questions are supported.")

    return router


async def run_bot(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    token = settings.telegram.bot_token
    if token is None:
        raise RuntimeError("DQA_TELEGRAM__BOT_TOKEN is required.")

    ask_use_case = build_ask_use_case(settings)

    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(ask_use_case))

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
