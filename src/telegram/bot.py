import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from langchain_core.runnables import Runnable

from config import Config
from rag.agent import make_vector_store, make_agent

logger = logging.getLogger(__name__)
router = Router()


def _ask(agent: Runnable, question: str) -> str:
    from langchain_core.messages import AIMessage

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            return msg.content
    return "I couldn't find an answer. Please try rephrasing your question."


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    await message.answer(
        "Hello! I'm a RAG-powered assistant.\n\n"
        "Send me any question and I'll search my knowledge base to answer it.\n\n"
        "Type /help for more info."
    )


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    await message.answer(
        "*Available commands:*\n\n"
        "/start — Welcome message\n"
        "/help — Show this help\n\n"
        "Just send me a text message with your question!",
        parse_mode=ParseMode.MARKDOWN,
    )


@router.message()
async def handle_question(message: Message, agent: Runnable) -> None:
    if not message.text:
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    try:
        answer = await asyncio.to_thread(_ask, agent, message.text)
    except Exception:
        logger.exception("Agent error")
        await message.answer(
            "Sorry, something went wrong while processing your question. Please try again later."
        )
        return

    try:
        await message.answer(answer, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await message.answer(answer)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    config = Config()
    if not config.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set. Add it to your .env file.")
        sys.exit(1)

    logger.info("Loading embedding model...")
    vector_store = make_vector_store(config)

    bot = Bot(token=config.telegram_bot_token.get_secret_value())
    dp = Dispatcher()
    dp.include_router(router)
    dp["agent"] = make_agent(config, vector_store)

    logger.info("Bot is starting...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
