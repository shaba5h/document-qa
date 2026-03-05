# document-qa

A Q&A system for document collections using RAG (Retrieval-Augmented Generation). Indexes your documents and answers questions based on retrieved context. Uses ChromaDB for vector storage, HuggingFace embeddings (local), and OpenRouter for LLMs. Supports both CLI and Telegram bot interfaces.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env` — at minimum you need `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.

## Usage

### CLI

Index files or directories:

```bash
uv run python src/main.py index path/to/file.pdf path/to/docs/
```

Ask a question:

```bash
uv run python src/main.py ask "What does the refund policy say?"
```

### Telegram bot

1. Get a bot token from [@BotFather](https://t.me/BotFather)
2. Add `TELEGRAM_BOT_TOKEN=...` to your `.env`
3. Start the bot:

```bash
uv run python src/main.py bot
```

The bot handles `/start`, `/help`, and answers any text message using the RAG pipeline.

## Supported formats

PDF, DOCX, PPTX, HTML, Markdown, XLSX, CSV, AsciiDoc, plain text.

## Configuration

All settings are in `.env`. See `.env.example` for the full list with defaults.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `OPENROUTER_MODEL` | — | Required |
| `TELEGRAM_BOT_TOKEN` | — | Required for `bot` command |
| `OPENROUTER_TEMPERATURE` | `0.0` | Higher = more creative answers |
| `HUGGINGFACE_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `HUGGINGFACE_DEVICE` | `cpu` | Set to `cuda` for GPU |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma` | Where the vector DB is stored |
| `CHROMA_COLLECTION_NAME` | `documents` | Collection name |
| `RETRIEVER_K` | `3` | Chunks passed to the LLM per query |
| `RETRIEVER_SCORE_THRESHOLD` | off | Drop chunks above this distance score |
| `SPLITTER_CHUNK_SIZE` | `1000` | Requires re-indexing to take effect |
| `SPLITTER_CHUNK_OVERLAP` | `200` | Requires re-indexing to take effect |
| `INDEXER_BATCH_SIZE` | `50` | ChromaDB write batch size |
| `AGENT_SYSTEM_PROMPT` | see `.env.example` | Tweak agent behavior without touching code |
