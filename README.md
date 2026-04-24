# document-qa

Local CLI for question answering over document collections with retrieval-augmented generation.

The current runtime uses:

- Docling for document conversion and chunking
- Hugging Face sentence-transformer embeddings
- LanceDB for local hybrid retrieval
- OpenRouter for answer generation
- Typer/Rich for the command-line interface
- aiogram for the Telegram bot interface

## Requirements

- Python 3.11 through 3.13
- `uv`
- An OpenRouter API key for `document-qa ask`

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set at least:

```dotenv
DQA_CHATMODEL__API_KEY=your_openrouter_api_key_here
DQA_CHATMODEL__MODEL_NAME=openai/gpt-4o-mini
```

The CLI reads `.env` and `.env.local` automatically. Settings use the `DQA_` prefix and `__` for nested fields.

## Usage

Index a document:

```bash
uv run document-qa ingest path/to/document.pdf
```

Ask a question against the indexed collection:

```bash
uv run document-qa ask "What does the document say about refunds?"
```

Inspect retrieved chunks without calling the chat model:

```bash
uv run document-qa retrieve "refund policy" --k 5
```

Show command help:

```bash
uv run document-qa --help
uv run document-qa ingest --help
uv run document-qa ask --help
uv run document-qa retrieve --help
```

## Telegram Bot

The Telegram interface uses the same indexed LanceDB collection and answer pipeline as the CLI.

1. Create a bot token with [@BotFather](https://t.me/BotFather).
2. Set `DQA_TELEGRAM__BOT_TOKEN` in `.env`.
3. Start polling:

```bash
uv run document-qa-telegram
```

Show bot command help without starting polling:

```bash
uv run document-qa-telegram --help
```

## Configuration

See `.env.example` for all supported environment variables.

Important defaults:

| Variable | Default | Description |
| --- | --- | --- |
| `DQA_KNOWLEDGE_BASE__PATH` | `.data/lance` | Local LanceDB directory |
| `DQA_KNOWLEDGE_BASE__TABLE_NAME` | `documents` | LanceDB table name |
| `DQA_EMBEDDER__MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DQA_CHUNKING__MAX_TOKENS` | `500` | Docling chunk target size |
| `DQA_RETRIEVAL__K` | `10` | Chunks retrieved for answers |
| `DQA_TELEGRAM__BOT_TOKEN` | unset | Required for the Telegram bot |

Changing the embedding model or chunking settings requires re-ingesting documents.

## Supported Inputs

Docling determines the supported file types. Common formats include PDF, DOCX, PPTX, HTML, Markdown, XLSX, CSV, AsciiDoc, and plain text.

## Development Checks

```bash
uv run ty check
uv run pytest
uv build
```
