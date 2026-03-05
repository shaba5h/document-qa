# rag-bot-py

RAG chatbot that indexes your documents and answers questions based on them. Uses ChromaDB for vector storage, HuggingFace embeddings (local), and any LLM via OpenRouter.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env` — at minimum you need `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.

## Usage

Index files or directories:

```bash
uv run python src/main.py index path/to/file.pdf path/to/docs/
```

Ask a question:

```bash
uv run python src/main.py ask "What does the refund policy say?"
```

## Supported formats

PDF, DOCX, PPTX, HTML, Markdown, XLSX, CSV, AsciiDoc, plain text.

## Configuration

All settings are in `.env`. See `.env.example` for the full list with defaults.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required |
| `OPENROUTER_MODEL` | — | Required |
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
