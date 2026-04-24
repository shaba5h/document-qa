from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markdown import Markdown

from document_qa.bootstrap import (
    build_ask_use_case,
    build_ingest_use_case,
    build_retrieve_use_case,
)
from document_qa.settings import Settings

console = Console()
app = typer.Typer(
    no_args_is_help=True,
    help="Index local documents and answer questions with retrieval-augmented generation.",
)


def _exit_with_error(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _require_chat_model_settings(settings: Settings) -> None:
    chat_model = settings.chatmodel

    if not chat_model.model_name.strip():
        _exit_with_error("DQA_CHATMODEL__MODEL_NAME is required for `ask`.")

    if chat_model.api_key is None:
        _exit_with_error("DQA_CHATMODEL__API_KEY is required for `ask`.")


@app.command("ingest", help="Parse one document and add its chunks to local LanceDB.")
def ingest(
    path: Annotated[Path, typer.Argument(help="Document file to parse and index.")],
) -> None:
    if not path.exists():
        _exit_with_error(f"Path does not exist: {path}")
    if not path.is_file():
        _exit_with_error(f"Path must be a file: {path}")

    settings = Settings()

    ingest_uc = build_ingest_use_case(settings)

    ingest_uc.execute(path)

    console.print(f"Ingested [bold]{path}[/bold]")


@app.command("ask", help="Answer a question using the indexed local knowledge base.")
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the RAG agent.")],
) -> None:
    question = question.strip()
    if not question:
        _exit_with_error("Question must not be empty.")

    settings = Settings()
    _require_chat_model_settings(settings)

    ask_uc = build_ask_use_case(settings)

    response = ask_uc.execute(question)

    console.print(Markdown(response.answer))


@app.command("retrieve", help="Show retrieved chunks without calling the chat model.")
def retrieve(
    query: Annotated[str, typer.Argument(help="Search query.")],
    k: Annotated[
        int | None, typer.Option("--k", "-k", help="Override retrieval k.")
    ] = None,
) -> None:
    query = query.strip()
    if not query:
        _exit_with_error("Query must not be empty.")
    if k is not None and k <= 0:
        _exit_with_error("--k must be greater than 0.")

    settings = Settings()
    if k is not None:
        settings.retrieval.k = k

    retrieve_uc = build_retrieve_use_case(settings)

    result = list(retrieve_uc.execute(query, settings.retrieval.k))

    if not result:
        typer.echo("No documents found.")
        return

    for index, (document, score) in enumerate(result, start=1):
        typer.echo(f"#{index}")
        typer.echo(f"score: {score:.4f}")
        typer.echo(f"source_filename: {document.source_filename}")
        typer.echo(f"section: {' > '.join(document.section_path)}")
        typer.echo(document.text)
        typer.echo("")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
