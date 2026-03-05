import typer
from pathlib import Path
from rich.console import Console
from config import Config

app = typer.Typer()
console = Console()


@app.command()
def index(files: list[Path]):
    """Index files into the vector store."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rag.indexer import Indexer, IndexProgress
    from rag.agent import make_vector_store
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    config = Config()

    with console.status("[bold blue]Loading embedding model..."):
        vector_store = make_vector_store(config)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.splitter_chunk_size,
        chunk_overlap=config.splitter_chunk_overlap,
        add_start_index=config.splitter_add_start_index,
    )

    indexer = Indexer(vector_store, splitter, batch_size=config.indexer_batch_size)
    progress_bar: Progress | None = None
    task_id = None

    def on_progress(p: IndexProgress) -> None:
        nonlocal progress_bar, task_id
        if p.total_chunks == 0 and p.files_found:
            console.print(f"Found [bold]{p.files_found}[/bold] file(s)")
        elif p.total_chunks > 0 and task_id is None:
            console.print(
                f"Loaded [bold]{p.docs_loaded}[/bold] doc(s), "
                f"split into [bold]{p.total_chunks}[/bold] chunk(s)"
            )
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            )
            progress_bar.start()
            task_id = progress_bar.add_task("Indexing", total=p.total_chunks)
        elif progress_bar and task_id is not None:
            progress_bar.update(task_id, completed=p.chunks_indexed)

    count = indexer.index(files, on_progress=on_progress)
    if progress_bar:
        progress_bar.stop()
    console.print(f"[bold green]Done![/] Indexed {count} chunks.")


@app.command()
def ask(question: str):
    """Ask a question to the RAG agent."""
    from rag.agent import make_vector_store, make_agent
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    from rich.live import Live
    from rich.panel import Panel
    from rich.markdown import Markdown

    config = Config()

    with console.status("[bold blue]Loading embedding model..."):
        vector_store = make_vector_store(config)

    agent = make_agent(config, vector_store)

    last_content = ""
    with Live(console=console, refresh_per_second=8) as live:
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            msg = step["messages"][-1]
            if isinstance(msg, HumanMessage):
                continue
            if isinstance(msg, ToolMessage):
                live.update(Panel("[dim]Searching knowledge base...[/dim]", title="[yellow]Tool", border_style="yellow"))
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_names = ", ".join(tc["name"] for tc in msg.tool_calls)
                    live.update(Panel(f"[dim]Calling: {tool_names}[/dim]", title="[yellow]Agent", border_style="yellow"))
                elif msg.content and msg.content != last_content:
                    last_content = msg.content
                    live.update(Panel(Markdown(msg.content), title="[green]Answer", border_style="green"))

    console.print()


@app.command()
def bot():
    """Start the Telegram bot."""
    import asyncio
    from telegram.bot import main as bot_main

    asyncio.run(bot_main())


if __name__ == "__main__":
    app()
