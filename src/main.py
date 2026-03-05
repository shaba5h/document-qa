import typer
from pathlib import Path
from rich.console import Console
from config import Config

app = typer.Typer()
console = Console()


def _make_vector_store(config: Config):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    with console.status("[bold blue]Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name=config.huggingface_model,
            model_kwargs={"device": config.huggingface_device},
        )

    return Chroma(
        embedding_function=embeddings,
        collection_name=config.chroma_collection_name,
        persist_directory=str(config.chroma_persist_directory),
    )


@app.command()
def index(files: list[Path]):
    """Index files into the vector store."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rag.indexer import Indexer

    config = Config()
    vector_store = _make_vector_store(config)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.splitter_chunk_size,
        chunk_overlap=config.splitter_chunk_overlap,
        add_start_index=config.splitter_add_start_index,
    )
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rag.indexer import IndexProgress

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
    from langchain_openrouter import ChatOpenRouter
    from langchain.tools import tool
    from langchain.agents import create_agent

    config = Config()
    vector_store = _make_vector_store(config)

    llm = ChatOpenRouter(
        model=config.openrouter_model,
        temperature=config.openrouter_temperature,
        openrouter_api_key=config.openrouter_api_key.get_secret_value(),
    )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Find relevant information to answer the question."""
        retrieved_docs = vector_store.similarity_search_with_score(query, k=config.retriever_k)
        if config.retriever_score_threshold is not None:
            retrieved_docs = [
                (doc, score) for doc, score in retrieved_docs
                if score <= config.retriever_score_threshold
            ]
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}\nScore: {score}"
            for doc, score in retrieved_docs
        )
        return serialized, retrieved_docs

    agent = create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=config.agent_system_prompt,
    )

    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    from rich.live import Live
    from rich.panel import Panel
    from rich.markdown import Markdown

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


if __name__ == "__main__":
    app()
