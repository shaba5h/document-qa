from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from document_qa.application.citations import format_answer_with_citations
from document_qa.application.evaluate import (
    EvaluationConfig,
    EvaluationDataset,
    EvaluationReport,
    evaluate_dataset,
    file_sha256,
    load_evaluation_dataset,
    resolve_corpus_paths,
)
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

    try:
        answer = format_answer_with_citations(response)
    except ValueError as exc:
        _exit_with_error(str(exc))

    console.print(answer, markup=False)


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


@app.command(
    "evaluate",
    help="Measure retrieval and optional end-to-end RAG quality on a gold dataset.",
)
def evaluate(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Versioned JSON dataset with questions and gold evidence."),
    ],
    k: Annotated[
        int | None,
        typer.Option("--k", "-k", help="Override retrieval depth."),
    ] = None,
    with_answers: Annotated[
        bool,
        typer.Option(
            "--with-answers",
            help="Call the chat model and score facts, citations, and abstentions.",
        ),
    ] = False,
    fresh_index: Annotated[
        bool,
        typer.Option(
            "--fresh-index",
            help="Ingest the dataset corpus into an isolated temporary LanceDB.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the complete machine-readable report."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the JSON report to this file."),
    ] = None,
) -> None:
    if not dataset_path.is_file():
        _exit_with_error(f"Evaluation dataset does not exist: {dataset_path}")
    if k is not None and k <= 0:
        _exit_with_error("--k must be greater than 0.")
    if output is not None and not output.parent.is_dir():
        _exit_with_error(f"Output directory does not exist: {output.parent}")

    try:
        dataset = load_evaluation_dataset(dataset_path)
    except (OSError, ValueError) as exc:
        _exit_with_error(f"Invalid evaluation dataset: {exc}")

    settings = Settings()
    if dataset.pipeline is not None:
        settings.embedder.model_name = dataset.pipeline.embedder_model
        settings.embedder.documents_encode_prompt_name = (
            dataset.pipeline.documents_prompt_name
        )
        settings.embedder.query_encode_prompt_name = dataset.pipeline.query_prompt_name
        settings.chunking.max_tokens = dataset.pipeline.chunk_max_tokens
        settings.retrieval.k = dataset.pipeline.retrieval_k
        settings.retrieval.vector_weight = dataset.pipeline.vector_weight
        settings.retrieval.fts_weight = dataset.pipeline.fts_weight
        settings.chatmodel.temperature = dataset.pipeline.generation_temperature
    if k is not None:
        settings.retrieval.k = k
    if with_answers:
        _require_chat_model_settings(settings)

    try:
        corpus_paths = (
            resolve_corpus_paths(dataset_path, dataset) if dataset.corpus else []
        )
    except ValueError as exc:
        _exit_with_error(str(exc))
    dataset_sha256 = file_sha256(dataset_path)
    corpus_sha256 = {
        path.name: file_sha256(path)
        for path in corpus_paths
    }

    if fresh_index:
        if not corpus_paths:
            _exit_with_error("The dataset does not define a corpus for --fresh-index.")

        try:
            with TemporaryDirectory(prefix="document-qa-evaluation-") as temp_dir:
                settings.knowledge_base.path = Path(temp_dir)
                ingest_use_case = build_ingest_use_case(settings)
                for corpus_path in corpus_paths:
                    ingest_use_case.execute(corpus_path)
                report = _run_evaluation(
                    dataset,
                    settings,
                    with_answers=with_answers,
                    fresh_index=True,
                    dataset_sha256=dataset_sha256,
                    corpus_sha256=corpus_sha256,
                )
        except Exception as exc:
            _exit_with_error(
                f"Evaluation pipeline failed ({type(exc).__name__}): {exc}"
            )
    else:
        try:
            report = _run_evaluation(
                dataset,
                settings,
                with_answers=with_answers,
                fresh_index=False,
                dataset_sha256=dataset_sha256,
                corpus_sha256=corpus_sha256,
            )
        except Exception as exc:
            _exit_with_error(
                f"Evaluation pipeline failed ({type(exc).__name__}): {exc}"
            )

    serialized_report = report.model_dump_json(indent=2)
    if output is not None:
        try:
            output.write_text(f"{serialized_report}\n", encoding="utf-8")
        except OSError as exc:
            _exit_with_error(f"Could not write evaluation report: {exc}")

    if json_output:
        typer.echo(serialized_report)
    else:
        _print_evaluation_report(report)
        if output is not None:
            typer.echo(f"Report: {output}")

    generation_errors = (
        report.generation.pipeline_error_count
        if report.generation is not None
        else 0
    )
    if report.retrieval.pipeline_error_count or generation_errors:
        raise typer.Exit(code=1)


def _run_evaluation(
    dataset: EvaluationDataset,
    settings: Settings,
    *,
    with_answers: bool,
    fresh_index: bool,
    dataset_sha256: str,
    corpus_sha256: dict[str, str],
) -> EvaluationReport:
    config = EvaluationConfig(
        k=settings.retrieval.k,
        embedder_model=settings.embedder.model_name,
        documents_prompt_name=settings.embedder.documents_encode_prompt_name,
        query_prompt_name=settings.embedder.query_encode_prompt_name,
        chunk_max_tokens=settings.chunking.max_tokens,
        vector_weight=settings.retrieval.vector_weight,
        fts_weight=settings.retrieval.fts_weight,
        fresh_index=fresh_index,
        chat_model=settings.chatmodel.model_name if with_answers else None,
        temperature=settings.chatmodel.temperature if with_answers else None,
        chat_timeout_seconds=(
            settings.chatmodel.timeout_seconds if with_answers else None
        ),
        chat_max_retries=settings.chatmodel.max_retries if with_answers else None,
        dataset_sha256=dataset_sha256,
        corpus_sha256=corpus_sha256,
    )
    if with_answers:
        return evaluate_dataset(
            dataset,
            config,
            ask_use_case=build_ask_use_case(settings),
            retrieve_use_case=build_retrieve_use_case(settings),
        )
    return evaluate_dataset(
        dataset,
        config,
        retrieve_use_case=build_retrieve_use_case(settings),
    )


def _print_evaluation_report(report: EvaluationReport) -> None:
    retrieval_metrics = report.retrieval
    typer.echo(f"Dataset: {report.dataset}")
    typer.echo(f"Cases: {report.total_cases}")
    typer.echo("")
    typer.echo("Retrieval")
    typer.echo(
        "  Hit@1: "
        f"{retrieval_metrics.hit_at_1_count}/"
        f"{retrieval_metrics.answerable_cases} "
        f"({_format_percentage(retrieval_metrics.hit_at_1)})"
    )
    typer.echo(
        f"  Hit@{report.config.k}: "
        f"{retrieval_metrics.hit_at_k_count}/"
        f"{retrieval_metrics.answerable_cases} "
        f"({_format_percentage(retrieval_metrics.hit_at_k)})"
    )
    typer.echo(
        f"  MRR@{report.config.k}: "
        f"{_format_percentage(retrieval_metrics.mrr_at_k)}"
    )
    typer.echo(
        f"  Evidence recall@{report.config.k}: "
        f"{_format_percentage(retrieval_metrics.evidence_recall_at_k)}"
    )
    typer.echo(f"  Pipeline errors: {retrieval_metrics.pipeline_error_count}")

    generation_metrics = report.generation
    typer.echo("")
    if generation_metrics is None:
        typer.echo("RAG overall accuracy: not measured (use --with-answers)")
        return

    typer.echo(
        "RAG overall accuracy: "
        f"{generation_metrics.passed_cases}/"
        f"{generation_metrics.evaluated_cases} "
        f"({_format_percentage(generation_metrics.overall_accuracy)})"
    )
    typer.echo(
        f"  Fact recall: {_format_percentage(generation_metrics.fact_recall)}"
    )
    typer.echo(
        "  All-facts accuracy: "
        f"{_format_percentage(generation_metrics.all_facts_accuracy)}"
    )
    typer.echo(
        "  Citation precision: "
        f"{_format_percentage(generation_metrics.citation_precision)}"
    )
    typer.echo(
        "  Citation recall: "
        f"{_format_percentage(generation_metrics.citation_recall)}"
    )
    typer.echo(
        "  No-answer accuracy: "
        f"{_format_percentage(generation_metrics.no_answer_accuracy)}"
    )
    typer.echo(
        "  Citation contract validity: "
        f"{_format_percentage(generation_metrics.contract_validity)}"
    )
    typer.echo(f"  Pipeline errors: {generation_metrics.pipeline_error_count}")


def _format_percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def main() -> None:
    app()


if __name__ == "__main__":
    main()
