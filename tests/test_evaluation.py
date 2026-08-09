import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import document_qa.entrypoints.cli as cli_entrypoint
from document_qa.application.ask import AskUseCase
from document_qa.application.evaluate import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationPipeline,
    GoldEvidence,
    evaluate_dataset,
    file_sha256,
    load_evaluation_dataset,
    resolve_corpus_paths,
)
from document_qa.application.retrieve import RetrieveUseCase
from document_qa.domain.models import Document, QARequest, QAResponse
from document_qa.entrypoints.cli import app


def make_document(document_id: str, filename: str) -> Document:
    return Document(
        id=document_id,
        text=f"Evidence from {filename}",
        source_path=Path("/knowledge") / filename,
        source_filename=filename,
        source_hash=f"hash-{document_id}",
        section_path=["Policy"],
    )


def make_config(*, k: int = 3, with_answers: bool = False) -> EvaluationConfig:
    return EvaluationConfig(
        k=k,
        embedder_model="test-embedder",
        documents_prompt_name=None,
        query_prompt_name=None,
        chunk_max_tokens=100,
        vector_weight=0.5,
        fts_weight=0.5,
        fresh_index=True,
        chat_model="test-chat" if with_answers else None,
        temperature=0.0 if with_answers else None,
        dataset_sha256="test-dataset-hash",
        corpus_sha256={},
    )


def gold_evidence(filename: str, section: str = "Policy") -> GoldEvidence:
    return GoldEvidence(
        source_filename=filename,
        section_path=[section],
    )


class MappingKnowledgeBase:
    def __init__(
        self,
        results: dict[str, list[tuple[Document, float]]],
    ) -> None:
        self.results = results

    def add_documents(self, documents: Iterable[Document]) -> None:
        raise AssertionError("Evaluation must not mutate the knowledge base")

    def search(self, query: str, k: int) -> Iterator[tuple[Document, float]]:
        yield from self.results.get(query, [])[:k]


class MappingAgent:
    def __init__(self, responses: dict[str, QAResponse]) -> None:
        self.responses = responses

    def run(self, request: QARequest) -> QAResponse:
        return self.responses[request.question]


def test_retrieval_evaluation_calculates_hit_mrr_and_multi_source_recall() -> None:
    relevant_a = make_document("a", "a.pdf")
    relevant_b = make_document("b", "b.pdf")
    relevant_c = make_document("c", "c.pdf")
    distractor = make_document("x", "noise.pdf")
    dataset = EvaluationDataset(
        name="retrieval-metrics",
        cases=[
            EvaluationCase(
                id="rank-two",
                question="question one",
                answerable=True,
                expected_evidence=[gold_evidence("a.pdf")],
                expected_facts=["fact a"],
            ),
            EvaluationCase(
                id="multi-source",
                question="question two",
                answerable=True,
                expected_evidence=[
                    gold_evidence("b.pdf"),
                    gold_evidence("c.pdf"),
                ],
                expected_facts=["fact b", "fact c"],
            ),
            EvaluationCase(
                id="unanswerable",
                question="question three",
                answerable=False,
            ),
        ],
    )
    knowledge_base = MappingKnowledgeBase(
        {
            "question one": [(distractor, 0.9), (relevant_a, 0.8)],
            "question two": [
                (relevant_b, 0.9),
                (distractor, 0.8),
                (relevant_c, 0.7),
            ],
            "question three": [(distractor, 0.9)],
        }
    )

    report = evaluate_dataset(
        dataset,
        make_config(),
        retrieve_use_case=RetrieveUseCase(knowledge_base),
    )

    assert report.generation is None
    assert report.retrieval.answerable_cases == 2
    assert report.retrieval.hit_at_1_count == 1
    assert report.retrieval.hit_at_1 == 0.5
    assert report.retrieval.hit_at_k == 1.0
    assert report.retrieval.mrr_at_k == 0.75
    assert report.retrieval.evidence_recall_at_k == 1.0
    assert report.cases[0].first_relevant_rank == 2
    assert report.cases[1].evidence_recall_at_k == 1.0
    assert report.cases[2].retrieval_hit_at_k is None


def test_generation_evaluation_scores_facts_citations_abstention_and_overall() -> None:
    policy = make_document("policy", "policy.pdf")
    handbook = make_document("handbook", "handbook.pdf")
    distractor = make_document("noise", "noise.pdf")
    dataset = EvaluationDataset(
        name="generation-metrics",
        cases=[
            EvaluationCase(
                id="correct",
                question="refund question",
                answerable=True,
                expected_evidence=[gold_evidence("policy.pdf")],
                expected_facts=["30 days"],
            ),
            EvaluationCase(
                id="correct-abstention",
                question="ceo question",
                answerable=False,
            ),
            EvaluationCase(
                id="invalid-citation",
                question="vacation question",
                answerable=True,
                expected_evidence=[gold_evidence("handbook.pdf")],
                expected_facts=["20 days"],
            ),
        ],
    )
    agent = MappingAgent(
        {
            "refund question": QAResponse(
                answer="The refund window is 30 days [2].",
                retrieved_documents=[(distractor, 0.9), (policy, 0.8)],
            ),
            "ceo question": QAResponse(
                answer="[NO_EVIDENCE] The documents do not identify a CEO.",
                retrieved_documents=[(distractor, 0.9)],
            ),
            "vacation question": QAResponse(
                answer="Employees receive 20 days [2].",
                retrieved_documents=[(handbook, 0.9)],
            ),
        }
    )

    report = evaluate_dataset(
        dataset,
        make_config(with_answers=True),
        ask_use_case=AskUseCase(agent),
    )

    assert report.generation is not None
    metrics = report.generation
    assert metrics.contract_validity == pytest.approx(2 / 3)
    assert metrics.all_facts_accuracy == 1.0
    assert metrics.fact_recall == 1.0
    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 0.5
    assert metrics.no_answer_accuracy == 1.0
    assert metrics.answerability_accuracy == 1.0
    assert metrics.overall_accuracy == pytest.approx(2 / 3)
    assert report.cases[0].passed is True
    assert report.cases[1].passed is True
    assert report.cases[2].passed is False
    assert "unknown citation" in (report.cases[2].citation_error or "")


def test_generation_evaluation_records_pipeline_errors_without_aborting() -> None:
    class FailingAgent:
        def run(self, request: QARequest) -> QAResponse:
            raise RuntimeError("provider failure")

    dataset = EvaluationDataset(
        name="provider-error",
        cases=[
            EvaluationCase(
                id="failed",
                question="question",
                answerable=True,
                expected_evidence=[gold_evidence("policy.pdf")],
                expected_facts=["30 days"],
            )
        ],
    )

    report = evaluate_dataset(
        dataset,
        make_config(with_answers=True),
        ask_use_case=AskUseCase(FailingAgent()),
        retrieve_use_case=RetrieveUseCase(
            MappingKnowledgeBase({"question": [(make_document("p", "policy.pdf"), 0.9)]})
        ),
    )

    assert report.generation is not None
    assert report.generation.pipeline_error_count == 1
    assert report.generation.overall_accuracy == 0.0
    assert report.retrieval.hit_at_1 == 1.0
    assert report.cases[0].generation_error == "RuntimeError"
    assert report.cases[0].passed is False


def test_fact_matching_does_not_count_a_number_inside_a_larger_number() -> None:
    policy = make_document("policy", "policy.pdf")
    dataset = EvaluationDataset(
        name="fact-boundary",
        cases=[
            EvaluationCase(
                id="refund",
                question="question",
                answerable=True,
                expected_evidence=[gold_evidence("policy.pdf")],
                expected_facts=["30 days"],
            )
        ],
    )
    agent = MappingAgent(
        {
            "question": QAResponse(
                answer="The refund window is 130 days [1].",
                retrieved_documents=[(policy, 0.9)],
            )
        }
    )

    report = evaluate_dataset(
        dataset,
        make_config(with_answers=True),
        ask_use_case=AskUseCase(agent),
    )

    assert report.generation is not None
    assert report.generation.fact_recall == 0.0
    assert report.generation.overall_accuracy == 0.0


def test_fact_matching_supports_unicode_dashes_and_subject_bound_patterns() -> None:
    policy = make_document("policy", "policy.pdf")
    dataset = EvaluationDataset(
        name="fact-normalization",
        cases=[
            EvaluationCase(
                id="dash",
                question="dash question",
                answerable=True,
                expected_evidence=[gold_evidence("policy.pdf")],
                expected_facts=["3-5 business days"],
            ),
            EvaluationCase(
                id="subject-bound",
                question="comparison question",
                answerable=True,
                expected_evidence=[gold_evidence("policy.pdf")],
                expected_facts=["physical product window is 30 calendar days"],
                fact_patterns={
                    "physical product window is 30 calendar days": [
                        "physical product.{0,80}30 calendar days"
                    ]
                },
            ),
        ],
    )
    agent = MappingAgent(
        {
            "dash question": QAResponse(
                answer="Delivery takes 3\u20115 business days [1].",
                retrieved_documents=[(policy, 0.9)],
            ),
            "comparison question": QAResponse(
                answer="A physical product can be returned within 30 calendar days [1].",
                retrieved_documents=[(policy, 0.9)],
            ),
        }
    )

    report = evaluate_dataset(
        dataset,
        make_config(with_answers=True),
        ask_use_case=AskUseCase(agent),
    )

    assert report.generation is not None
    assert report.generation.fact_recall == 1.0
    assert report.generation.overall_accuracy == 1.0


def test_evaluation_dataset_rejects_answerable_case_without_gold() -> None:
    with pytest.raises(ValidationError, match="expected_evidence"):
        EvaluationDataset(
            name="invalid",
            cases=[
                EvaluationCase(
                    id="missing-gold",
                    question="question",
                    answerable=True,
                )
            ],
        )


def test_sample_benchmark_manifest_is_valid_and_corpus_exists() -> None:
    dataset_path = (
        Path(__file__).parents[1] / "benchmarks" / "sample" / "dataset.json"
    )

    dataset = load_evaluation_dataset(dataset_path)
    corpus_paths = resolve_corpus_paths(dataset_path, dataset)

    assert len(dataset.cases) == 30
    assert sum(case.answerable for case in dataset.cases) == 24
    assert dataset.pipeline is not None
    assert dataset.pipeline.embedder_model == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert len(corpus_paths) == 6
    assert all(path.is_file() for path in corpus_paths)


def test_published_baseline_matches_versioned_dataset_corpus_and_lockfile() -> None:
    root = Path(__file__).parents[1]
    dataset_path = root / "benchmarks" / "sample" / "dataset.json"
    baseline = json.loads(
        (root / "benchmarks" / "sample" / "baseline.json").read_text(
            encoding="utf-8"
        )
    )
    dataset = load_evaluation_dataset(dataset_path)

    assert baseline["dataset_sha256"] == file_sha256(dataset_path)
    assert baseline["uv_lock_sha256"] == file_sha256(root / "uv.lock")
    assert baseline["corpus_sha256"] == {
        path.name: file_sha256(path)
        for path in resolve_corpus_paths(dataset_path, dataset)
    }
    assert baseline["generation"]["passed_cases"] == 29
    assert baseline["generation"]["overall_accuracy"] == pytest.approx(29 / 30)


def test_evaluation_dataset_rejects_duplicate_corpus_filenames() -> None:
    with pytest.raises(ValidationError, match="Corpus filenames must be unique"):
        EvaluationDataset(
            name="duplicate-filenames",
            corpus=[Path("hr/policy.md"), Path("sales/policy.md")],
            cases=[
                EvaluationCase(
                    id="policy",
                    question="question",
                    answerable=True,
                    expected_evidence=[gold_evidence("policy.md")],
                    expected_facts=["fact"],
                )
            ],
        )


def test_retrieval_evaluation_reports_search_errors() -> None:
    class FailingKnowledgeBase(MappingKnowledgeBase):
        def search(self, query: str, k: int) -> Iterator[tuple[Document, float]]:
            raise RuntimeError("database unavailable")
            yield

    dataset = EvaluationDataset(
        name="retrieval-error",
        cases=[
            EvaluationCase(
                id="failed",
                question="question",
                answerable=True,
                expected_evidence=[gold_evidence("policy.pdf")],
                expected_facts=["fact"],
            )
        ],
    )

    report = evaluate_dataset(
        dataset,
        make_config(),
        retrieve_use_case=RetrieveUseCase(FailingKnowledgeBase({})),
    )

    assert report.retrieval.pipeline_error_count == 1
    assert report.retrieval.hit_at_k == 0.0
    assert report.cases[0].retrieval_error == "RuntimeError"


def test_non_applicable_retrieval_metrics_are_null() -> None:
    dataset = EvaluationDataset(
        name="unanswerable-only",
        cases=[
            EvaluationCase(
                id="unknown",
                question="question",
                answerable=False,
            )
        ],
    )

    report = evaluate_dataset(
        dataset,
        make_config(),
        retrieve_use_case=RetrieveUseCase(MappingKnowledgeBase({})),
    )

    assert report.retrieval.answerable_cases == 0
    assert report.retrieval.hit_at_1 is None
    assert report.retrieval.mrr_at_k is None


def clear_dqa_environment(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("DQA_"):
            monkeypatch.delenv(name, raising=False)


def test_evaluate_cli_prints_machine_readable_retrieval_report(
    monkeypatch,
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        EvaluationDataset(
            name="cli-evaluation",
            cases=[
                EvaluationCase(
                    id="refund",
                    question="refund question",
                    answerable=True,
                    expected_evidence=[gold_evidence("policy.pdf")],
                    expected_facts=["30 days"],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    policy = make_document("policy", "policy.pdf")
    knowledge_base = MappingKnowledgeBase(
        {"refund question": [(policy, 0.9)]}
    )

    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)
    monkeypatch.setattr(
        cli_entrypoint,
        "build_retrieve_use_case",
        lambda settings: RetrieveUseCase(knowledge_base),
    )

    result = CliRunner().invoke(
        app,
        ["evaluate", str(dataset_path), "--k", "2", "--json"],
    )

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["dataset"] == "cli-evaluation"
    assert report["config"]["k"] == 2
    assert report["retrieval"]["hit_at_1"] == 1.0
    assert report["generation"] is None


def test_evaluate_cli_fresh_index_ingests_manifest_corpus_in_temporary_db(
    monkeypatch,
    tmp_path,
) -> None:
    corpus_path = tmp_path / "policy.md"
    corpus_path.write_text("Refunds are available for 30 days.", encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        EvaluationDataset(
            name="fresh-index",
            pipeline=EvaluationPipeline(
                embedder_model="benchmark-embedder",
                documents_prompt_name="document",
                query_prompt_name="query",
                chunk_max_tokens=123,
                retrieval_k=2,
                vector_weight=0.6,
                fts_weight=0.4,
                generation_temperature=0.0,
            ),
            corpus=[Path("policy.md")],
            cases=[
                EvaluationCase(
                    id="refund",
                    question="refund question",
                    answerable=True,
                    expected_evidence=[gold_evidence("policy.md")],
                    expected_facts=["30 days"],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    policy = make_document("policy", "policy.md")
    knowledge_base = MappingKnowledgeBase(
        {"refund question": [(policy, 0.9)]}
    )
    ingested_paths: list[Path] = []
    evaluation_paths: list[Path] = []

    class FakeIngestUseCase:
        def execute(self, path: Path) -> None:
            ingested_paths.append(path)

    def build_ingest(settings):
        assert settings.embedder.model_name == "benchmark-embedder"
        assert settings.embedder.documents_encode_prompt_name == "document"
        assert settings.embedder.query_encode_prompt_name == "query"
        assert settings.chunking.max_tokens == 123
        assert settings.retrieval.k == 2
        evaluation_paths.append(settings.knowledge_base.path)
        return FakeIngestUseCase()

    def build_retrieve(settings):
        assert settings.knowledge_base.path == evaluation_paths[0]
        return RetrieveUseCase(knowledge_base)

    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)
    monkeypatch.setattr(cli_entrypoint, "build_ingest_use_case", build_ingest)
    monkeypatch.setattr(cli_entrypoint, "build_retrieve_use_case", build_retrieve)

    result = CliRunner().invoke(
        app,
        ["evaluate", str(dataset_path), "--fresh-index"],
    )

    assert result.exit_code == 0
    assert ingested_paths == [corpus_path]
    assert len(evaluation_paths) == 1
    assert not evaluation_paths[0].exists()
    assert "Hit@1: 1/1 (100.0%)" in result.output
    assert "RAG overall accuracy: not measured" in result.output


def test_evaluate_cli_rejects_output_directory_before_building_pipeline(
    monkeypatch,
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        EvaluationDataset(
            name="output-validation",
            cases=[
                EvaluationCase(
                    id="refund",
                    question="question",
                    answerable=True,
                    expected_evidence=[gold_evidence("policy.pdf")],
                    expected_facts=["30 days"],
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    clear_dqa_environment(monkeypatch)
    monkeypatch.setattr(
        cli_entrypoint,
        "build_retrieve_use_case",
        lambda settings: (_ for _ in ()).throw(
            AssertionError("pipeline must not be built")
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(dataset_path),
            "--output",
            str(tmp_path / "missing" / "report.json"),
        ],
    )

    assert result.exit_code == 1
    assert "Output directory does not exist" in result.output
