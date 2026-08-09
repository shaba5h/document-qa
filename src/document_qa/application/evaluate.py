from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_qa.application.ask import AskUseCase
from document_qa.application.citations import validate_answer
from document_qa.application.retrieve import RetrieveUseCase
from document_qa.domain.models import QAResponse

_INLINE_CITATION_PATTERN = re.compile(
    r"\[(?:0|[1-9]\d*)(?:\s*,\s*(?:0|[1-9]\d*))*]"
)
_DASH_TRANSLATION = str.maketrans(
    {chr(codepoint): "-" for codepoint in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212)}
)


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_filename: str = Field(min_length=1)
    section_path: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_non_blank(self) -> Self:
        if not self.source_filename.strip() or any(
            not section.strip() for section in self.section_path
        ):
            raise ValueError("Gold evidence values must not be blank.")
        return self


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool
    expected_evidence: list[GoldEvidence] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    fact_aliases: dict[str, list[str]] = Field(default_factory=dict)
    fact_patterns: dict[str, list[str]] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_gold(self) -> Self:
        if not self.id.strip() or not self.question.strip():
            raise ValueError("Case id and question must not be blank.")
        if self.answerable and (
            not self.expected_evidence or not self.expected_facts
        ):
            raise ValueError(
                "Answerable cases require expected_evidence and expected_facts."
            )
        if not self.answerable and (self.expected_evidence or self.expected_facts):
            raise ValueError(
                "Unanswerable cases must not define expected_evidence or expected_facts."
            )
        for field_name in ("expected_facts", "tags"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} must not contain blank values.")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates.")
        evidence_keys = [_evidence_key(item) for item in self.expected_evidence]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("expected_evidence must not contain duplicates.")
        if set(self.fact_aliases) - set(self.expected_facts):
            raise ValueError("fact_aliases keys must exist in expected_facts.")
        for aliases in self.fact_aliases.values():
            if not aliases or any(not alias.strip() for alias in aliases):
                raise ValueError("fact_aliases must contain non-blank alternatives.")
            if len(aliases) != len(set(aliases)):
                raise ValueError("fact_aliases must not contain duplicates.")
        if set(self.fact_patterns) - set(self.expected_facts):
            raise ValueError("fact_patterns keys must exist in expected_facts.")
        for patterns in self.fact_patterns.values():
            if not patterns or any(not pattern.strip() for pattern in patterns):
                raise ValueError("fact_patterns must contain non-blank patterns.")
            if len(patterns) != len(set(patterns)):
                raise ValueError("fact_patterns must not contain duplicates.")
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Invalid fact pattern: {pattern}") from exc
        return self


class EvaluationPipeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedder_model: str = Field(min_length=1)
    documents_prompt_name: str | None = None
    query_prompt_name: str | None = None
    chunk_max_tokens: int = Field(gt=0)
    retrieval_k: int = Field(gt=0)
    vector_weight: float = Field(ge=0.0, le=1.0)
    fts_weight: float = Field(ge=0.0, le=1.0)
    generation_temperature: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if abs(self.vector_weight + self.fts_weight - 1.0) > 1e-6:
            raise ValueError("Retrieval weights must add up to 1.0.")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1)
    name: str = Field(min_length=1)
    description: str = ""
    pipeline: EvaluationPipeline | None = None
    corpus: list[Path] = Field(default_factory=list)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> Self:
        if not self.name.strip():
            raise ValueError("Dataset name must not be blank.")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case ids must be unique.")
        if len(self.corpus) != len(set(self.corpus)):
            raise ValueError("Corpus paths must be unique.")
        corpus_basenames = [path.name for path in self.corpus]
        if len(corpus_basenames) != len(set(corpus_basenames)):
            raise ValueError("Corpus filenames must be unique.")
        if corpus_basenames:
            expected_sources = {
                evidence.source_filename
                for case in self.cases
                for evidence in case.expected_evidence
            }
            unknown_sources = expected_sources - set(corpus_basenames)
            if unknown_sources:
                raise ValueError(
                    "Gold evidence references a file outside the corpus: "
                    f"{sorted(unknown_sources)[0]}"
                )
        return self


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(gt=0)
    embedder_model: str
    documents_prompt_name: str | None
    query_prompt_name: str | None
    chunk_max_tokens: int
    vector_weight: float
    fts_weight: float
    fresh_index: bool
    chat_model: str | None = None
    temperature: float | None = None
    dataset_sha256: str
    corpus_sha256: dict[str, str]


class RetrievedHit(BaseModel):
    rank: int
    document_id: str
    source_filename: str
    section_path: list[str]
    score: float


class EvaluationCaseResult(BaseModel):
    case_id: str
    question: str
    answerable: bool
    expected_evidence: list[GoldEvidence]
    expected_facts: list[str]
    fact_aliases: dict[str, list[str]]
    fact_patterns: dict[str, list[str]]
    tags: list[str]
    retrieved_hits: list[RetrievedHit]
    first_relevant_rank: int | None = None
    retrieval_hit_at_1: bool | None = None
    retrieval_hit_at_k: bool | None = None
    evidence_recall_at_k: float | None = None
    answer: str | None = None
    matched_facts: list[str] = Field(default_factory=list)
    cited_hits: list[RetrievedHit] = Field(default_factory=list)
    invalid_citation_count: int = 0
    citation_error: str | None = None
    retrieval_error: str | None = None
    generation_error: str | None = None
    facts_correct: bool | None = None
    citations_correct: bool | None = None
    passed: bool | None = None


class RetrievalMetrics(BaseModel):
    answerable_cases: int
    pipeline_error_count: int
    hit_at_1_count: int
    hit_at_1: float | None
    hit_at_k_count: int
    hit_at_k: float | None
    mrr_at_k: float | None
    evidence_recall_at_k: float | None


class GenerationMetrics(BaseModel):
    evaluated_cases: int
    contract_valid_count: int
    contract_validity: float | None
    answerable_cases: int
    all_facts_correct_count: int
    all_facts_accuracy: float | None
    matched_facts: int
    expected_facts: int
    fact_recall: float | None
    correct_citations: int
    citations: int
    cited_expected_evidence: int
    expected_evidence: int
    citation_precision: float | None
    citation_recall: float | None
    unanswerable_cases: int
    correct_no_answer_count: int
    no_answer_accuracy: float | None
    answerability_correct_count: int
    answerability_accuracy: float | None
    pipeline_error_count: int
    passed_cases: int
    overall_accuracy: float | None


class EvaluationReport(BaseModel):
    dataset: str
    total_cases: int
    config: EvaluationConfig
    retrieval: RetrievalMetrics
    generation: GenerationMetrics | None
    cases: list[EvaluationCaseResult]


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_corpus_paths(
    dataset_path: Path,
    dataset: EvaluationDataset,
) -> list[Path]:
    if not dataset.corpus:
        raise ValueError("The dataset does not define a corpus for --fresh-index.")

    root = dataset_path.expanduser().resolve().parent
    paths = [path if path.is_absolute() else root / path for path in dataset.corpus]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Corpus file does not exist: {missing[0]}")
    return paths


def evaluate_dataset(
    dataset: EvaluationDataset,
    config: EvaluationConfig,
    *,
    retrieve_use_case: RetrieveUseCase | None = None,
    ask_use_case: AskUseCase | None = None,
) -> EvaluationReport:
    if retrieve_use_case is None and ask_use_case is None:
        raise ValueError("Provide a retrieval or answer use case.")

    case_results: list[EvaluationCaseResult] = []
    retrieval_error_count = 0
    hit_at_1_count = 0
    hit_at_k_count = 0
    reciprocal_rank_total = 0.0
    evidence_recall_total = 0.0
    answerable_cases = 0

    contract_valid_count = 0
    all_facts_correct_count = 0
    matched_fact_count = 0
    expected_fact_count = 0
    correct_citation_count = 0
    citation_count = 0
    cited_expected_evidence_count = 0
    expected_evidence_count = 0
    unanswerable_cases = 0
    correct_no_answer_count = 0
    answerability_correct_count = 0
    generation_error_count = 0
    passed_cases = 0

    for case in dataset.cases:
        response: QAResponse | None = None
        generation_error: str | None = None
        retrieval_error: str | None = None

        if ask_use_case is not None:
            try:
                response = ask_use_case.execute(case.question)
                retrieval_results = response.retrieved_documents
            except Exception as exc:
                generation_error = type(exc).__name__
                generation_error_count += 1
                retrieval_results = []
        else:
            retrieval_results = []

        if response is None and retrieve_use_case is not None:
            try:
                retrieval_results = list(
                    retrieve_use_case.execute(case.question, config.k)
                )
            except Exception as exc:
                retrieval_error = type(exc).__name__
                retrieval_error_count += 1
                retrieval_results = []

        retrieval_results = list(retrieval_results[: config.k])

        hits = [
            RetrievedHit(
                rank=rank,
                document_id=document.id,
                source_filename=document.source_filename,
                section_path=document.section_path,
                score=float(score),
            )
            for rank, (document, score) in enumerate(retrieval_results, start=1)
        ]
        expected_evidence = {
            _evidence_key(evidence) for evidence in case.expected_evidence
        }
        first_relevant_rank = next(
            (
                hit.rank
                for hit in hits
                if _hit_evidence_key(hit) in expected_evidence
            ),
            None,
        )

        retrieval_hit_at_1: bool | None = None
        retrieval_hit_at_k: bool | None = None
        evidence_recall_at_k: float | None = None
        if case.answerable:
            answerable_cases += 1
            retrieval_hit_at_1 = bool(
                hits and _hit_evidence_key(hits[0]) in expected_evidence
            )
            retrieval_hit_at_k = first_relevant_rank is not None
            retrieved_evidence = {_hit_evidence_key(hit) for hit in hits}
            evidence_recall_at_k = len(
                expected_evidence & retrieved_evidence
            ) / len(expected_evidence)
            hit_at_1_count += retrieval_hit_at_1
            hit_at_k_count += retrieval_hit_at_k
            reciprocal_rank_total += (
                1 / first_relevant_rank if first_relevant_rank is not None else 0
            )
            evidence_recall_total += evidence_recall_at_k

        answer = response.answer if response is not None else None
        matched_facts: list[str] = []
        cited_hits: list[RetrievedHit] = []
        invalid_citation_count = 0
        citation_error: str | None = None
        facts_correct: bool | None = None
        citations_correct: bool | None = None
        passed: bool | None = None

        if ask_use_case is not None:
            contract_valid = False
            no_evidence = False
            raw_no_evidence = False
            if response is not None:
                raw_answer = response.answer.strip()
                raw_no_evidence = raw_answer.startswith("[NO_EVIDENCE]")
                normalized_answer = _normalize_text(
                    _INLINE_CITATION_PATTERN.sub(
                        "",
                        raw_answer.removeprefix("[NO_EVIDENCE]").strip(),
                    )
                )
                if not raw_no_evidence:
                    matched_facts = [
                        fact
                        for fact in case.expected_facts
                        if _matches_fact(
                            normalized_answer,
                            fact,
                            case.fact_aliases.get(fact, []),
                            case.fact_patterns.get(fact, []),
                        )
                    ]

                try:
                    validated = validate_answer(response)
                    if not validated.text:
                        raise ValueError("Answer is empty.")
                    contract_valid = True
                    no_evidence = validated.no_evidence
                except ValueError as exc:
                    citation_error = str(exc)

                citation_text = validated.text if contract_valid else raw_answer
                citation_indices = _extract_citation_indices(citation_text)
                for index in citation_indices:
                    citation_count += 1
                    if 1 <= index <= len(response.retrieved_documents):
                        document, score = response.retrieved_documents[index - 1]
                        cited_hit = RetrievedHit(
                            rank=index,
                            document_id=document.id,
                            source_filename=document.source_filename,
                            section_path=document.section_path,
                            score=float(score),
                        )
                        cited_hits.append(cited_hit)
                        if (
                            case.answerable
                            and _hit_evidence_key(cited_hit) in expected_evidence
                        ):
                            correct_citation_count += 1
                    else:
                        invalid_citation_count += 1

            contract_valid_count += contract_valid
            expected_fact_count += len(case.expected_facts)
            matched_fact_count += len(matched_facts)

            if case.answerable:
                facts_correct = (
                    not no_evidence
                    and response is not None
                    and len(matched_facts) == len(case.expected_facts)
                )
                all_facts_correct_count += facts_correct

                cited_evidence = {
                    _hit_evidence_key(hit) for hit in cited_hits
                }
                cited_expected_evidence_count += len(
                    cited_evidence & expected_evidence
                )
                expected_evidence_count += len(expected_evidence)
                citations_correct = (
                    contract_valid
                    and not invalid_citation_count
                    and cited_evidence == expected_evidence
                    and bool(cited_evidence)
                )
                answerability_correct = response is not None and not raw_no_evidence
                passed = (
                    facts_correct
                    and citations_correct
                    and evidence_recall_at_k == 1.0
                    and generation_error is None
                    and retrieval_error is None
                )
            else:
                unanswerable_cases += 1
                correct_no_answer = contract_valid and no_evidence
                correct_no_answer_count += correct_no_answer
                answerability_correct = response is not None and raw_no_evidence
                passed = correct_no_answer

            answerability_correct_count += answerability_correct
            passed_cases += bool(passed)

        case_results.append(
            EvaluationCaseResult(
                case_id=case.id,
                question=case.question,
                answerable=case.answerable,
                expected_evidence=case.expected_evidence,
                expected_facts=case.expected_facts,
                fact_aliases=case.fact_aliases,
                fact_patterns=case.fact_patterns,
                tags=case.tags,
                retrieved_hits=hits,
                first_relevant_rank=first_relevant_rank,
                retrieval_hit_at_1=retrieval_hit_at_1,
                retrieval_hit_at_k=retrieval_hit_at_k,
                evidence_recall_at_k=evidence_recall_at_k,
                answer=answer,
                matched_facts=matched_facts,
                cited_hits=cited_hits,
                invalid_citation_count=invalid_citation_count,
                citation_error=citation_error,
                retrieval_error=retrieval_error,
                generation_error=generation_error,
                facts_correct=facts_correct,
                citations_correct=citations_correct,
                passed=passed,
            )
        )

    retrieval_metrics = RetrievalMetrics(
        answerable_cases=answerable_cases,
        pipeline_error_count=retrieval_error_count,
        hit_at_1_count=hit_at_1_count,
        hit_at_1=_rate(hit_at_1_count, answerable_cases),
        hit_at_k_count=hit_at_k_count,
        hit_at_k=_rate(hit_at_k_count, answerable_cases),
        mrr_at_k=_rate(reciprocal_rank_total, answerable_cases),
        evidence_recall_at_k=_rate(evidence_recall_total, answerable_cases),
    )

    generation_metrics = None
    if ask_use_case is not None:
        total_cases = len(dataset.cases)
        generation_metrics = GenerationMetrics(
            evaluated_cases=total_cases,
            contract_valid_count=contract_valid_count,
            contract_validity=_rate(contract_valid_count, total_cases),
            answerable_cases=answerable_cases,
            all_facts_correct_count=all_facts_correct_count,
            all_facts_accuracy=_rate(
                all_facts_correct_count,
                answerable_cases,
            ),
            matched_facts=matched_fact_count,
            expected_facts=expected_fact_count,
            fact_recall=_rate(matched_fact_count, expected_fact_count),
            correct_citations=correct_citation_count,
            citations=citation_count,
            cited_expected_evidence=cited_expected_evidence_count,
            expected_evidence=expected_evidence_count,
            citation_precision=_rate(
                correct_citation_count,
                citation_count,
            ),
            citation_recall=_rate(
                cited_expected_evidence_count,
                expected_evidence_count,
            ),
            unanswerable_cases=unanswerable_cases,
            correct_no_answer_count=correct_no_answer_count,
            no_answer_accuracy=_rate(
                correct_no_answer_count,
                unanswerable_cases,
            ),
            answerability_correct_count=answerability_correct_count,
            answerability_accuracy=_rate(
                answerability_correct_count,
                total_cases,
            ),
            pipeline_error_count=generation_error_count,
            passed_cases=passed_cases,
            overall_accuracy=_rate(passed_cases, total_cases),
        )

    return EvaluationReport(
        dataset=dataset.name,
        total_cases=len(dataset.cases),
        config=config,
        retrieval=retrieval_metrics,
        generation=generation_metrics,
        cases=case_results,
    )


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_DASH_TRANSLATION)
    return " ".join(normalized.casefold().split())


def _evidence_key(evidence: GoldEvidence) -> tuple[str, tuple[str, ...]]:
    return evidence.source_filename, tuple(evidence.section_path)


def _hit_evidence_key(hit: RetrievedHit) -> tuple[str, tuple[str, ...]]:
    return hit.source_filename, tuple(hit.section_path)


def _matches_fact(
    normalized_text: str,
    fact: str,
    aliases: list[str],
    patterns: list[str],
) -> bool:
    literal_match = any(
        _contains_fact(normalized_text, mention)
        for mention in [fact, *aliases]
    )
    return literal_match or any(
        re.search(pattern, normalized_text) is not None
        for pattern in patterns
    )


def _contains_fact(normalized_text: str, fact: str) -> bool:
    normalized_fact = _normalize_text(fact)
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_fact)}(?!\w)",
            normalized_text,
        )
        is not None
    )


def _extract_citation_indices(answer: str) -> list[int]:
    indices: list[int] = []
    for match in _INLINE_CITATION_PATTERN.findall(answer):
        indices.extend(int(value.strip()) for value in match[1:-1].split(","))
    return list(dict.fromkeys(indices))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(numerator: int | float, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None
