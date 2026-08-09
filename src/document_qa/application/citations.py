from __future__ import annotations

import re
from dataclasses import dataclass

from document_qa.domain.models import QAResponse

_BRACKET_PATTERN = re.compile(r"\[([^\[\]]*)]")
_CITATION_PATTERN = re.compile(
    r"(?:0|[1-9]\d*)(?:\s*,\s*(?:0|[1-9]\d*))*"
)
_CJK_CITATION_PATTERN = re.compile(
    r"\u3010((?:0|[1-9]\d*)(?:\s*,\s*(?:0|[1-9]\d*))*)\u3011"
)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[[^]]+]\s*\(")
_SOURCE_LIST_PATTERN = re.compile(r"(?i)\b(?:sources?|references?)(?:\*\*)?\s*:")
_NO_EVIDENCE = "[NO_EVIDENCE]"


@dataclass(frozen=True)
class ValidatedAnswer:
    text: str
    citation_indices: tuple[int, ...]
    no_evidence: bool


def validate_answer(response: QAResponse) -> ValidatedAnswer:
    answer = _CJK_CITATION_PATTERN.sub(r"[\1]", response.answer.strip())
    if not answer:
        return ValidatedAnswer(text="", citation_indices=(), no_evidence=False)
    if _SOURCE_LIST_PATTERN.search(answer):
        raise ValueError("Answer must not contain a model-generated source list.")
    if "\n" in answer:
        raise ValueError("Answer must be a single paragraph.")
    if _MARKDOWN_LINK_PATTERN.search(answer):
        raise ValueError("Answer must not contain Markdown links.")

    indices: list[int] = []
    bracket_values = _BRACKET_PATTERN.findall(answer)
    text_without_brackets = _BRACKET_PATTERN.sub("", answer)
    if "[" in text_without_brackets or "]" in text_without_brackets:
        raise ValueError("Answer contains malformed brackets.")

    no_evidence_count = bracket_values.count("NO_EVIDENCE")
    if no_evidence_count and (
        no_evidence_count != 1 or not answer.startswith(_NO_EVIDENCE)
    ):
        raise ValueError("[NO_EVIDENCE] must appear exactly once at the start.")

    for value in bracket_values:
        if value == "NO_EVIDENCE":
            continue
        if not _CITATION_PATTERN.fullmatch(value):
            raise ValueError(f"Answer contains malformed citation: [{value}]")
        indices.extend(int(index.strip()) for index in value.split(","))
    indices = list(dict.fromkeys(indices))

    if answer.startswith(_NO_EVIDENCE):
        if indices:
            raise ValueError("A no-evidence answer must not contain citations.")
        explanation = answer.removeprefix(_NO_EVIDENCE).strip()
        if not explanation:
            raise ValueError("A no-evidence answer must include an explanation.")
        return ValidatedAnswer(
            text=explanation,
            citation_indices=(),
            no_evidence=True,
        )
    if not indices:
        raise ValueError("Answer must contain a citation or [NO_EVIDENCE].")

    invalid_indices = [
        index
        for index in indices
        if index < 1 or index > len(response.retrieved_documents)
    ]
    if invalid_indices:
        invalid = ", ".join(f"[{index}]" for index in invalid_indices)
        raise ValueError(f"Answer contains unknown citation(s): {invalid}")

    return ValidatedAnswer(
        text=answer,
        citation_indices=tuple(indices),
        no_evidence=False,
    )


def format_answer_with_citations(response: QAResponse) -> str:
    validated = validate_answer(response)
    if not validated.text or validated.no_evidence:
        return validated.text

    sources = []
    for index in validated.citation_indices:
        document, _ = response.retrieved_documents[index - 1]
        section = " > ".join(document.section_path)
        location = f" - {section}" if section else ""
        sources.append(f"- [{index}] {document.source_filename}{location}")

    return f"{validated.text}\n\nSources:\n" + "\n".join(sources)
