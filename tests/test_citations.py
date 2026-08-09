from pathlib import Path

import pytest

from document_qa.application.citations import format_answer_with_citations
from document_qa.domain.models import Document, QAResponse


def make_document(filename: str, section: list[str]) -> Document:
    return Document(
        id=filename,
        text=f"Evidence from {filename}",
        source_path=Path("/tmp") / filename,
        source_filename=filename,
        source_hash=f"hash-{filename}",
        section_path=section,
    )


def test_format_answer_maps_inline_citations_to_sources_in_first_use_order() -> None:
    first = make_document("policy.pdf", ["Refunds"])
    second = make_document("handbook.pdf", ["Benefits", "Leave"])
    response = QAResponse(
        answer="Leave is 20 days [2]. Refunds take 30 days [1][2].",
        retrieved_documents=[(first, 0.9), (second, 0.8)],
    )

    formatted = format_answer_with_citations(response)

    assert formatted == (
        "Leave is 20 days [2]. Refunds take 30 days [1][2].\n\n"
        "Sources:\n"
        "- [2] handbook.pdf - Benefits > Leave\n"
        "- [1] policy.pdf - Refunds"
    )


def test_format_answer_rejects_citation_without_retrieved_document() -> None:
    response = QAResponse(
        answer="The policy changed [2].",
        retrieved_documents=[(make_document("policy.pdf", []), 0.9)],
    )

    with pytest.raises(ValueError, match=r"unknown citation\(s\): \[2]"):
        format_answer_with_citations(response)


def test_format_answer_accepts_explicit_insufficient_evidence_response() -> None:
    response = QAResponse(
        answer=(
            "[NO_EVIDENCE] The indexed documents do not contain enough evidence."
        ),
        retrieved_documents=[],
    )

    assert format_answer_with_citations(response) == (
        "The indexed documents do not contain enough evidence."
    )


def test_format_answer_rejects_uncited_claim() -> None:
    response = QAResponse(
        answer="The refund period is 30 days.",
        retrieved_documents=[(make_document("policy.pdf", []), 0.9)],
    )

    with pytest.raises(ValueError, match="must contain a citation"):
        format_answer_with_citations(response)


def test_format_answer_validates_every_index_in_grouped_citation() -> None:
    response = QAResponse(
        answer="The policies agree [1, 3].",
        retrieved_documents=[
            (make_document("policy.pdf", []), 0.9),
            (make_document("handbook.pdf", []), 0.8),
        ],
    )

    with pytest.raises(ValueError, match=r"unknown citation\(s\): \[3]"):
        format_answer_with_citations(response)


@pytest.mark.parametrize(
    "answer",
    [
        "The refund period is 30 days [1].\nSources: [1] fake.pdf",
        "The refund period is 30 days [1]. Sources: [1] fake.pdf",
    ],
)
def test_format_answer_rejects_model_generated_source_list(answer: str) -> None:
    response = QAResponse(
        answer=answer,
        retrieved_documents=[(make_document("policy.pdf", []), 0.9)],
    )

    with pytest.raises(ValueError, match="model-generated source list"):
        format_answer_with_citations(response)


def test_format_answer_canonicalizes_safe_cjk_numeric_citation() -> None:
    response = QAResponse(
        answer="The refund period is 30 days \u30101\u3011.",
        retrieved_documents=[(make_document("policy.pdf", ["Refunds"]), 0.9)],
    )

    assert format_answer_with_citations(response) == (
        "The refund period is 30 days [1].\n\n"
        "Sources:\n- [1] policy.pdf - Refunds"
    )


@pytest.mark.parametrize(
    "answer",
    [
        "The refund period is 30 days [01].",
        "The refund period is 30 days [[1]].",
        "The refund period is 30 days [1]].",
        "The refund period is 30 days [1] and maybe [999.",
        "The refund period is [1](https://fake.invalid).",
    ],
)
def test_format_answer_rejects_malformed_or_linked_citations(answer: str) -> None:
    response = QAResponse(
        answer=answer,
        retrieved_documents=[(make_document("policy.pdf", []), 0.9)],
    )

    with pytest.raises(ValueError):
        format_answer_with_citations(response)


@pytest.mark.parametrize(
    "answer",
    [
        "No evidence. [NO_EVIDENCE]",
        "[NO_EVIDENCE] [NO_EVIDENCE] No evidence.",
        "[NO_EVIDENCE]",
        "[NO_EVIDENCE] No evidence [1].",
    ],
)
def test_format_answer_rejects_invalid_no_evidence_contract(answer: str) -> None:
    response = QAResponse(
        answer=answer,
        retrieved_documents=[(make_document("policy.pdf", []), 0.9)],
    )

    with pytest.raises(ValueError):
        format_answer_with_citations(response)
