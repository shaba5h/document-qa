from pathlib import Path

from document_qa.domain.models import Document
from document_qa.infrastructure.langchain.context import build_compact_context


def test_build_compact_context_includes_source_section_score_and_text() -> None:
    document = Document(
        id="doc-1",
        text="The refund window is 30 days.",
        source_path=Path("/tmp/policy.pdf"),
        source_filename="policy.pdf",
        source_hash="hash",
        section_path=["Policy", "Refunds"],
    )

    context = build_compact_context([(document, 0.876)])

    assert context.startswith("<retrieved_context>")
    assert '<doc index="1" score="0.88">' in context
    assert "source: policy.pdf" in context
    assert "section: Policy > Refunds" in context
    assert "The refund window is 30 days." in context
    assert context.endswith("</retrieved_context>")
