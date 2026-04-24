from __future__ import annotations

from document_qa.domain.models import Document


def build_compact_context(
    results: list[tuple[Document, float]],
) -> str:
    parts = ["<retrieved_context>"]
    for index, (doc, score) in enumerate(results, start=1):
        section = " > ".join(doc.section_path) if doc.section_path else "-"

        parts.append(
            "\n".join(
                [
                    f'<doc index="{index}" score="{score:.2f}">',
                    f"source: {doc.source_filename}",
                    f"section: {section}",
                    "excerpt:",
                    doc.text,
                    "</doc>",
                ]
            )
        )
    parts.append("</retrieved_context>")
    return "\n".join(parts)
