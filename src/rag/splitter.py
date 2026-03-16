from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _segment(text: str) -> list[tuple[str, bool]]:
    """Split text into (content, is_table) segments.

    A table segment is a run of consecutive lines starting with '|'.
    Everything else is a text segment.
    """
    if not text:
        return []

    segments: list[tuple[str, bool]] = []
    current_lines: list[str] = []
    current_is_table: bool | None = None

    for line in text.split("\n"):
        is_table = line.strip().startswith("|")
        if current_is_table is None:
            current_is_table = is_table
        if is_table != current_is_table:
            segments.append(("\n".join(current_lines), current_is_table))
            current_lines = []
            current_is_table = is_table
        current_lines.append(line)

    if current_lines:
        segments.append(("\n".join(current_lines), current_is_table or False))

    return segments


def _parse_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse a markdown table into (headers, data_rows).

    Returns None if the table has fewer than 3 lines or can't be parsed.
    """
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return None

    def parse_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = parse_row(lines[0])
    # lines[1] is the separator row (|---|---|), skip it
    data_rows = [parse_row(l) for l in lines[2:] if l.strip()]
    return headers, data_rows


def _table_row_chunks(
    table_text: str,
    base_meta: dict,
    breadcrumb: str,
    start_index: int,
) -> list[Document]:
    """Convert a markdown table into one Document per data row.

    Each chunk's page_content is a natural-language sentence:
        "Header1: Value1. Header2: Value2. ..."
    prefixed with the section breadcrumb for embedding context.

    The full original table is stored in metadata["table"] so the LLM
    receives complete context when the chunk is retrieved.
    """
    parsed = _parse_table(table_text)
    if parsed is None:
        # Unparseable table — fall back to single atomic chunk
        return [Document(
            page_content=table_text,
            metadata={**base_meta, "start_index": start_index},
        )]

    headers, data_rows = parsed
    chunks: list[Document] = []

    for i, row in enumerate(data_rows):
        pairs = [
            f"{h}: {v}"
            for h, v in zip(headers, row)
            if h.strip() and v.strip()
        ]
        row_text = ". ".join(pairs)
        if breadcrumb:
            row_text = breadcrumb + "\n\n" + row_text

        chunks.append(Document(
            page_content=row_text,
            metadata={
                **base_meta,
                "start_index": start_index + i,
                "table": table_text,
            },
        ))

    return chunks


class MarkdownAwareSplitter(TextSplitter):
    """TextSplitter with markdown-aware splitting for .md files.

    For .md files:
    1. Split by headers → semantically scoped sections with breadcrumb context
    2. Within each section, detect markdown tables and split them per row:
       - page_content: "Header1: Value1. Header2: Value2." (natural language for embedding)
       - metadata["table"]: full original table (passed to LLM as context)
    3. Split non-table text by character count (RecursiveCharacterTextSplitter)
    4. Track file-scoped start_index for unique chunk IDs

    For .txt files: standard RecursiveCharacterTextSplitter only.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        add_start_index: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
            **kwargs,
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS,
            strip_headers=False,
        )

    # Required by TextSplitter abstract base
    def split_text(self, text: str) -> list[str]:
        return self._char_splitter.split_text(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            source = doc.metadata.get("source", "")
            if Path(source).suffix.lower() == ".md":
                result.extend(self._split_markdown(doc))
            else:
                result.extend(self._char_splitter.split_documents([doc]))
        return result

    def _split_markdown(self, doc: Document) -> list[Document]:
        sections = self._header_splitter.split_text(doc.page_content)
        chunks: list[Document] = []
        section_offset = 0

        for section in sections:
            crumb_parts = [
                section.metadata[k]
                for k in ("h1", "h2", "h3")
                if section.metadata.get(k)
            ]
            breadcrumb = " | ".join(crumb_parts)
            enriched = (breadcrumb + "\n\n" + section.page_content) if breadcrumb else section.page_content

            base_meta = {**doc.metadata, **section.metadata}
            within_offset = 0

            for segment_text, is_table in _segment(enriched):
                if not segment_text.strip():
                    within_offset += len(segment_text)
                    continue

                if is_table:
                    chunks.extend(_table_row_chunks(
                        table_text=segment_text,
                        base_meta=base_meta,
                        breadcrumb=breadcrumb,
                        start_index=section_offset + within_offset,
                    ))
                else:
                    for chunk in self._char_splitter.create_documents([segment_text]):
                        local_start = chunk.metadata.get("start_index", 0)
                        chunks.append(Document(
                            page_content=chunk.page_content,
                            metadata={**base_meta, "start_index": section_offset + within_offset + local_start},
                        ))

                within_offset += len(segment_text)

            section_offset += len(enriched)

        return chunks
