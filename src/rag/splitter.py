from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_HTML_TABLE_RE = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


def _segment(text: str) -> list[tuple[str, bool]]:
    """Split text into (content, is_table) segments.

    Detects HTML <table>...</table> blocks as table segments.
    Everything else is a text segment.
    """
    segments: list[tuple[str, bool]] = []
    last_end = 0

    for match in _HTML_TABLE_RE.finditer(text):
        before = text[last_end:match.start()]
        if before:
            segments.append((before, False))
        segments.append((match.group(), True))
        last_end = match.end()

    tail = text[last_end:]
    if tail:
        segments.append((tail, False))

    return segments or [(text, False)]


def _parse_html_table(html: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse an HTML table into (headers, data_rows), respecting colspan and rowspan.

    Multi-row headers are collapsed into composite column names joined with ' > '
    (e.g. "Доза внесения, кг/га > Азот"). A row is considered a header row if it
    belongs to <thead> or contains at least one <th> cell.

    Uses a grid-fill approach: each cell is placed at the next unoccupied (row, col)
    position, and spans are pre-filled so downstream rows see the correct value.
    Returns None if the table cannot be parsed or has no data rows.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return None

    all_rows = table.find_all("tr")
    grid: dict[tuple[int, int], str] = {}
    is_header_row: dict[int, bool] = {}

    for r_idx, row in enumerate(all_rows):
        # A row is a header row if it's in <thead> or contains any <th>
        in_thead = row.parent and row.parent.name == "thead"
        has_th = bool(row.find("th"))
        is_header_row[r_idx] = bool(in_thead or has_th)

        c_idx = 0
        for cell in row.find_all(["td", "th"]):
            while (r_idx, c_idx) in grid:
                c_idx += 1

            value = cell.get_text(separator=" ", strip=True)
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))

            for rs in range(rowspan):
                for cs in range(colspan):
                    grid[(r_idx + rs, c_idx + cs)] = value

            c_idx += colspan

    if not grid:
        return None

    max_row = max(r for r, _ in grid)
    max_col = max(c for _, c in grid)

    header_row_indices = [r for r in range(max_row + 1) if is_header_row.get(r)]
    data_row_indices = [r for r in range(max_row + 1) if not is_header_row.get(r)]

    if not header_row_indices or not data_row_indices:
        return None

    # Build composite column headers from all header rows per column
    # De-duplicate adjacent repeated values (from rowspan fill) within a column
    headers: list[str] = []
    for c in range(max_col + 1):
        parts: list[str] = []
        prev = None
        for r in header_row_indices:
            v = grid.get((r, c), "")
            if v and v != prev:
                parts.append(v)
            prev = v
        headers.append(" > ".join(parts) if parts else "")

    data_rows = [
        [grid.get((r, c), "") for c in range(max_col + 1)]
        for r in data_row_indices
    ]

    return headers, data_rows


def _table_row_chunks(
    table_html: str,
    base_meta: dict,
    breadcrumb: str,
    start_index: int,
) -> list[Document]:
    """Convert an HTML table into one Document per data row.

    Each chunk's page_content is a natural-language sentence:
        "Header1: Value1. Header2: Value2. ..."
    prefixed with the section breadcrumb for embedding context.

    The full original HTML table is stored in metadata["table"] so the LLM
    receives complete context when the chunk is retrieved.
    """
    parsed = _parse_html_table(table_html)
    if parsed is None:
        # Unparseable — fall back to single chunk with raw HTML
        return [Document(
            page_content=breadcrumb + "\n\n" + table_html if breadcrumb else table_html,
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
                "table": table_html,
            },
        ))

    return chunks


class MarkdownAwareSplitter(TextSplitter):
    """TextSplitter with markdown-aware splitting for .md files.

    For .md files:
    1. Split by headers → semantically scoped sections with breadcrumb context
    2. Detect HTML <table> blocks within each section
    3. Each table is split per row into natural-language sentences:
       - colspan and rowspan are resolved via grid-fill
       - page_content: "Header1: Value1. Header2: Value2." (for embedding)
       - metadata["table"]: full original HTML table (passed to LLM as context)
    4. Non-table text is split by character count (RecursiveCharacterTextSplitter)
    5. File-scoped start_index ensures unique chunk IDs

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
                        table_html=segment_text,
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
