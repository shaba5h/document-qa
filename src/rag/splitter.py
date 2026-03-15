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


class MarkdownAwareSplitter(TextSplitter):
    """TextSplitter with markdown-aware splitting for .md files.

    For .md files:
    1. Split by headers (MarkdownHeaderTextSplitter) → semantically scoped sections
    2. Inject header breadcrumb into page_content for richer embeddings
    3. Within each section, detect markdown tables and keep them as atomic chunks
       (never split mid-table, even if the table exceeds chunk_size)
    4. Split non-table text by character count (RecursiveCharacterTextSplitter)
    5. Track file-scoped start_index across all sections for unique chunk IDs

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
            # Build breadcrumb from non-empty header levels
            crumb_parts = [
                section.metadata[k]
                for k in ("h1", "h2", "h3")
                if section.metadata.get(k)
            ]
            enriched = (
                " | ".join(crumb_parts) + "\n\n" + section.page_content
                if crumb_parts
                else section.page_content
            )

            base_meta = {**doc.metadata, **section.metadata}
            within_offset = 0

            for segment_text, is_table in _segment(enriched):
                if not segment_text.strip():
                    within_offset += len(segment_text)
                    continue

                if is_table:
                    # Tables are atomic — one chunk regardless of size
                    chunks.append(Document(
                        page_content=segment_text,
                        metadata={**base_meta, "start_index": section_offset + within_offset},
                    ))
                else:
                    # Regular text — split by character count
                    for chunk in self._char_splitter.create_documents([segment_text]):
                        local_start = chunk.metadata.get("start_index", 0)
                        chunks.append(Document(
                            page_content=chunk.page_content,
                            metadata={**base_meta, "start_index": section_offset + within_offset + local_start},
                        ))

                within_offset += len(segment_text)

            section_offset += len(enriched)

        return chunks
