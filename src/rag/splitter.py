from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class MarkdownAwareSplitter(TextSplitter):
    """TextSplitter that applies a two-stage markdown-aware pipeline for .md files
    and standard character splitting for .txt and other plain-text files.

    For .md files:
    1. Split by headers (MarkdownHeaderTextSplitter) to produce semantically scoped sections
    2. Inject header breadcrumb into each section's page_content for richer embeddings
    3. Split by character count (RecursiveCharacterTextSplitter) for final chunk sizing
       with file-scoped start_index to ensure unique chunk IDs

    For other files: standard RecursiveCharacterTextSplitter only.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, add_start_index: bool = True, **kwargs: Any) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index, **kwargs)
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
        # Stage 1: split by headers
        sections = self._header_splitter.split_text(doc.page_content)

        # Stage 2: inject breadcrumb and split by size with file-scoped start_index
        chunks: list[Document] = []
        section_offset = 0

        for section in sections:
            # Build breadcrumb from non-empty header metadata
            crumb_parts = [section.metadata[k] for k in ("h1", "h2", "h3") if section.metadata.get(k)]
            enriched_content = " | ".join(crumb_parts) + "\n\n" + section.page_content if crumb_parts else section.page_content

            # Split the enriched section by character count
            sub_chunks = self._char_splitter.create_documents([enriched_content])

            # Merge metadata: original doc metadata + header metadata, then offset start_index
            for chunk in sub_chunks:
                merged_meta = {**doc.metadata, **section.metadata}
                local_start = chunk.metadata.get("start_index", 0)
                merged_meta["start_index"] = section_offset + local_start
                chunks.append(Document(page_content=chunk.page_content, metadata=merged_meta))

            section_offset += len(enriched_content)

        return chunks
