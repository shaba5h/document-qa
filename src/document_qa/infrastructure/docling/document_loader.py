from __future__ import annotations

from pathlib import Path
from typing import Iterator

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.base import BaseChunker

from document_qa.domain.models import Document


class DoclingDocumentLoader:
    def __init__(
        self,
        converter: DocumentConverter,
        chunker: BaseChunker,
    ) -> None:
        self._converter = converter
        self._chunker = chunker

    def load(self, path: Path) -> Iterator[Document]:
        resolved_path = path.expanduser().resolve()

        conversion_result = self._converter.convert(resolved_path)

        chunks_iter = self._chunker.chunk(conversion_result.document)

        for index, chunk in enumerate(chunks_iter):
            meta = chunk.meta.model_dump()

            origin = meta.get("origin", {})
            section_path = meta.get("headings", []) or []
            source_hash = origin.get("binary_hash")
            text = self._chunker.contextualize(chunk)
            yield Document(
                id=f"{source_hash}:{index}",
                text=text,
                source_path=resolved_path,
                source_filename=resolved_path.name,
                source_hash=str(source_hash),
                section_path=section_path,
            )
