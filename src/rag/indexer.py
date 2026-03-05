from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import TextSplitter
from pathlib import Path
from rag.loader import UniversalDocumentLoader


@dataclass
class IndexProgress:
    files_found: int = 0
    docs_loaded: int = 0
    total_chunks: int = 0
    chunks_indexed: int = 0


ProgressCallback = Callable[[IndexProgress], None]


class Indexer:
    """
    Indexer class for indexing documents in a vector store.
    """
    def __init__(
        self,
        vector_store: VectorStore,
        splitter: TextSplitter,
        *,
        batch_size: int,
    ):
        self._vector_store = vector_store
        self._splitter = splitter
        self._batch_size = batch_size

    def index(
        self,
        paths: list[Path],
        *,
        recursive: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> int:
        progress = IndexProgress()
        notify = on_progress or (lambda _: None)

        loader = UniversalDocumentLoader(paths, recursive=recursive)
        files = loader._collect_files()
        progress.files_found = len(files)
        notify(progress)

        docs = list(loader.lazy_load())
        chunks = self._splitter.split_documents(docs)
        progress.docs_loaded = len(docs)
        progress.total_chunks = len(chunks)
        notify(progress)

        if not chunks:
            return 0

        ids = [self._chunk_id(chunk) for chunk in chunks]

        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            batch_ids = ids[i : i + self._batch_size]
            self._vector_store.add_documents(batch, ids=batch_ids)
            progress.chunks_indexed += len(batch)
            notify(progress)

        return len(chunks)

    @staticmethod
    def _chunk_id(chunk: Document) -> str:
        source = chunk.metadata.get("source", "")
        start = chunk.metadata.get("start_index", 0)
        content = f"{source}:{start}"
        return hashlib.sha256(content.encode()).hexdigest()
