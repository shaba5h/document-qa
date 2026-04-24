from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import lancedb
from lancedb.rerankers import MRRReranker

from document_qa.domain.models import Document
from document_qa.domain.ports import Embedder


class LanceKnowledgeBase:
    def __init__(
        self,
        path: Path,
        table_name: str,
        embedder: Embedder,
        *,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        batch_size: int = 256,
    ):
        self._db = lancedb.connect(str(path))
        self._table_name = table_name
        self._table = self._open_table(table_name)
        self._fts_index_ready = False
        self._embedder = embedder
        self._batch_size = batch_size
        self._reranker = MRRReranker(
            weight_vector=vector_weight,
            weight_fts=fts_weight,
        )

    def _open_table(self, table_name: str):
        table_names = set(self._db.table_names())
        if table_name in table_names:
            return self._db.open_table(table_name)
        return None

    def _ensure_table(self, rows: list[dict[str, Any]]):
        if self._table is None:
            self._table = self._db.create_table(
                self._table_name,
                data=rows,
                mode="create",
            )
            return self._table

        self._table.add(rows)
        return self._table

    def _ensure_fts_index(self) -> None:
        if self._table is None or self._fts_index_ready:
            return

        if self._has_text_fts_index():
            self._fts_index_ready = True
            return

        self._table.create_fts_index(
            "text",
            replace=True,
            stem=False,
            remove_stop_words=False,
            ascii_folding=False,
        )
        self._fts_index_ready = True

    def _has_text_fts_index(self) -> bool:
        if self._table is None:
            return False

        for index in self._table.list_indices():
            index_type = str(getattr(index, "index_type", "")).upper()
            columns = getattr(index, "columns", [])
            if index_type == "FTS" and "text" in columns:
                return True
        return False

    def add_documents(self, documents: Iterable[Document]) -> None:
        inserted_any = False
        batch: list[Document] = []

        for doc in documents:
            batch.append(doc)
            if len(batch) >= self._batch_size:
                self._ingest_batch(batch)
                inserted_any = True
                batch = []

        if batch:
            self._ingest_batch(batch)
            inserted_any = True

        if inserted_any:
            self._fts_index_ready = False
            self._ensure_fts_index()

    def _ingest_batch(self, batch: list[Document]) -> None:
        embeddings = self._embedder.embed_documents(batch)
        rows = [_document_to_row(doc, emb) for doc, emb in zip(batch, embeddings)]
        self._ensure_table(rows)

    def search(self, query: str, k: int) -> Iterator[tuple[Document, float]]:
        normalized_query = query.strip()
        if self._table is None or k <= 0 or not normalized_query:
            return

        self._ensure_fts_index()
        query_embedding = self._embedder.embed_query(normalized_query)
        rows = (
            self._table.search(query_type="hybrid")
            .vector(query_embedding)
            .text(normalized_query)
            .rerank(self._reranker)
            .limit(k)
            .to_list()
        )

        for row in rows:
            yield _row_to_document(row), row["_relevance_score"]


def _document_to_row(
    document: Document,
    embedding: Iterable[float],
) -> dict[str, Any]:
    return {
        "id": document.id,
        "text": document.text,
        "embedding": [float(value) for value in embedding],
        "source_path": str(document.source_path),
        "source_filename": document.source_filename,
        "source_hash": document.source_hash,
        "section_path": list(document.section_path),
    }


def _row_to_document(row: dict[str, Any]) -> Document:
    return Document(
        id=str(row["id"]),
        text=str(row["text"]),
        source_path=Path(row["source_path"]),
        source_filename=str(row["source_filename"]),
        source_hash=str(row["source_hash"]),
        section_path=list(row["section_path"]),
    )
