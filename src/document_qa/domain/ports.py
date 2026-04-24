from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Protocol

from document_qa.domain.models import Document, QARequest, QAResponse


class DocumentLoader(Protocol):
    def load(self, path: Path) -> Iterator[Document]: ...


class Embedder(Protocol):
    def embed_documents(self, document: Iterable[Document]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class KnowledgeBase(Protocol):
    def add_documents(self, documents: Iterable[Document]) -> None: ...

    def search(self, query: str, k: int) -> Iterator[tuple[Document, float]]: ...


class QuestionAnsweringAgent(Protocol):
    def run(self, request: QARequest) -> QAResponse: ...
