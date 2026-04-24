from typing import Iterator

from document_qa.domain.models import Document
from document_qa.domain.ports import KnowledgeBase


class RetrieveUseCase:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base

    def execute(self, query: str, k: int) -> Iterator[tuple[Document, float]]:
        return self.knowledge_base.search(query, k)
