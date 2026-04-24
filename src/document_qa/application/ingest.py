from pathlib import Path

from document_qa.domain.ports import DocumentLoader, KnowledgeBase


class IngestUseCase:
    def __init__(
        self,
        document_loader: DocumentLoader,
        knowledge_base: KnowledgeBase,
    ):
        self.document_loader = document_loader
        self.knowledge_base = knowledge_base

    def execute(self, path: Path) -> None:
        docs_iter = self.document_loader.load(path)

        self.knowledge_base.add_documents(docs_iter)
