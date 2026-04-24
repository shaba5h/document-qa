from pathlib import Path

from document_qa.application.ask import AskUseCase
from document_qa.application.ingest import IngestUseCase
from document_qa.application.retrieve import RetrieveUseCase
from document_qa.domain.models import Document, QARequest, QAResponse


def make_document(text: str = "chunk text") -> Document:
    return Document(
        id="doc-1",
        text=text,
        source_path=Path("/tmp/source.pdf"),
        source_filename="source.pdf",
        source_hash="hash",
        section_path=["Section"],
    )


class FakeQuestionAnsweringAgent:
    def __init__(self) -> None:
        self.request: QARequest | None = None

    def run(self, request: QARequest) -> QAResponse:
        self.request = request
        return QAResponse(answer="answer", retrieved_documents=[])


def test_ask_use_case_wraps_question_for_agent() -> None:
    agent = FakeQuestionAnsweringAgent()
    use_case = AskUseCase(agent)

    response = use_case.execute("What changed?")

    assert response.answer == "answer"
    assert agent.request == QARequest(question="What changed?")


class FakeDocumentLoader:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.path: Path | None = None

    def load(self, path: Path):
        self.path = path
        return iter(self.documents)


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.added_documents: list[Document] = []

    def add_documents(self, documents) -> None:
        self.added_documents = list(documents)

    def search(self, query: str, k: int):
        yield make_document(query), 0.75


def test_ingest_use_case_streams_loader_documents_to_knowledge_base() -> None:
    document = make_document()
    loader = FakeDocumentLoader([document])
    knowledge_base = FakeKnowledgeBase()
    use_case = IngestUseCase(loader, knowledge_base)

    use_case.execute(Path("input.pdf"))

    assert loader.path == Path("input.pdf")
    assert knowledge_base.added_documents == [document]


def test_retrieve_use_case_delegates_query_and_k() -> None:
    knowledge_base = FakeKnowledgeBase()
    use_case = RetrieveUseCase(knowledge_base)

    result = list(use_case.execute("refund", 3))

    assert result == [(make_document("refund"), 0.75)]
