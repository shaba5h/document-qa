from document_qa.domain.models import QARequest, QAResponse
from document_qa.domain.ports import QuestionAnsweringAgent


class AskUseCase:
    def __init__(
        self,
        qa_agent: QuestionAnsweringAgent,
    ):
        self._qa_agent = qa_agent

    def execute(self, question: str) -> QAResponse:
        response = self._qa_agent.run(QARequest(question=question))

        return response
