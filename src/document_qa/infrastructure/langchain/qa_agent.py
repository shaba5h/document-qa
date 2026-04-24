from langchain_core.language_models import BaseChatModel

from document_qa.domain.models import QARequest, QAResponse
from document_qa.domain.ports import KnowledgeBase
from document_qa.infrastructure.langchain.context import (
    build_compact_context,
)
from document_qa.infrastructure.langchain.prompts import (
    grounded_answer_system_prompt,
)


class LangchainQuestionAnsweringAgent:
    def __init__(
        self,
        chat_model: BaseChatModel,
        knowledge_base: KnowledgeBase,
        *,
        k: int,
    ):
        self._chat_model = chat_model
        self._knowledge_base = knowledge_base
        self._retrieval_k = k

    def run(self, request: QARequest) -> QAResponse:

        retrieval_results = list(
            self._knowledge_base.search(request.question, k=self._retrieval_k)
        )

        context = build_compact_context(retrieval_results)
        messages = grounded_answer_system_prompt.invoke(
            {"question": request.question, "context": context}
        )
        answer = self._chat_model.invoke(messages)

        response = QAResponse(answer=answer.text, retrieved_documents=retrieval_results)

        return response
