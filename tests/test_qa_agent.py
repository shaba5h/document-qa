from pathlib import Path
from types import SimpleNamespace
from typing import cast

from langchain_core.language_models import BaseChatModel

from document_qa.application.citations import format_answer_with_citations
from document_qa.domain.models import Document, QARequest
from document_qa.domain.ports import KnowledgeBase
from document_qa.infrastructure.langchain.qa_agent import (
    LangchainQuestionAnsweringAgent,
)


def make_document(
    document_id: str,
    text: str,
    filename: str,
    section: list[str],
) -> Document:
    return Document(
        id=document_id,
        text=text,
        source_path=Path("/tmp") / filename,
        source_filename=filename,
        source_hash=f"hash-{document_id}",
        section_path=section,
    )


class RecordingKnowledgeBase:
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        self.results = results
        self.search_calls: list[tuple[str, int]] = []

    def add_documents(self, documents) -> None:
        raise AssertionError("add_documents must not be called while answering")

    def search(self, query: str, k: int):
        self.search_calls.append((query, k))
        return iter(self.results)


class RecordingChatModel:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.answer)


def test_rag_agent_retrieves_once_builds_grounded_prompt_and_preserves_sources() -> None:
    refund_policy = make_document(
        "refunds",
        "Customers can request a refund within 30 calendar days.",
        "policy.pdf",
        ["Sales", "Refunds"],
    )
    shipping_policy = make_document(
        "shipping",
        "Express shipping takes two business days.",
        "shipping.pdf",
        ["Delivery"],
    )
    results = [(refund_policy, 0.91), (shipping_policy, 0.72)]
    knowledge_base = RecordingKnowledgeBase(results)
    chat_model = RecordingChatModel("Refunds are available for 30 days [1].")
    agent = LangchainQuestionAnsweringAgent(
        cast(BaseChatModel, chat_model),
        cast(KnowledgeBase, knowledge_base),
        k=2,
    )

    response = agent.run(QARequest(question="What is the refund window?"))

    assert knowledge_base.search_calls == [("What is the refund window?", 2)]
    assert len(chat_model.prompts) == 1
    prompt_messages = chat_model.prompts[0].to_messages()
    rendered_prompt = "\n".join(str(message.content) for message in prompt_messages)
    assert "Cite every evidence-backed claim" in rendered_prompt
    assert "What is the refund window?" in rendered_prompt
    assert '<doc index="1" score="0.91">' in rendered_prompt
    assert '<doc index="2" score="0.72">' in rendered_prompt
    assert "source: policy.pdf" in rendered_prompt
    assert "source: shipping.pdf" in rendered_prompt
    assert "section: Sales > Refunds" in rendered_prompt
    assert refund_policy.text in rendered_prompt
    assert shipping_policy.text in rendered_prompt
    assert response.retrieved_documents == results
    assert format_answer_with_citations(response).endswith(
        "Sources:\n- [1] policy.pdf - Sales > Refunds"
    )
