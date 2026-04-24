from document_qa.application.ask import AskUseCase
from document_qa.application.ingest import IngestUseCase
from document_qa.application.retrieve import RetrieveUseCase
from document_qa.domain.ports import (
    DocumentLoader,
    Embedder,
    KnowledgeBase,
    QuestionAnsweringAgent,
)
from document_qa.settings import Settings


def _build_document_loader(
    settings: Settings,
) -> DocumentLoader:
    from document_qa.infrastructure.docling.document_loader import DoclingDocumentLoader
    from document_qa.infrastructure.docling.factories import (
        build_docling_chunker,
        build_docling_converter,
        build_docling_tokenizer,
    )

    converter = build_docling_converter()
    tokenizer = build_docling_tokenizer(settings.embedder, settings.chunking)
    chunker = build_docling_chunker(tokenizer)

    return DoclingDocumentLoader(
        converter=converter,
        chunker=chunker,
    )


def _build_embedder(
    settings: Settings,
) -> Embedder:
    match settings.embedder.provider:
        case "huggingface":
            from document_qa.infrastructure.hf_embedder import HuggingFaceEmbedder

            return HuggingFaceEmbedder(
                model_name=settings.embedder.model_name,
                batch_size=settings.embedder.batch_size,
                embed_documents_prompt_name=settings.embedder.documents_encode_prompt_name,
                embed_query_prompt_name=settings.embedder.query_encode_prompt_name,
            )
        case _:
            raise ValueError(
                f"Unknown embedding provider: {settings.embedder.provider}"
            )


def _build_knowledge_base(
    settings: Settings,
    embedder: Embedder,
) -> KnowledgeBase:
    match settings.knowledge_base.provider:
        case "lance":
            from document_qa.infrastructure.lance_knowledge_base import (
                LanceKnowledgeBase,
            )

            return LanceKnowledgeBase(
                table_name=settings.knowledge_base.table_name,
                path=settings.knowledge_base.path,
                embedder=embedder,
                vector_weight=settings.retrieval.vector_weight,
                fts_weight=settings.retrieval.fts_weight,
            )
        case _:
            raise ValueError(
                f"Unknown knowledge base provider: {settings.knowledge_base.provider}"
            )


def _build_qa_agent(
    settings: Settings, knowledge_base: KnowledgeBase
) -> QuestionAnsweringAgent:
    from document_qa.infrastructure.langchain.factories import (
        build_langchain_chat_model,
    )
    from document_qa.infrastructure.langchain.qa_agent import (
        LangchainQuestionAnsweringAgent,
    )

    chat_model = build_langchain_chat_model(settings.chatmodel)
    return LangchainQuestionAnsweringAgent(
        chat_model, knowledge_base, k=settings.retrieval.k
    )


def build_ingest_use_case(
    settings: Settings,
) -> IngestUseCase:
    document_loader = _build_document_loader(settings)
    embedder = _build_embedder(settings)
    knowledge_base = _build_knowledge_base(settings, embedder)

    return IngestUseCase(document_loader, knowledge_base)


def build_retrieve_use_case(settings: Settings) -> RetrieveUseCase:
    embedder = _build_embedder(settings)
    knowledge_base = _build_knowledge_base(settings, embedder)

    return RetrieveUseCase(knowledge_base)


def build_ask_use_case(
    settings: Settings,
) -> AskUseCase:
    embedder = _build_embedder(settings)
    knowledge_base = _build_knowledge_base(settings, embedder)
    qa_agent = _build_qa_agent(settings, knowledge_base)

    return AskUseCase(qa_agent)
