from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.base import BaseChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

from document_qa.settings import ChunkingSettings, EmbedderSettings


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def build_docling_converter() -> DocumentConverter:
    return DocumentConverter()


def build_docling_tokenizer(
    embedding_settings: EmbedderSettings, chunking_settings: ChunkingSettings
) -> BaseTokenizer:
    if embedding_settings.provider == "huggingface":
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )

        return HuggingFaceTokenizer.from_pretrained(
            model_name=embedding_settings.model_name,
            max_tokens=chunking_settings.max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {embedding_settings.provider}"
        )


def build_docling_chunker(tokenizer: BaseTokenizer) -> BaseChunker:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    return HybridChunker(
        tokenizer=tokenizer,
        repeat_table_header=True,
        omit_header_on_overflow=True,
        serializer_provider=MarkdownTableSerializerProvider(),
    )
