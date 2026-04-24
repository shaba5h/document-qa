from typing import Iterable

from document_qa.domain.models import Document


class HuggingFaceEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        embed_documents_prompt_name: str | None = None,
        embed_query_prompt_name: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.documents_encode_prompt_name = embed_documents_prompt_name
        self.query_encode_prompt_name = embed_query_prompt_name

    def embed_documents(self, document: Iterable[Document]) -> list[list[float]]:
        return self.model.encode_document(
            list(map(lambda d: d.text, document)),
            batch_size=self.batch_size,
            prompt_name=self.documents_encode_prompt_name,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode_query(
            text,
            batch_size=self.batch_size,
            prompt_name=self.query_encode_prompt_name,
        ).tolist()
