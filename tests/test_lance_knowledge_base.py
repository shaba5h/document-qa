from pathlib import Path

import pytest

from document_qa.domain.models import Document
from document_qa.infrastructure.lance_knowledge_base import LanceKnowledgeBase

KEYWORDS = ("refund", "shipping", "vacation")
DOCUMENT_VECTORS = {
    "refund-policy": [1.0, 0.0, 0.0, 0.0],
    "refund-form": [-1.0, 0.0, 0.0, 0.0],
    "shipping-policy": [0.0, 1.0, 0.0, 0.0],
    "vacation-policy": [0.0, 0.0, 1.0, 0.0],
}


class KeywordEmbedder:
    def embed_documents(self, document):
        return [DOCUMENT_VECTORS[item.id] for item in document]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [float(keyword in normalized) for keyword in KEYWORDS] + [0.0]


class ShortEmbedder(KeywordEmbedder):
    def embed_documents(self, document):
        embeddings = super().embed_documents(document)
        return embeddings[:-1]


def make_document(document_id: str, text: str, section: str) -> Document:
    filename = f"{document_id}.pdf"
    return Document(
        id=document_id,
        text=text,
        source_path=Path("/knowledge") / filename,
        source_filename=filename,
        source_hash=f"hash-{document_id}",
        section_path=[section],
    )


DOCUMENTS = [
    make_document(
        "refund-policy",
        "Customers may get their money back within 30 calendar days.",
        "Refunds",
    ),
    make_document(
        "refund-form",
        "The refund form was archived and does not state eligibility or timing.",
        "Forms",
    ),
    make_document(
        "shipping-policy",
        "Express shipping takes 2 business days.",
        "Delivery",
    ),
    make_document(
        "vacation-policy",
        "Employees receive 20 vacation days per year.",
        "Benefits",
    ),
]


def test_hybrid_retrieval_passes_offline_gold_queries_and_preserves_provenance(
    tmp_path,
) -> None:
    knowledge_base = LanceKnowledgeBase(
        path=tmp_path / "lance",
        table_name="documents",
        embedder=KeywordEmbedder(),
        vector_weight=1.0,
        fts_weight=0.0,
    )
    knowledge_base.add_documents(DOCUMENTS)
    gold_queries = [
        ("What is the refund period?", "refund-policy"),
        ("How long does express shipping take?", "shipping-policy"),
        ("How many vacation days do employees receive?", "vacation-policy"),
    ]

    hits_at_one = 0
    for query, expected_document_id in gold_queries:
        hits = list(knowledge_base.search(query, k=1))
        assert len(hits) == 1
        document, score = hits[0]
        hits_at_one += document.id == expected_document_id
        assert document == next(
            candidate for candidate in DOCUMENTS if candidate.id == expected_document_id
        )
        assert isinstance(score, float)

    assert hits_at_one / len(gold_queries) == 1.0


def test_hybrid_weights_control_vector_and_fts_disagreement(tmp_path) -> None:
    vector_first = LanceKnowledgeBase(
        path=tmp_path / "lance",
        table_name="documents",
        embedder=KeywordEmbedder(),
        vector_weight=1.0,
        fts_weight=0.0,
    )
    vector_first.add_documents(DOCUMENTS)
    fts_first = LanceKnowledgeBase(
        path=tmp_path / "lance",
        table_name="documents",
        embedder=KeywordEmbedder(),
        vector_weight=0.0,
        fts_weight=1.0,
    )

    vector_hit = next(vector_first.search("refund period", k=1))[0]
    fts_hits = list(fts_first.search("refund period", k=len(DOCUMENTS)))

    assert vector_hit.id == "refund-policy"
    assert [document.id for document, _ in fts_hits] == ["refund-form"]
    assert all(score > 0 for _, score in fts_hits)


def test_ingestion_rejects_embedding_count_mismatch(tmp_path) -> None:
    knowledge_base = LanceKnowledgeBase(
        path=tmp_path / "lance",
        table_name="documents",
        embedder=ShortEmbedder(),
    )

    with pytest.raises(ValueError, match="3 vectors for 4 documents"):
        knowledge_base.add_documents(DOCUMENTS)
