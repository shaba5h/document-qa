from __future__ import annotations

from pathlib import Path

from pydantic.dataclasses import dataclass


@dataclass
class Document:
    id: str

    text: str

    source_path: Path
    source_filename: str
    source_hash: str

    section_path: list[str]


@dataclass
class QARequest:
    question: str


@dataclass
class QAResponse:
    answer: str
    retrieved_documents: list[tuple[Document, float]]
