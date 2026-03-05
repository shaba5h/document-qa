from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_community.document_loaders import TextLoader
from langchain_docling.loader import DoclingLoader

LoaderFactory = Callable[[str], BaseLoader]

logger = logging.getLogger(__name__)


class UniversalDocumentLoader(BaseLoader):
    """LangChain-compatible loader that handles multiple file types and directories.

    Args:
        paths:         One or more file/directory paths.
        loaders:       A dictionary mapping file extensions (without dots) to LangChain
                       loader factories. Example:
                       {"pdf": DoclingLoader, "txt": lambda p: TextLoader(p, encoding="utf-8")}
        recursive:     Traverse directories recursively (default: True).
    """

    def __init__(
        self,
        paths: Path | list[Path],
        loaders: dict[str, LoaderFactory] | None = None,
        *,
        recursive: bool = True,
    ) -> None:
        if not loaders:
            loaders = {
                "pdf": DoclingLoader,
                "docx": DoclingLoader,
                "pptx": DoclingLoader,
                "html": DoclingLoader,
                "md": DoclingLoader,
                "xlsx": DoclingLoader,
                "asciidoc": DoclingLoader,
                "csv": DoclingLoader,
                "txt": lambda p: TextLoader(p, encoding="utf-8"),
            }

        if isinstance(paths, Path):
            paths = [paths]
        self._paths = paths
        self._recursive = recursive

        # Normalize extensions: lowercased and without leading dot
        self._loaders: dict[str, LoaderFactory] = {
            ext.lower().lstrip("."): factory for ext, factory in loaders.items()
        }

    def lazy_load(self) -> Iterator[Document]:
        for file_path in self._collect_files():
            yield from self._load_file(file_path)

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for raw in self._paths:
            p = raw.expanduser().resolve()
            if p.is_file():
                if self._ext(p) in self._loaders:
                    files.append(p)
                else:
                    logger.debug(f"Skipping: no loader for '.{p.suffix.lstrip('.')}' — {p.name}")
            elif p.is_dir():
                pattern = "**/*" if self._recursive else "*"
                for child in sorted(p.glob(pattern)):
                    if child.is_file() and self._ext(child) in self._loaders:
                        files.append(child)
            else:
                logger.warning(f"Path not found — {p}")

        seen: set[Path] = set()
        unique: list[Path] = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    def _load_file(self, path: Path) -> Iterator[Document]:
        ext = self._ext(path)
        factory = self._loaders[ext]
        logger.info(f"  Loading ({ext}): {path}")
        try:
            yield from factory(str(path)).lazy_load()
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  Error loading {path.name}: {exc}")

    @staticmethod
    def _ext(path: Path) -> str:
        return path.suffix.lower().lstrip(".")
