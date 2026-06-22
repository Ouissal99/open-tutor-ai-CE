"""Agentic SKG chunker.

Splits loaded course documents into retrieval chunks.
"""

import re
from typing import Dict, List

from ai.agentic.core.schemas import new_id


class AgenticSKGChunker:
    """Simple word-window chunker for Phase 9A."""

    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        chunks = []

        for document in documents:
            chunks.extend(self.chunk_document(document))

        return chunks

    def chunk_document(self, document: Dict) -> List[Dict]:
        content = self._clean_text(document.get("content", ""))
        words = content.split()

        if not words:
            return []

        chunks = []
        start = 0
        chunk_index = 1

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "chunk_id": new_id("SKG"),
                    "document_id": document.get("document_id"),
                    "source_file": document.get("source_file"),
                    "page": document.get("page"),
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "metadata": {
                        **(document.get("metadata") or {}),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                }
            )

            if end == len(words):
                break

            start = max(0, end - self.chunk_overlap)
            chunk_index += 1

        return chunks

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "")
        return text.strip()
