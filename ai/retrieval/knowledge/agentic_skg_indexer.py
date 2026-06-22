"""Agentic SKG JSONL indexer."""

import json
from pathlib import Path
from typing import Dict, List

from ai.retrieval.knowledge.agentic_skg_chunker import AgenticSKGChunker
from ai.retrieval.knowledge.agentic_skg_loader import AgenticSKGLoader


class AgenticSKGIndexer:
    """Build a local JSONL SKG index from documents."""

    def __init__(
        self,
        index_dir: str | Path = "var/agentic_memory/skg_index",
        chunk_size: int = 180,
        chunk_overlap: int = 40,
    ):
        self.index_dir = Path(index_dir)
        self.documents_dir = self.index_dir / "documents"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.metadata_path = self.index_dir / "metadata.json"

        self.loader = AgenticSKGLoader()
        self.chunker = AgenticSKGChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build(self) -> Dict:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        documents = self.loader.load_directory(self.documents_dir)
        chunks = self.chunker.chunk_documents(documents)

        with self.chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        metadata = {
            "index_type": "jsonl_keyword",
            "index_dir": str(self.index_dir),
            "documents_dir": str(self.documents_dir),
            "chunks_path": str(self.chunks_path),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
        }

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return metadata
