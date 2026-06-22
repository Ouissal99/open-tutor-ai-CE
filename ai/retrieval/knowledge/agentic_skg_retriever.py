"""Agentic SKG retriever.

Phase 9A: JSONL keyword retriever.
Phase 9B can add ChromaDB/sentence-transformers while preserving this API.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List


class AgenticSKGRetriever:
    """Retrieve relevant course chunks from a local JSONL SKG index."""

    def __init__(self, index_dir: str | Path = "var/agentic_memory/skg_index"):
        self.index_dir = Path(index_dir)
        self.chunks_path = self.index_dir / "chunks.jsonl"

    def is_available(self) -> bool:
        return self.chunks_path.exists() and self.chunks_path.stat().st_size > 0

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        chunks = self._load_chunks()

        if not chunks:
            return []

        query_tokens = self._tokenize(query)

        scored = []

        for chunk in chunks:
            content = chunk.get("content", "")
            score = self._score(query_tokens, content)

            if score > 0:
                scored.append(
                    {
                        **chunk,
                        "score": score,
                    }
                )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _load_chunks(self) -> List[Dict]:
        if not self.chunks_path.exists():
            return []

        chunks = []

        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "with", "for",
            "is", "are", "be", "this", "that", "it", "as", "by", "on",
            "explain", "simple", "example",
        }

        return [token for token in tokens if token not in stopwords and len(token) > 2]

    def _score(self, query_tokens: List[str], content: str) -> float:
        content_tokens = self._tokenize(content)
        content_set = set(content_tokens)

        if not query_tokens or not content_tokens:
            return 0.0

        score = 0.0

        for token in query_tokens:
            if token in content_set:
                # Light term-frequency score.
                tf = content_tokens.count(token)
                score += 1.0 + math.log(1 + tf)

        # Small boost for convolution-specific educational terms.
        important_terms = {"convolution", "kernel", "matrix", "patch", "stride", "cnn"}
        score += 0.25 * len(set(query_tokens) & important_terms & content_set)

        return score
