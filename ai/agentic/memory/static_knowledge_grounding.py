import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class StaticKnowledgeGrounding:
    """
    Lightweight DeepTutor-style Static Knowledge Grounding.

    This prototype follows the same role as DeepTutor SKG:
    it provides course/domain knowledge context for tutoring.

    Full DeepTutor may use graph + dense retrieval.
    This prototype uses keyword-based retrieval to stay simple and reproducible.
    """

    def __init__(
        self,
        kb_name: str = "default",
        base_dir: str = "var/agentic_memory/skg/knowledge_bases",
    ):
        self.kb_name = kb_name
        self.base_dir = Path(base_dir)
        self.kb_dir = self.base_dir / kb_name
        self.chunks_path = self.kb_dir / "chunks.json"

    def load_chunks(self) -> List[Dict[str, Any]]:
        if not self.chunks_path.exists():
            return []

        try:
            with self.chunks_path.open("r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception:
            return []

        if not isinstance(chunks, list):
            return []

        return chunks

    def retrieve(
        self,
        query: str,
        topic: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        chunks = self.load_chunks()

        query_words = self._normalize_words(query)
        topic_text = (topic or "").lower().strip()

        scored_chunks = []

        for chunk in chunks:
            chunk_text = " ".join([
                str(chunk.get("topic", "")),
                str(chunk.get("content", "")),
                " ".join(chunk.get("tags", []) or []),
            ]).lower()

            chunk_words = self._normalize_words(chunk_text)
            score = len(query_words.intersection(chunk_words))

            if topic_text and topic_text in chunk_text:
                score += 3

            if score > 0:
                item = dict(chunk)
                item["retrieval_score"] = score
                scored_chunks.append(item)

        scored_chunks.sort(key=lambda item: item["retrieval_score"], reverse=True)
        return scored_chunks[:limit]

    def build_grounding_context(
        self,
        query: str,
        topic: Optional[str] = None,
        limit: int = 3,
    ) -> Dict[str, Any]:
        snippets = self.retrieve(query=query, topic=topic, limit=limit)

        return {
            "kb_name": self.kb_name,
            "query": query,
            "topic": topic,
            "snippets": snippets,
            "sources": [item.get("source") for item in snippets if item.get("source")],
            "grounding_text": "\n".join(
                f"- {item.get('content', '')}"
                for item in snippets
            ),
        }

    def _normalize_words(self, text: str) -> set:
        cleaned = (
            text.lower()
            .replace(".", " ")
            .replace(",", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace('"', " ")
            .replace("'", " ")
            .replace("/", " ")
        )

        words = set()

        for word in cleaned.split():
            if len(word) >= 3:
                words.add(word)

        return words
