"""Agentic SKG retriever.

Phase 9A: JSONL keyword retriever.
Phase 9B: Resource-safe hybrid lexical retriever.

This module intentionally avoids heavy local vector dependencies such as:
- ChromaDB
- sentence-transformers
- torch

It uses lightweight scoring over local JSONL chunks:
- BM25-like lexical scoring
- phrase matching
- educational synonym expansion
- title/source boost
- transparent score metadata
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class AgenticSKGRetriever:
    """Retrieve relevant course chunks from a local JSONL SKG index."""

    SYNONYM_MAP = {
        "convolution": [
            "kernel",
            "filter",
            "matrix",
            "patch",
            "sliding",
            "cnn",
            "correlation",
        ],
        "kernel": [
            "filter",
            "mask",
            "window",
            "matrix",
        ],
        "matrix": [
            "grid",
            "array",
            "input",
            "output",
        ],
        "patch": [
            "window",
            "region",
            "local",
        ],
        "stride": [
            "step",
            "movement",
            "sliding",
        ],
        "image": [
            "pixel",
            "matrix",
            "grid",
        ],
        "neural": [
            "network",
            "cnn",
            "deep",
        ],
        "visual": [
            "diagram",
            "matrix",
            "step",
            "example",
        ],
    }

    IMPORTANT_TERMS = {
        "convolution",
        "kernel",
        "matrix",
        "patch",
        "stride",
        "cnn",
        "filter",
        "sliding",
        "output",
        "input",
    }

    def __init__(self, index_dir: str | Path = "var/agentic_memory/skg_index"):
        self.index_dir = Path(index_dir)
        self.chunks_path = self.index_dir / "chunks.jsonl"

    def is_available(self) -> bool:
        return self.chunks_path.exists() and self.chunks_path.stat().st_size > 0

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        chunks = self._load_chunks()

        if not chunks:
            return []

        query_terms, expanded_terms, phrases = self._build_query_profile(query)
        if not query_terms:
            return []

        document_stats = self._build_document_stats(chunks)
        scored = []

        for chunk in chunks:
            score, matched_terms, score_breakdown = self._score_chunk(
                chunk=chunk,
                query_terms=query_terms,
                phrases=phrases,
                document_stats=document_stats,
            )

            if score > 0:
                scored.append(
                    {
                        **chunk,
                        "score": round(score, 6),
                        "retrieval_mode": "jsonl_hybrid_lexical",
                        "matched_terms": matched_terms,
                        "expanded_terms": sorted(expanded_terms),
                        "score_breakdown": score_breakdown,
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

    def _build_query_profile(self, query: str) -> Tuple[Counter, set, List[str]]:
        base_tokens = self._tokenize(query)
        query_terms = Counter()
        expanded_terms = set()

        for token in base_tokens:
            query_terms[token] += 1.0

            for synonym in self.SYNONYM_MAP.get(token, []):
                query_terms[synonym] += 0.45
                expanded_terms.add(synonym)

        phrases = self._extract_phrases(base_tokens)
        return query_terms, expanded_terms, phrases

    def _extract_phrases(self, tokens: List[str]) -> List[str]:
        phrases = []

        for n in (2, 3):
            for i in range(0, max(0, len(tokens) - n + 1)):
                phrase = " ".join(tokens[i : i + n])
                if phrase:
                    phrases.append(phrase)

        domain_phrases = [
            "input matrix",
            "output matrix",
            "valid convolution",
            "kernel sliding",
            "sliding window",
            "local patch",
            "feature map",
        ]

        token_text = " ".join(tokens)
        for phrase in domain_phrases:
            phrase_tokens = phrase.split()
            if any(token in token_text for token in phrase_tokens):
                phrases.append(phrase)

        return list(dict.fromkeys(phrases))

    def _build_document_stats(self, chunks: List[Dict]) -> Dict:
        document_count = len(chunks)
        document_frequency = Counter()
        document_lengths = []

        for chunk in chunks:
            tokens = self._tokenize(self._chunk_search_text(chunk))
            document_lengths.append(len(tokens))

            for token in set(tokens):
                document_frequency[token] += 1

        average_length = (
            sum(document_lengths) / len(document_lengths)
            if document_lengths
            else 1.0
        )

        return {
            "document_count": document_count,
            "document_frequency": document_frequency,
            "average_length": average_length,
        }

    def _score_chunk(
        self,
        chunk: Dict,
        query_terms: Counter,
        phrases: List[str],
        document_stats: Dict,
    ) -> Tuple[float, List[str], Dict]:
        search_text = self._chunk_search_text(chunk)
        search_text_lower = search_text.lower()
        tokens = self._tokenize(search_text)

        if not tokens:
            return 0.0, [], {}

        token_counts = Counter(tokens)
        token_set = set(tokens)

        document_count = document_stats["document_count"]
        document_frequency = document_stats["document_frequency"]
        average_length = max(document_stats["average_length"], 1.0)

        k1 = 1.2
        b = 0.75
        document_length = len(tokens)

        lexical_score = 0.0
        matched_terms = []

        for term, query_weight in query_terms.items():
            tf = token_counts.get(term, 0)
            if tf <= 0:
                continue

            df = document_frequency.get(term, 0)
            idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
            denominator = tf + k1 * (1 - b + b * document_length / average_length)
            bm25_tf = (tf * (k1 + 1)) / denominator

            lexical_score += query_weight * idf * bm25_tf
            matched_terms.append(term)

        phrase_score = 0.0
        matched_phrases = []

        for phrase in phrases:
            if phrase and phrase in search_text_lower:
                phrase_score += 1.25
                matched_phrases.append(phrase)

        important_term_score = 0.25 * len(self.IMPORTANT_TERMS & token_set & set(query_terms))

        title_source_score = self._title_source_boost(chunk, set(query_terms))

        total_score = lexical_score + phrase_score + important_term_score + title_source_score

        score_breakdown = {
            "lexical_score": round(lexical_score, 6),
            "phrase_score": round(phrase_score, 6),
            "important_term_score": round(important_term_score, 6),
            "title_source_score": round(title_source_score, 6),
            "matched_phrases": matched_phrases,
        }

        return total_score, sorted(set(matched_terms)), score_breakdown

    def _title_source_boost(self, chunk: Dict, query_terms: set) -> float:
        metadata_text = " ".join(
            str(chunk.get(field, ""))
            for field in ("title", "source_file", "source", "section")
        ).lower()

        metadata_tokens = set(self._tokenize(metadata_text))
        return 0.35 * len(metadata_tokens & query_terms)

    def _chunk_search_text(self, chunk: Dict) -> str:
        parts = [
            chunk.get("title", ""),
            chunk.get("section", ""),
            chunk.get("source_file", ""),
            chunk.get("content", ""),
        ]

        return " ".join(str(part) for part in parts if part)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "with",
            "for",
            "is",
            "are",
            "be",
            "this",
            "that",
            "it",
            "as",
            "by",
            "on",
            "how",
            "what",
            "why",
            "use",
            "using",
            "explain",
            "simple",
            "example",
            "step",
            "steps",
            "student",
            "learner",
        }

        return [
            token
            for token in tokens
            if token not in stopwords and len(token) > 2
        ]
