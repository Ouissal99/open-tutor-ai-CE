"""RAGTool: retrieves course-grounded evidence from OpenTutorAI SKG retrieval."""

from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool
from ai.retrieval.knowledge.agentic_skg_retriever import AgenticSKGRetriever


class RAGTool(BaseTool):
    name = "RAGTool"
    description = "Retrieve course-grounded evidence from the OpenTutorAI knowledge retrieval layer."
    category = "knowledge"
    risk_level = "low"
    requires_network = False
    requires_sandbox = False

    CONVOLUTION_TERMS = [
        "convolution",
        "kernel",
        "cross-correlation",
        "sliding window",
        "slide the kernel",
        "kernel sliding",
        "local patch",
        "cnn",
    ]

    MATRIX_MULTIPLICATION_TERMS = [
        "matrix multiplication",
        "matrix_multiplication",
        "multiply matrices",
        "multiplication of matrices",
        "row by column",
        "row-by-column",
        "dot product",
    ]

    def __init__(self, retriever: AgenticSKGRetriever | None = None):
        self.retriever = retriever or AgenticSKGRetriever()

    def run(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> ToolResult:
        query = self._build_query(request, step, analyzed_task)

        retrieved_chunks = []
        if self.retriever.is_available():
            retrieved_chunks = self.retriever.search(query=query, top_k=5)

        filtered_chunks = self._filter_topic_consistent_chunks(
            chunks=retrieved_chunks,
            request=request,
            step=step,
            analyzed_task=analyzed_task,
        )

        if filtered_chunks:
            return self._build_result_from_retriever(
                query=query,
                chunks=filtered_chunks,
                attempt=attempt,
                original_chunk_count=len(retrieved_chunks),
            )

        if self._is_plain_matrix_multiplication_request(request=request, step=step, analyzed_task=analyzed_task):
            return self._build_matrix_multiplication_topic_fallback(
                query=query,
                retrieved_chunk_count=len(retrieved_chunks),
                attempt=attempt,
            )

        if retrieved_chunks:
            return self._build_result_from_retriever(
                query=query,
                chunks=retrieved_chunks[:3],
                attempt=attempt,
                original_chunk_count=len(retrieved_chunks),
            )

        return self._build_fallback_from_collected_context(
            collected_context=collected_context,
            attempt=attempt,
        )

    def _build_query(self, request: Any, step: Dict[str, Any], analyzed_task: Dict[str, Any]) -> str:
        parts = [
            getattr(request, "student_question", ""),
            getattr(request, "step_goal", ""),
            getattr(request, "expected_output", ""),
            analyzed_task.get("topic", ""),
            analyzed_task.get("current_step", ""),
            step.get("purpose", ""),
            step.get("goal", ""),
            step.get("required_capability", ""),
        ]

        query = " ".join(str(part) for part in parts if part)

        if self._is_plain_matrix_multiplication_request(request=request, step=step, analyzed_task=analyzed_task):
            query += " row by column dot product matrix product A times B"

        return query

    def _filter_topic_consistent_chunks(
        self,
        chunks: List[Dict],
        request: Any,
        step: Dict[str, Any],
        analyzed_task: Dict[str, Any],
    ) -> List[Dict]:
        if not chunks:
            return []

        if not self._is_plain_matrix_multiplication_request(request=request, step=step, analyzed_task=analyzed_task):
            return chunks[:3]

        filtered = []

        for chunk in chunks:
            text = self._chunk_text(chunk)
            if self._is_convolution_text(text):
                continue
            filtered.append(chunk)

        return filtered[:3]

    def _build_result_from_retriever(
        self,
        query: str,
        chunks: List[Dict],
        attempt: int,
        original_chunk_count: int | None = None,
    ) -> ToolResult:
        evidence = []
        references = []
        content_lines = []

        for chunk in chunks:
            text = chunk.get("content", "")
            source_file = chunk.get("source_file", "unknown source")
            page = chunk.get("page")

            if text:
                evidence.append(text)

            reference = source_file
            if page is not None:
                reference += f", page {page}"

            references.append(reference)
            content_lines.append(f"- {text}")

        output = (
            "Course evidence retrieved from OpenTutorAI Agentic SKG index:\n"
            + "\n".join(content_lines)
        )

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=evidence,
            references=references,
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "grounding_source": "OpenTutorAI Agentic SKG Hybrid Retriever",
                "retrieval_mode": chunks[0].get("retrieval_mode", "jsonl_hybrid_lexical") if chunks else "jsonl_hybrid_lexical",
                "query": query,
                "retrieved_knowledge_chunks": len(chunks),
                "original_retrieved_knowledge_chunks": original_chunk_count if original_chunk_count is not None else len(chunks),
                "topic_filter_removed_chunks": max((original_chunk_count or len(chunks)) - len(chunks), 0),
                "scores": [chunk.get("score") for chunk in chunks],
                "chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
                "matched_terms": [chunk.get("matched_terms") for chunk in chunks],
                "score_breakdown": [chunk.get("score_breakdown") for chunk in chunks],
                "expanded_terms": chunks[0].get("expanded_terms", []) if chunks else [],
                "registry_tool": True,
            },
        )

    def _build_matrix_multiplication_topic_fallback(
        self,
        query: str,
        retrieved_chunk_count: int,
        attempt: int,
    ) -> ToolResult:
        output = (
            "Topic-consistent matrix multiplication support:\n"
            "- Matrix multiplication combines rows of the first matrix with columns of the second matrix.\n"
            "- Each output value is a dot product between one row and one column.\n"
            "- Example:\n"
            "  A = [[1, 2], [3, 4]]\n"
            "  B = [[5, 6], [7, 8]]\n"
            "  C[1,1] = (1×5) + (2×7) = 19\n"
            "  C[1,2] = (1×6) + (2×8) = 22\n"
            "  C[2,1] = (3×5) + (4×7) = 43\n"
            "  C[2,2] = (3×6) + (4×8) = 50\n"
            "  Therefore C = [[19, 22], [43, 50]]."
        )

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=[
                "Matrix multiplication computes each result cell as a row-by-column dot product.",
                "For A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]], the result is [[19, 22], [43, 50]].",
            ],
            references=["RAGTool topic-consistency fallback: matrix multiplication"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "grounding_source": "Topic-consistency fallback after SKG topic filtering",
                "retrieval_mode": "topic_consistency_fallback",
                "query": query,
                "retrieved_knowledge_chunks": 0,
                "original_retrieved_knowledge_chunks": retrieved_chunk_count,
                "topic_filter_removed_chunks": retrieved_chunk_count,
                "registry_tool": True,
            },
        )

    def _build_fallback_from_collected_context(
        self,
        collected_context: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        skg = collected_context.get("skg", {}) or {}
        snippets = skg.get("snippets", []) or []
        grounding_text = skg.get("grounding_text", "")

        filtered_snippets = []

        for item in snippets:
            text = " ".join(
                [
                    str(item.get("content", "")),
                    str(item.get("source", "")),
                ]
            ).lower()

            if self._is_convolution_text(text):
                continue

            filtered_snippets.append(item)

        if filtered_snippets:
            evidence = [item.get("content", "") for item in filtered_snippets if item.get("content")]
            references = [item.get("source", "") for item in filtered_snippets if item.get("source")]
            output = "Course evidence retrieved from fallback SKG context:\n" + grounding_text
        else:
            evidence = ["No topic-consistent SKG snippet found; fallback evidence used."]
            references = ["Fallback knowledge source"]
            output = "Fallback RAG result: no topic-consistent SKG snippet was retrieved."

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=evidence,
            references=references,
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "grounding_source": "Fallback StaticKnowledgeGrounding context",
                "retrieval_mode": "fallback_collected_context",
                "retrieved_knowledge_chunks": len(filtered_snippets),
                "registry_tool": True,
            },
        )

    def _is_plain_matrix_multiplication_request(
        self,
        request: Any,
        step: Dict[str, Any],
        analyzed_task: Dict[str, Any],
    ) -> bool:
        text = self._combined_request_text(request=request, step=step, analyzed_task=analyzed_task)

        has_matrix_multiplication = any(term in text for term in self.MATRIX_MULTIPLICATION_TERMS)
        has_convolution = any(term in text for term in self.CONVOLUTION_TERMS)

        return has_matrix_multiplication and not has_convolution

    def _combined_request_text(
        self,
        request: Any,
        step: Dict[str, Any],
        analyzed_task: Dict[str, Any],
    ) -> str:
        parts = [
            getattr(request, "student_question", ""),
            getattr(request, "user_query", ""),
            getattr(request, "query", ""),
            getattr(request, "current_step", ""),
            getattr(request, "step_goal", ""),
            getattr(request, "expected_output", ""),
            analyzed_task.get("topic", ""),
            analyzed_task.get("task_type", ""),
            analyzed_task.get("current_step", ""),
            analyzed_task.get("expected_output", ""),
            step.get("goal", ""),
            step.get("purpose", ""),
            step.get("required_capability", ""),
        ]

        return " ".join(str(part) for part in parts if part).lower()

    def _chunk_text(self, chunk: Dict) -> str:
        return " ".join(
            [
                str(chunk.get("content", "")),
                str(chunk.get("source_file", "")),
                str(chunk.get("title", "")),
                str(chunk.get("chunk_id", "")),
            ]
        ).lower()

    def _is_convolution_text(self, text: str) -> bool:
        return any(term in text for term in self.CONVOLUTION_TERMS)