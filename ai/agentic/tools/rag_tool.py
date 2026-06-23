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
            retrieved_chunks = self.retriever.search(query=query, top_k=3)

        if retrieved_chunks:
            return self._build_result_from_retriever(
                query=query,
                chunks=retrieved_chunks,
                attempt=attempt,
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
        ]

        return " ".join(str(part) for part in parts if part)

    def _build_result_from_retriever(
        self,
        query: str,
        chunks: List[Dict],
        attempt: int,
    ) -> ToolResult:
        evidence = []
        references = []
        content_lines = []

        for chunk in chunks:
            text = chunk.get("content", "")
            source_file = chunk.get("source_file", "unknown source")
            page = chunk.get("page")
            score = chunk.get("score")

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
                "scores": [chunk.get("score") for chunk in chunks],
                "chunk_ids": [chunk.get("chunk_id") for chunk in chunks],
                "matched_terms": [chunk.get("matched_terms") for chunk in chunks],
                "score_breakdown": [chunk.get("score_breakdown") for chunk in chunks],
                "expanded_terms": chunks[0].get("expanded_terms", []) if chunks else [],
                "registry_tool": True,
            },
        )

    def _build_fallback_from_collected_context(
        self,
        collected_context: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        skg = collected_context.get("skg", {})
        snippets = skg.get("snippets", [])
        grounding_text = skg.get("grounding_text", "")

        if snippets:
            evidence = [item.get("content", "") for item in snippets if item.get("content")]
            references = [item.get("source", "") for item in snippets if item.get("source")]
            output = "Course evidence retrieved from fallback SKG context:\n" + grounding_text
        else:
            evidence = ["No SKG snippet found; fallback evidence used."]
            references = ["Fallback knowledge source"]
            output = "Fallback RAG result: no SKG snippet was retrieved."

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
                "retrieved_knowledge_chunks": len(snippets),
                "registry_tool": True,
            },
        )
