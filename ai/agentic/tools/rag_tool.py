"""RAGTool: retrieves course-grounded evidence from SKG context."""

from typing import Any, Dict

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


class RAGTool(BaseTool):
    name = "RAGTool"
    description = "Retrieve course-grounded evidence from the Static Knowledge Grounding context."
    category = "knowledge"
    risk_level = "low"
    requires_network = False
    requires_sandbox = False

    def run(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> ToolResult:
        skg = collected_context.get("skg", {})
        snippets = skg.get("snippets", [])
        grounding_text = skg.get("grounding_text", "")

        if snippets:
            evidence = [item.get("content", "") for item in snippets if item.get("content")]
            references = [item.get("source", "") for item in snippets if item.get("source")]
            output = "Course evidence retrieved from SKG:\n" + grounding_text
            status = "success"
            success = True
        else:
            evidence = ["No SKG snippet found; fallback evidence used."]
            references = ["Fallback knowledge source"]
            output = "Fallback RAG result: no SKG snippet was retrieved."
            status = "success"
            success = True

        return ToolResult(
            tool_name=self.name,
            status=status,
            success=success,
            output=output,
            evidence=evidence,
            references=references,
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "grounding_source": "StaticKnowledgeGrounding",
                "retrieved_knowledge_chunks": len(snippets),
                "registry_tool": True,
            },
        )
