"""TraceSearchTool: exposes trace-guided tutoring history as a tool result."""

from typing import Any, Dict

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


class TraceSearchTool(BaseTool):
    name = "TraceSearchTool"
    description = "Retrieve useful previous successful and failed tool interaction traces."
    category = "memory"
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
        summary = collected_context.get("summary", {})
        trace_context = collected_context.get("traces", {}) or collected_context.get("trace_toolkit", {})

        similar_count = summary.get("similar_successful_trace_count", 0)
        failed_count = summary.get("failed_trace_count", 0)

        selection = analyzed_task.get("tool_selection", {})
        reference_trace_id = selection.get("reference_trace_id")

        output = (
            "Trace guidance retrieved. "
            f"Similar successful traces: {similar_count}. "
            f"Failed traces considered: {failed_count}."
        )

        if reference_trace_id:
            output += f" Reusing strategy from trace {reference_trace_id}."

        evidence = [
            f"TraceToolkit found {similar_count} similar successful trace(s).",
            f"TraceToolkit considered {failed_count} failed trace(s).",
        ]

        if reference_trace_id:
            evidence.append(f"Reference trace used for tool selection: {reference_trace_id}.")

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=evidence,
            references=["TraceToolkit memory traces"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "similar_successful_trace_count": similar_count,
                "failed_trace_count": failed_count,
                "reference_trace_id": reference_trace_id,
                "trace_context": trace_context,
                "registry_tool": True,
            },
        )
