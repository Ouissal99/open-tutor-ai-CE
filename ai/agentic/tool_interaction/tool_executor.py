from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tool_interaction.tool_registry import ToolRegistry


class ToolExecutor:
    """
    Executes external tools through the ToolRegistry.

    The executor no longer hardcodes mock tool branches.
    """

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()

    def execute(self, plan: Any, request: Any, attempt: int = 1) -> List[ToolResult]:
        results: List[ToolResult] = []

        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
        metadata = plan_dict.get("metadata", {})
        collected_context = metadata.get("collected_context", {})
        analyzed_task = {
            "task_type": plan_dict.get("task_type"),
            "selected_tools": plan_dict.get("selected_tools", []),
            "tool_selection": metadata.get("tool_selection", {}),
            "collected_context": collected_context,
            "topic": metadata.get("topic"),
            "current_step": metadata.get("current_step"),
        }

        for step in getattr(plan, "steps", []):
            tool_name = step.get("tool_name")

            try:
                tool = self.registry.get(tool_name)
                result = tool.run(
                    request=request,
                    step=step,
                    collected_context=collected_context,
                    analyzed_task=analyzed_task,
                    attempt=attempt,
                )
                results.append(result)

            except Exception as exc:
                results.append(
                    ToolResult(
                        tool_name=tool_name or "UnknownTool",
                        status="failed",
                        success=False,
                        output=f"Tool execution failed for {tool_name}: {exc}",
                        evidence=[],
                        references=[],
                        metadata={
                            "attempt": attempt,
                            "source_component": "ToolExecutor",
                            "failure_reason": "tool_registry_execution_failed",
                            "error": str(exc),
                            "available_tools": self.registry.list_tools(),
                        },
                    )
                )

        return results
