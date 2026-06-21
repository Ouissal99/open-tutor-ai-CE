"""VisualMatrixTool: creates structured matrix/kernel visual explanation support."""

from typing import Any, Dict

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


class VisualMatrixTool(BaseTool):
    name = "VisualMatrixTool"
    description = "Generate beginner-friendly visual support for matrix and convolution examples."
    category = "visualization"
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
        dpm = collected_context.get("dpm", {})
        learner_level = dpm.get("learner_level", "unknown")
        preferred_examples = dpm.get("preferred_examples", [])

        output = (
            "Generated visual explanation: use a 2x2 kernel sliding over a 3x3 matrix. "
            "The kernel moves over each local patch; each patch is multiplied element-wise "
            "with the kernel and summed to produce one output value. "
            f"The explanation is adapted for learner_level={learner_level}."
        )

        if preferred_examples:
            output += f" Preferred example style: {', '.join(preferred_examples)}."

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=[
                "VisualMatrixTool describes kernel movement step by step.",
                "The visual example is adapted using learner preferences from DPM.",
            ],
            references=["VisualMatrixTool generated matrix example"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "personalization_source": "DynamicPersonalMemory",
                "learner_level": learner_level,
                "preferred_examples": preferred_examples,
                "registry_tool": True,
            },
        )
