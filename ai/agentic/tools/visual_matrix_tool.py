"""VisualMatrixTool: creates structured visual explanation support for matrix operations."""

from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


class VisualMatrixTool(BaseTool):
    name = "VisualMatrixTool"
    description = "Generate beginner-friendly visual support for matrix operations and convolution examples."
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
        dpm = collected_context.get("dpm", {}) or {}
        learner_level = dpm.get("learner_level", "unknown")
        preferred_examples = dpm.get("preferred_examples", []) or []

        mode = self._detect_visual_mode(request=request, step=step, analyzed_task=analyzed_task)

        if mode == "convolution":
            output = self._build_convolution_visual_output(
                learner_level=learner_level,
                preferred_examples=preferred_examples,
            )
            evidence = [
                "VisualMatrixTool generated a convolution visual explanation with kernel movement.",
                "The visual example is adapted using learner preferences from DPM.",
            ]
            references = ["VisualMatrixTool generated convolution visual example"]

        else:
            output = self._build_matrix_multiplication_visual_output(
                learner_level=learner_level,
                preferred_examples=preferred_examples,
            )
            evidence = [
                "VisualMatrixTool generated a matrix multiplication visual explanation using row-by-column dot products.",
                "The visual example is adapted using learner preferences from DPM.",
            ]
            references = ["VisualMatrixTool generated matrix multiplication visual example"]

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
                "personalization_source": "DynamicPersonalMemory",
                "learner_level": learner_level,
                "preferred_examples": preferred_examples,
                "visual_mode": mode,
                "registry_tool": True,
            },
        )

    def _detect_visual_mode(self, request: Any, step: Dict[str, Any], analyzed_task: Dict[str, Any]) -> str:
        text = self._combined_text(request=request, step=step, analyzed_task=analyzed_task)

        convolution_terms = [
            "convolution",
            "kernel",
            "cnn",
            "cross-correlation",
            "sliding window",
            "slide the kernel",
            "local patch",
        ]

        matrix_multiplication_terms = [
            "matrix multiplication",
            "matrix_multiplication",
            "multiply matrices",
            "multiplication of matrices",
            "row by column",
            "row-by-column",
            "dot product",
        ]

        has_convolution = any(term in text for term in convolution_terms)
        has_matrix_multiplication = any(term in text for term in matrix_multiplication_terms)

        if has_convolution and not has_matrix_multiplication:
            return "convolution"

        return "matrix_multiplication"

    def _build_matrix_multiplication_visual_output(
        self,
        learner_level: str,
        preferred_examples: List[Any],
    ) -> str:
        output = (
            "Generated visual explanation for matrix multiplication.\n\n"
            "Use two small matrices:\n\n"
            "A =\n"
            "1  2\n"
            "3  4\n\n"
            "B =\n"
            "5  6\n"
            "7  8\n\n"
            "To get each value in the result matrix C, take one row from A and one column from B.\n\n"
            "Top-left value C[1,1]:\n"
            "row 1 of A · column 1 of B = (1×5) + (2×7) = 19\n\n"
            "Top-right value C[1,2]:\n"
            "row 1 of A · column 2 of B = (1×6) + (2×8) = 22\n\n"
            "Bottom-left value C[2,1]:\n"
            "row 2 of A · column 1 of B = (3×5) + (4×7) = 43\n\n"
            "Bottom-right value C[2,2]:\n"
            "row 2 of A · column 2 of B = (3×6) + (4×8) = 50\n\n"
            "So the result is:\n\n"
            "C =\n"
            "19  22\n"
            "43  50\n\n"
            f"The explanation is adapted for learner_level={learner_level}."
        )

        if preferred_examples:
            output += f" Preferred example style: {', '.join(str(item) for item in preferred_examples)}."

        return output

    def _build_convolution_visual_output(
        self,
        learner_level: str,
        preferred_examples: List[Any],
    ) -> str:
        output = (
            "Generated visual explanation for convolution.\n\n"
            "Use a 2x2 kernel sliding over a 3x3 input matrix. "
            "At each position, the kernel is aligned with one local patch. "
            "The corresponding values are multiplied and then summed to produce one output value. "
            f"The explanation is adapted for learner_level={learner_level}."
        )

        if preferred_examples:
            output += f" Preferred example style: {', '.join(str(item) for item in preferred_examples)}."

        return output

    def _combined_text(self, request: Any, step: Dict[str, Any], analyzed_task: Dict[str, Any]) -> str:
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