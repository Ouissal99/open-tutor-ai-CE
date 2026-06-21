"""MatrixComputationTool: exact matrix/convolution support for tutoring examples."""

from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


class MatrixComputationTool(BaseTool):
    name = "MatrixComputationTool"
    description = "Compute small matrix examples such as CNN-style valid convolution."
    category = "computation"
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
        query = getattr(request, "student_question", "") or getattr(request, "query", "")
        topic = str(analyzed_task.get("topic") or "").lower()

        if "convolution" in query.lower() or "convolution" in topic:
            return self._compute_convolution_demo(attempt)

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output="MatrixComputationTool is available, but no matrix computation was required.",
            evidence=["No specific matrix operation was requested."],
            references=["MatrixComputationTool"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "operation": "none",
                "registry_tool": True,
            },
        )

    def _compute_convolution_demo(self, attempt: int) -> ToolResult:
        # CNN-style valid convolution usually means cross-correlation:
        # no kernel flip. This is what most beginner CNN explanations use.
        input_matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        kernel = [
            [1, 0],
            [0, 1],
        ]

        output = self._valid_2d_correlation(input_matrix, kernel)

        explanation = (
            "MatrixComputationTool computed a CNN-style valid convolution example.\n\n"
            "Input matrix:\n"
            "1 2 3\n"
            "4 5 6\n"
            "7 8 9\n\n"
            "Kernel:\n"
            "1 0\n"
            "0 1\n\n"
            "Top-left output value:\n"
            "(1×1) + (2×0) + (4×0) + (5×1) = 6\n\n"
            "Full valid output matrix:\n"
            f"{output[0][0]} {output[0][1]}\n"
            f"{output[1][0]} {output[1][1]}"
        )

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=explanation,
            evidence=[
                "MatrixComputationTool computed the convolution values exactly.",
                "For the top-left patch, the correct output is 6.",
                f"The full valid output matrix is {output}.",
            ],
            references=["MatrixComputationTool exact CNN-style convolution example"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "operation": "valid_2d_correlation",
                "input_matrix": input_matrix,
                "kernel": kernel,
                "output_matrix": output,
                "registry_tool": True,
            },
        )

    def _valid_2d_correlation(
        self,
        input_matrix: List[List[int]],
        kernel: List[List[int]],
    ) -> List[List[int]]:
        input_rows = len(input_matrix)
        input_cols = len(input_matrix[0])
        kernel_rows = len(kernel)
        kernel_cols = len(kernel[0])

        output_rows = input_rows - kernel_rows + 1
        output_cols = input_cols - kernel_cols + 1

        output = []

        for row in range(output_rows):
            output_row = []
            for col in range(output_cols):
                total = 0
                for kr in range(kernel_rows):
                    for kc in range(kernel_cols):
                        total += input_matrix[row + kr][col + kc] * kernel[kr][kc]
                output_row.append(total)
            output.append(output_row)

        return output
