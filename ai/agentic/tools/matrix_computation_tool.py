"""MatrixComputationTool: exact matrix operation support for tutoring examples."""

from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


class MatrixComputationTool(BaseTool):
    name = "MatrixComputationTool"
    description = "Compute small matrix examples such as matrix multiplication and CNN-style valid convolution."
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
        text = self._combined_text(request=request, step=step, analyzed_task=analyzed_task)

        if self._is_convolution_request(text):
            return self._compute_convolution_demo(attempt)

        if self._is_matrix_multiplication_request(text):
            return self._compute_matrix_multiplication_demo(attempt)

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output="MatrixComputationTool is available, but no explicit matrix computation was required for this step.",
            evidence=["No specific matrix operation was requested."],
            references=["MatrixComputationTool"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "operation": "none",
                "registry_tool": True,
            },
        )

    def _compute_matrix_multiplication_demo(self, attempt: int) -> ToolResult:
        matrix_a = [
            [1, 2],
            [3, 4],
        ]

        matrix_b = [
            [5, 6],
            [7, 8],
        ]

        output = self._matrix_multiply(matrix_a, matrix_b)

        explanation = (
            "MatrixComputationTool computed a standard matrix multiplication example.\n\n"
            "Matrix A:\n"
            "1 2\n"
            "3 4\n\n"
            "Matrix B:\n"
            "5 6\n"
            "7 8\n\n"
            "Each output value is computed using a row-by-column dot product.\n\n"
            "C[1,1] = (1×5) + (2×7) = 19\n"
            "C[1,2] = (1×6) + (2×8) = 22\n"
            "C[2,1] = (3×5) + (4×7) = 43\n"
            "C[2,2] = (3×6) + (4×8) = 50\n\n"
            "Full result matrix C = A × B:\n"
            f"{output[0][0]} {output[0][1]}\n"
            f"{output[1][0]} {output[1][1]}"
        )

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=explanation,
            evidence=[
                "MatrixComputationTool computed a standard matrix multiplication example exactly.",
                "The top-left result is C[1,1] = (1×5) + (2×7) = 19.",
                f"The full result matrix is {output}.",
            ],
            references=["MatrixComputationTool exact matrix multiplication example"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "operation": "matrix_multiplication",
                "matrix_a": matrix_a,
                "matrix_b": matrix_b,
                "output_matrix": output,
                "registry_tool": True,
            },
        )

    def _compute_convolution_demo(self, attempt: int) -> ToolResult:
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

    def _matrix_multiply(
        self,
        matrix_a: List[List[int]],
        matrix_b: List[List[int]],
    ) -> List[List[int]]:
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])

        if cols_a != rows_b:
            raise ValueError("Matrix multiplication requires columns of A to equal rows of B.")

        output = []

        for row in range(rows_a):
            output_row = []
            for col in range(cols_b):
                total = 0
                for k in range(cols_a):
                    total += matrix_a[row][k] * matrix_b[k][col]
                output_row.append(total)
            output.append(output_row)

        return output

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

    def _is_matrix_multiplication_request(self, text: str) -> bool:
        matrix_terms = [
            "matrix multiplication",
            "matrix_multiplication",
            "multiply matrices",
            "multiplication of matrices",
            "row by column",
            "row-by-column",
            "dot product",
        ]

        return any(term in text for term in matrix_terms)

    def _is_convolution_request(self, text: str) -> bool:
        convolution_terms = [
            "convolution",
            "kernel",
            "cnn",
            "cross-correlation",
            "sliding window",
            "slide the kernel",
            "local patch",
        ]

        return any(term in text for term in convolution_terms)