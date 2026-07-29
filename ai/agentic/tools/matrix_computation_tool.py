"""MatrixComputationTool: exact execution of structured matrix operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


Number = int | float
Matrix = List[List[Number]]
Vector = List[Number]


class MatrixComputationTool(BaseTool):
    """
    Execute matrix and convolution operations already classified by the
    Task & Context Analyzer.

    This component does not classify the request and does not select tools.
    """

    name = "MatrixComputationTool"
    description = (
        "Compute matrix compatibility, matrix products, dot products, "
        "valid two-dimensional correlation, kernel fit, and window counts."
    )
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
        operation_type = str(
            analyzed_task.get("operation_type")
            or ""
        )

        parameters = dict(
            analyzed_task.get("operation_parameters")
            or {}
        )

        handlers = {
            "matrix_compatibility": (
                self._execute_matrix_compatibility
            ),
            "matrix_multiplication": (
                self._execute_matrix_multiplication
            ),
            "matrix_entry": (
                self._execute_matrix_entry
            ),
            "dot_product": (
                self._execute_dot_product
            ),
            "valid_2d_convolution": (
                self._execute_valid_convolution
            ),
            "convolution_window_count": (
                self._execute_window_count
            ),
            "kernel_fit": (
                self._execute_kernel_fit
            ),
            "matrix_concept": (
                self._execute_matrix_concept
            ),
            "convolution_concept": (
                self._execute_convolution_concept
            ),
        }

        handler = handlers.get(operation_type)

        if not handler:
            return self._failed_result(
                attempt=attempt,
                operation=operation_type or "none",
                reason="unsupported_matrix_operation",
                output=(
                    "MatrixComputationTool received no supported "
                    "structured matrix operation from the "
                    "Task & Context Analyzer."
                ),
            )

        try:
            return handler(
                parameters=parameters,
                attempt=attempt,
            )

        except Exception as exc:
            return self._failed_result(
                attempt=attempt,
                operation=operation_type,
                reason="matrix_computation_failed",
                output=(
                    "MatrixComputationTool could not execute "
                    f"the requested operation: {exc}"
                ),
                extra_metadata={
                    "error": str(exc),
                    "operation_parameters": parameters,
                },
            )

    def _execute_matrix_concept(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        matrix_a = [
            [1, 2],
            [3, 4],
        ]

        matrix_b = [
            [5, 6],
            [7, 8],
        ]

        output_matrix = self._matrix_multiply(
            matrix_a,
            matrix_b,
        )

        output = (
            "Matrix multiplication uses rows from the "
            "first matrix and columns from the second "
            "matrix.\n\n"
            "Example matrices:\n\n"
            "A:\n"
            f"{self._format_matrix(matrix_a)}\n\n"
            "B:\n"
            f"{self._format_matrix(matrix_b)}\n\n"
            "The first output entry is:\n"
            "(1x5) + (2x7) = 19\n\n"
            "Full product:\n"
            f"{self._format_matrix(output_matrix)}"
        )

        return self._success_result(
            attempt=attempt,
            operation="matrix_concept",
            output=output,
            evidence=[
                (
                    "Matrix multiplication is performed "
                    "with row-by-column dot products."
                ),
                (
                    "The example product is "
                    f"{output_matrix}."
                ),
            ],
            references=[
                (
                    "MatrixComputationTool pedagogical "
                    "matrix example"
                )
            ],
            metadata={
                "matrix_a": matrix_a,
                "matrix_b": matrix_b,
                "output_matrix": output_matrix,
                "example_generated": True,
            },
        )

    def _execute_convolution_concept(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        input_matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        kernel = [
            [1, 0],
            [0, 1],
        ]

        output_matrix = self._valid_2d_correlation(
            input_matrix,
            kernel,
        )

        output = (
            "A convolution-style operation moves a small "
            "kernel across local regions of an input "
            "matrix.\n\n"
            "Input matrix:\n"
            f"{self._format_matrix(input_matrix)}\n\n"
            "Kernel:\n"
            f"{self._format_matrix(kernel)}\n\n"
            "At the top-left position:\n"
            "(1x1) + (2x0) + (4x0) + (5x1) = 6\n\n"
            "Full valid output:\n"
            f"{self._format_matrix(output_matrix)}"
        )

        return self._success_result(
            attempt=attempt,
            operation="convolution_concept",
            output=output,
            evidence=[
                (
                    "The kernel combines values from one "
                    "local input region at a time."
                ),
                (
                    "The example valid output is "
                    f"{output_matrix}."
                ),
            ],
            references=[
                (
                    "MatrixComputationTool pedagogical "
                    "convolution example"
                )
            ],
            metadata={
                "input_matrix": input_matrix,
                "kernel": kernel,
                "output_matrix": output_matrix,
                "example_generated": True,
            },
        )

    def _execute_matrix_compatibility(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        left_shape = self._normalize_shape(
            parameters.get("left_shape")
        )

        right_shape = self._normalize_shape(
            parameters.get("right_shape")
        )

        if not left_shape or not right_shape:
            raise ValueError(
                "Both left_shape and right_shape are required."
            )

        compatible = left_shape[1] == right_shape[0]

        result_shape = (
            [left_shape[0], right_shape[1]]
            if compatible
            else None
        )

        if compatible:
            explanation = (
                f"A {left_shape[0]}x{left_shape[1]} matrix "
                f"can be multiplied by a "
                f"{right_shape[0]}x{right_shape[1]} matrix "
                "because the inner dimensions are equal: "
                f"{left_shape[1]} = {right_shape[0]}.\n\n"
                "The resulting matrix shape is "
                f"{result_shape[0]}x{result_shape[1]}."
            )

        else:
            explanation = (
                f"A {left_shape[0]}x{left_shape[1]} matrix "
                f"cannot be multiplied by a "
                f"{right_shape[0]}x{right_shape[1]} matrix "
                "in that order because the inner dimensions "
                "are different: "
                f"{left_shape[1]} != {right_shape[0]}."
            )

        return self._success_result(
            attempt=attempt,
            operation="matrix_compatibility",
            output=explanation,
            evidence=[
                (
                    "Matrix multiplication is possible exactly "
                    "when the number of columns in the left "
                    "matrix equals the number of rows in the "
                    "right matrix."
                ),
                (
                    f"Compatibility result: {compatible}."
                ),
            ],
            references=[
                "MatrixComputationTool matrix-dimension verifier"
            ],
            metadata={
                "left_shape": left_shape,
                "right_shape": right_shape,
                "compatible": compatible,
                "result_shape": result_shape,
            },
        )

    def _execute_matrix_multiplication(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        matrix_a = self._normalize_matrix(
            parameters.get("matrix_a")
        )

        matrix_b = self._normalize_matrix(
            parameters.get("matrix_b")
        )

        if not matrix_a or not matrix_b:
            raise ValueError(
                "matrix_a and matrix_b are required."
            )

        shape_a = self._matrix_shape(matrix_a)
        shape_b = self._matrix_shape(matrix_b)

        compatible = shape_a[1] == shape_b[0]

        if not compatible:
            return self._success_result(
                attempt=attempt,
                operation="matrix_multiplication",
                output=(
                    f"The matrices cannot be multiplied in "
                    f"this order because A has shape "
                    f"{shape_a[0]}x{shape_a[1]} and B has "
                    f"shape {shape_b[0]}x{shape_b[1]}. "
                    f"The inner dimensions "
                    f"{shape_a[1]} and {shape_b[0]} "
                    "do not match."
                ),
                evidence=[
                    (
                        "The supplied matrices are incompatible "
                        "for multiplication in the requested order."
                    )
                ],
                references=[
                    "MatrixComputationTool exact matrix multiplication"
                ],
                metadata={
                    "matrix_a": matrix_a,
                    "matrix_b": matrix_b,
                    "shape_a": shape_a,
                    "shape_b": shape_b,
                    "compatible": False,
                    "output_matrix": None,
                },
            )

        output_matrix = self._matrix_multiply(
            matrix_a,
            matrix_b,
        )

        explanation = (
            "MatrixComputationTool multiplied the supplied "
            "matrices using row-by-column dot products.\n\n"
            "Matrix A:\n"
            f"{self._format_matrix(matrix_a)}\n\n"
            "Matrix B:\n"
            f"{self._format_matrix(matrix_b)}\n\n"
            "Result A x B:\n"
            f"{self._format_matrix(output_matrix)}"
        )

        return self._success_result(
            attempt=attempt,
            operation="matrix_multiplication",
            output=explanation,
            evidence=[
                (
                    "The exact matrix product is "
                    f"{output_matrix}."
                )
            ],
            references=[
                "MatrixComputationTool exact matrix multiplication"
            ],
            metadata={
                "matrix_a": matrix_a,
                "matrix_b": matrix_b,
                "shape_a": shape_a,
                "shape_b": shape_b,
                "compatible": True,
                "result_shape": (
                    self._matrix_shape(output_matrix)
                ),
                "output_matrix": output_matrix,
            },
        )

    def _execute_matrix_entry(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        matrix_a = self._normalize_matrix(
            parameters.get("matrix_a")
        )

        matrix_b = self._normalize_matrix(
            parameters.get("matrix_b")
        )

        entry = parameters.get("entry") or [0, 0]

        if not matrix_a or not matrix_b:
            raise ValueError(
                "matrix_a and matrix_b are required."
            )

        output_matrix = self._matrix_multiply(
            matrix_a,
            matrix_b,
        )

        row = int(entry[0])
        column = int(entry[1])

        if (
            row < 0
            or column < 0
            or row >= len(output_matrix)
            or column >= len(output_matrix[0])
        ):
            raise ValueError(
                "The requested matrix entry is out of range."
            )

        value = output_matrix[row][column]

        left_row = matrix_a[row]
        right_column = [
            matrix_b[index][column]
            for index in range(len(matrix_b))
        ]

        terms = [
            f"({left}x{right})"
            for left, right in zip(
                left_row,
                right_column,
            )
        ]

        explanation = (
            "The requested product entry is computed by "
            "taking the corresponding row from A and column "
            "from B.\n\n"
            f"{' + '.join(terms)} = {value}\n\n"
            f"The verified entry is {value}."
        )

        return self._success_result(
            attempt=attempt,
            operation="matrix_entry",
            output=explanation,
            evidence=[
                (
                    f"The requested matrix-product entry is "
                    f"{value}."
                ),
                (
                    f"The full product is {output_matrix}."
                ),
            ],
            references=[
                "MatrixComputationTool exact matrix-entry evaluator"
            ],
            metadata={
                "matrix_a": matrix_a,
                "matrix_b": matrix_b,
                "entry": [row, column],
                "entry_value": value,
                "output_matrix": output_matrix,
            },
        )

    def _execute_dot_product(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        left_vector = self._normalize_vector(
            parameters.get("left_vector")
        )

        right_vector = self._normalize_vector(
            parameters.get("right_vector")
        )

        used_default_example = False

        if not left_vector or not right_vector:
            # A conceptual row-by-column request may not supply
            # explicit vectors. A transparent educational example
            # is used in that case.
            left_vector = [1, 2]
            right_vector = [5, 7]
            used_default_example = True

        if len(left_vector) != len(right_vector):
            raise ValueError(
                "Dot-product vectors must have equal length."
            )

        products = [
            left * right
            for left, right in zip(
                left_vector,
                right_vector,
            )
        ]

        result = sum(products)

        terms = [
            f"({left}x{right})"
            for left, right in zip(
                left_vector,
                right_vector,
            )
        ]

        explanation = (
            "A dot product multiplies corresponding values "
            "and then adds those products.\n\n"
            f"{' + '.join(terms)} = {result}"
        )

        return self._success_result(
            attempt=attempt,
            operation="dot_product",
            output=explanation,
            evidence=[
                (
                    f"The verified dot product is {result}."
                )
            ],
            references=[
                "MatrixComputationTool exact dot-product evaluator"
            ],
            metadata={
                "left_vector": left_vector,
                "right_vector": right_vector,
                "products": products,
                "result": result,
                "used_default_example": (
                    used_default_example
                ),
            },
        )

    def _execute_valid_convolution(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        input_matrix = self._normalize_matrix(
            parameters.get("input_matrix")
        )

        kernel = self._normalize_matrix(
            parameters.get("kernel")
        )

        if not input_matrix or not kernel:
            raise ValueError(
                "input_matrix and kernel are required."
            )

        output_matrix = self._valid_2d_correlation(
            input_matrix,
            kernel,
        )

        first_patch_terms = []

        for kernel_row in range(len(kernel)):
            for kernel_col in range(len(kernel[0])):
                input_value = input_matrix[
                    kernel_row
                ][kernel_col]

                kernel_value = kernel[
                    kernel_row
                ][kernel_col]

                first_patch_terms.append(
                    f"({input_value}x{kernel_value})"
                )

        top_left_value = output_matrix[0][0]

        explanation = (
            "MatrixComputationTool computed the supplied "
            "CNN-style valid correlation exactly.\n\n"
            "Input matrix:\n"
            f"{self._format_matrix(input_matrix)}\n\n"
            "Kernel:\n"
            f"{self._format_matrix(kernel)}\n\n"
            "Top-left output value:\n"
            f"{' + '.join(first_patch_terms)} = "
            f"{top_left_value}\n\n"
            "Full valid output matrix:\n"
            f"{self._format_matrix(output_matrix)}"
        )

        return self._success_result(
            attempt=attempt,
            operation="valid_2d_convolution",
            output=explanation,
            evidence=[
                (
                    "The verified top-left convolution value "
                    f"is {top_left_value}."
                ),
                (
                    "The verified full output matrix is "
                    f"{output_matrix}."
                ),
            ],
            references=[
                (
                    "MatrixComputationTool exact CNN-style "
                    "valid correlation evaluator"
                )
            ],
            metadata={
                "input_matrix": input_matrix,
                "kernel": kernel,
                "input_shape": (
                    self._matrix_shape(input_matrix)
                ),
                "kernel_shape": (
                    self._matrix_shape(kernel)
                ),
                "output_shape": (
                    self._matrix_shape(output_matrix)
                ),
                "top_left_value": top_left_value,
                "output_matrix": output_matrix,
            },
        )

    def _execute_window_count(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        input_shape, kernel_shape = (
            self._resolve_convolution_shapes(
                parameters
            )
        )

        stride = self._positive_int(
            parameters.get("stride"),
            default=1,
        )

        padding = self._nonnegative_int(
            parameters.get("padding"),
            default=0,
        )

        output_shape = self._convolution_output_shape(
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            stride=stride,
            padding=padding,
        )

        output_rows, output_cols = output_shape
        window_count = output_rows * output_cols

        explanation = (
            f"A {kernel_shape[0]}x{kernel_shape[1]} "
            f"window has {output_rows} valid vertical "
            f"positions and {output_cols} valid horizontal "
            "positions inside the "
            f"{input_shape[0]}x{input_shape[1]} input.\n\n"
            f"Number of windows = {output_rows} x "
            f"{output_cols} = {window_count}."
        )

        return self._success_result(
            attempt=attempt,
            operation="convolution_window_count",
            output=explanation,
            evidence=[
                (
                    f"The verified number of valid windows "
                    f"is {window_count}."
                )
            ],
            references=[
                (
                    "MatrixComputationTool valid-window "
                    "count evaluator"
                )
            ],
            metadata={
                "input_shape": input_shape,
                "kernel_shape": kernel_shape,
                "stride": stride,
                "padding": padding,
                "output_shape": output_shape,
                "window_count": window_count,
            },
        )

    def _execute_kernel_fit(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        input_shape, kernel_shape = (
            self._resolve_convolution_shapes(
                parameters
            )
        )

        fits = (
            kernel_shape[0] <= input_shape[0]
            and kernel_shape[1] <= input_shape[1]
        )

        if fits:
            output_shape = (
                self._convolution_output_shape(
                    input_shape=input_shape,
                    kernel_shape=kernel_shape,
                    stride=1,
                    padding=0,
                )
            )

            window_count = (
                output_shape[0]
                * output_shape[1]
            )

            explanation = (
                f"Yes. A {kernel_shape[0]}x"
                f"{kernel_shape[1]} kernel can slide over "
                f"a {input_shape[0]}x{input_shape[1]} "
                "matrix in valid convolution.\n\n"
                f"The output shape is "
                f"{output_shape[0]}x{output_shape[1]}, "
                f"so there are {window_count} valid windows."
            )

        else:
            output_shape = None
            window_count = 0

            explanation = (
                f"No. A {kernel_shape[0]}x"
                f"{kernel_shape[1]} kernel is larger than "
                f"the {input_shape[0]}x{input_shape[1]} "
                "input in at least one dimension."
            )

        return self._success_result(
            attempt=attempt,
            operation="kernel_fit",
            output=explanation,
            evidence=[
                f"Kernel fit result: {fits}.",
                (
                    "Valid-window count: "
                    f"{window_count}."
                ),
            ],
            references=[
                "MatrixComputationTool kernel-fit evaluator"
            ],
            metadata={
                "input_shape": input_shape,
                "kernel_shape": kernel_shape,
                "fits": fits,
                "output_shape": output_shape,
                "window_count": window_count,
            },
        )

    def _resolve_convolution_shapes(
        self,
        parameters: Dict[str, Any],
    ) -> tuple[List[int], List[int]]:
        input_shape = self._normalize_shape(
            parameters.get("input_shape")
        )

        kernel_shape = self._normalize_shape(
            parameters.get("kernel_shape")
        )

        shapes = parameters.get("shapes") or []

        if (
            (not input_shape or not kernel_shape)
            and isinstance(shapes, list)
            and len(shapes) >= 2
        ):
            # For requests such as “3x3 windows in a 5x5
            # matrix”, the first shape is the window/kernel
            # and the second is the input.
            kernel_shape = (
                kernel_shape
                or self._normalize_shape(shapes[0])
            )

            input_shape = (
                input_shape
                or self._normalize_shape(shapes[1])
            )

        if not input_shape or not kernel_shape:
            raise ValueError(
                "input_shape and kernel_shape are required."
            )

        return input_shape, kernel_shape

    def _convolution_output_shape(
        self,
        input_shape: List[int],
        kernel_shape: List[int],
        stride: int,
        padding: int,
    ) -> List[int]:
        numerator_rows = (
            input_shape[0]
            + (2 * padding)
            - kernel_shape[0]
        )

        numerator_cols = (
            input_shape[1]
            + (2 * padding)
            - kernel_shape[1]
        )

        if numerator_rows < 0 or numerator_cols < 0:
            raise ValueError(
                "The kernel is larger than the padded input."
            )

        return [
            (numerator_rows // stride) + 1,
            (numerator_cols // stride) + 1,
        ]

    def _matrix_multiply(
        self,
        matrix_a: Matrix,
        matrix_b: Matrix,
    ) -> Matrix:
        rows_a, cols_a = self._matrix_shape(
            matrix_a
        )

        rows_b, cols_b = self._matrix_shape(
            matrix_b
        )

        if cols_a != rows_b:
            raise ValueError(
                "Matrix multiplication requires columns of "
                "A to equal rows of B."
            )

        output: Matrix = []

        for row in range(rows_a):
            output_row: List[Number] = []

            for column in range(cols_b):
                total: Number = 0

                for index in range(cols_a):
                    total += (
                        matrix_a[row][index]
                        * matrix_b[index][column]
                    )

                output_row.append(total)

            output.append(output_row)

        return output

    def _valid_2d_correlation(
        self,
        input_matrix: Matrix,
        kernel: Matrix,
    ) -> Matrix:
        input_rows, input_cols = (
            self._matrix_shape(input_matrix)
        )

        kernel_rows, kernel_cols = (
            self._matrix_shape(kernel)
        )

        output_rows = (
            input_rows - kernel_rows + 1
        )

        output_cols = (
            input_cols - kernel_cols + 1
        )

        if output_rows <= 0 or output_cols <= 0:
            raise ValueError(
                "The kernel does not fit inside the input."
            )

        output: Matrix = []

        for row in range(output_rows):
            output_row: List[Number] = []

            for column in range(output_cols):
                total: Number = 0

                for kernel_row in range(
                    kernel_rows
                ):
                    for kernel_column in range(
                        kernel_cols
                    ):
                        total += (
                            input_matrix[
                                row + kernel_row
                            ][
                                column + kernel_column
                            ]
                            * kernel[
                                kernel_row
                            ][
                                kernel_column
                            ]
                        )

                output_row.append(total)

            output.append(output_row)

        return output

    def _normalize_matrix(
        self,
        value: Any,
    ) -> Optional[Matrix]:
        if not isinstance(value, list) or not value:
            return None

        if not all(
            isinstance(row, list) and row
            for row in value
        ):
            return None

        width = len(value[0])

        if any(len(row) != width for row in value):
            return None

        if not all(
            isinstance(item, (int, float))
            for row in value
            for item in row
        ):
            return None

        return [
            list(row)
            for row in value
        ]

    def _normalize_vector(
        self,
        value: Any,
    ) -> Optional[Vector]:
        if not isinstance(value, list) or not value:
            return None

        if not all(
            isinstance(item, (int, float))
            for item in value
        ):
            return None

        return list(value)

    def _normalize_shape(
        self,
        value: Any,
    ) -> Optional[List[int]]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
        ):
            return None

        try:
            rows = int(value[0])
            columns = int(value[1])

        except (TypeError, ValueError):
            return None

        if rows <= 0 or columns <= 0:
            return None

        return [rows, columns]

    def _matrix_shape(
        self,
        matrix: Matrix,
    ) -> List[int]:
        return [
            len(matrix),
            len(matrix[0]),
        ]

    def _format_matrix(
        self,
        matrix: Matrix,
    ) -> str:
        return "\n".join(
            "  ".join(
                str(value)
                for value in row
            )
            for row in matrix
        )

    def _positive_int(
        self,
        value: Any,
        default: int,
    ) -> int:
        try:
            parsed = int(value)

        except (TypeError, ValueError):
            return default

        return parsed if parsed > 0 else default

    def _nonnegative_int(
        self,
        value: Any,
        default: int,
    ) -> int:
        try:
            parsed = int(value)

        except (TypeError, ValueError):
            return default

        return parsed if parsed >= 0 else default

    def _success_result(
        self,
        attempt: int,
        operation: str,
        output: str,
        evidence: List[str],
        references: List[str],
        metadata: Dict[str, Any],
    ) -> ToolResult:
        final_metadata = {
            "attempt": attempt,
            "source_component": self.name,
            "operation": operation,
            "operation_type": operation,
            "registry_tool": True,
            **metadata,
        }

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=evidence,
            references=references,
            metadata=final_metadata,
        )

    def _failed_result(
        self,
        attempt: int,
        operation: str,
        reason: str,
        output: str,
        extra_metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ToolResult:
        metadata = {
            "attempt": attempt,
            "source_component": self.name,
            "operation": operation,
            "failure_reason": reason,
            "registry_tool": True,
        }

        metadata.update(extra_metadata or {})

        return ToolResult(
            tool_name=self.name,
            status="failed",
            success=False,
            output=output,
            evidence=[],
            references=[
                "MatrixComputationTool execution policy"
            ],
            metadata=metadata,
        )
