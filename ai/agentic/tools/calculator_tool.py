"""CalculatorTool: deterministic scalar and convolution-size computation."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict, List, Optional

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool(BaseTool):
    """
    Execute deterministic scalar arithmetic and convolution output-size
    calculations.

    Task classification remains the responsibility of TaskContextAnalyzer.
    This tool only executes the structured operation it receives.
    """

    name = "CalculatorTool"
    description = (
        "Evaluate safe scalar arithmetic expressions and deterministic "
        "convolution output-size formulas."
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

        combined_text = self._combined_text(
            request=request,
            step=step,
            analyzed_task=analyzed_task,
        )

        if (
            operation_type == "convolution_output_size"
            or (
                not operation_type
                and self._is_convolution_output_size_request(
                    combined_text
                )
            )
        ):
            if not parameters.get("input_shape"):
                parameters.update(
                    self._extract_convolution_parameters(
                        combined_text
                    )
                )

            return self._build_output_size_result(
                parameters=parameters,
                attempt=attempt,
            )

        explicit_expression = (
            parameters.get("expression")
            or step.get("expression")
            or analyzed_task.get("expression")
            or (
                getattr(request, "metadata", {})
                or {}
            ).get("expression")
            or ""
        )

        expression = (
            str(explicit_expression).strip()
            or self._extract_scalar_expression(
                combined_text
            )
            or ""
        )

        if not expression:
            return self._failed_result(
                attempt=attempt,
                reason="missing_calculator_input",
                output=(
                    "CalculatorTool could not execute because "
                    "the Task & Context Analyzer did not provide "
                    "a usable arithmetic expression or supported "
                    "convolution output-size parameters."
                ),
                operation=operation_type or "none",
            )

        try:
            result = self._safe_eval(expression)

        except Exception as exc:
            return self._failed_result(
                attempt=attempt,
                reason="calculation_failed",
                output=(
                    "CalculatorTool failed to evaluate "
                    f"the expression: {exc}"
                ),
                operation="arithmetic",
                extra_metadata={
                    "expression": expression,
                    "error": str(exc),
                },
            )

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=(
                "CalculatorTool computed the requested "
                f"expression exactly:\n\n"
                f"{expression} = {result}"
            ),
            evidence=[
                (
                    "The verified arithmetic result of "
                    f"{expression} is {result}."
                )
            ],
            references=[
                "CalculatorTool safe arithmetic evaluator"
            ],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "operation": "arithmetic",
                "operation_type": (
                    operation_type
                    or "scalar_arithmetic"
                ),
                "expression": expression,
                "result": result,
                "registry_tool": True,
            },
        )

    def _build_output_size_result(
        self,
        parameters: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        input_shape = self._normalize_shape(
            parameters.get("input_shape")
        )

        kernel_shape = self._normalize_shape(
            parameters.get("kernel_shape")
        )

        stride = self._positive_int(
            parameters.get("stride"),
            default=1,
        )

        padding = self._nonnegative_int(
            parameters.get("padding"),
            default=0,
        )

        if not input_shape or not kernel_shape:
            formula = (
                "O = floor((N + 2P - K) / S) + 1"
            )

            output = (
                "CalculatorTool verified the general "
                "convolution output-size rule.\n\n"
                f"Formula: {formula}\n\n"
                "Where:\n"
                "- N is the input size.\n"
                "- K is the kernel size.\n"
                "- P is the padding.\n"
                "- S is the stride.\n\n"
                "The formula is applied independently to "
                "the height and width. A concrete numerical "
                "output requires the input shape, kernel "
                "shape, stride, and padding."
            )

            return ToolResult(
                tool_name=self.name,
                status="success",
                success=True,
                output=output,
                evidence=[
                    (
                        "The verified convolution output-size "
                        f"formula is {formula}."
                    ),
                    (
                        "Output size depends on input size, "
                        "kernel size, stride, and padding."
                    ),
                ],
                references=[
                    (
                        "CalculatorTool deterministic "
                        "convolution output-size evaluator"
                    )
                ],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "operation": (
                        "convolution_output_size"
                    ),
                    "operation_type": (
                        "convolution_output_size"
                    ),
                    "formula": formula,
                    "formula_only": True,
                    "input_shape": input_shape,
                    "kernel_shape": kernel_shape,
                    "stride": stride,
                    "padding": padding,
                    "output_shape": None,
                    "registry_tool": True,
                },
            )

        input_rows, input_cols = input_shape
        kernel_rows, kernel_cols = kernel_shape

        numerator_rows = (
            input_rows
            + (2 * padding)
            - kernel_rows
        )

        numerator_cols = (
            input_cols
            + (2 * padding)
            - kernel_cols
        )

        if numerator_rows < 0 or numerator_cols < 0:
            return self._failed_result(
                attempt=attempt,
                reason="kernel_larger_than_padded_input",
                output=(
                    "The kernel is larger than the padded input, "
                    "so no valid output position exists."
                ),
                operation="convolution_output_size",
                extra_metadata={
                    "input_shape": input_shape,
                    "kernel_shape": kernel_shape,
                    "stride": stride,
                    "padding": padding,
                },
            )

        output_rows = (
            numerator_rows // stride
        ) + 1

        output_cols = (
            numerator_cols // stride
        ) + 1

        output_shape = [
            output_rows,
            output_cols,
        ]

        formula = (
            "O = floor((N + 2P - K) / S) + 1"
        )

        row_expression = (
            "floor(("
            f"{input_rows} + 2({padding}) - "
            f"{kernel_rows}) / {stride}) + 1"
        )

        col_expression = (
            "floor(("
            f"{input_cols} + 2({padding}) - "
            f"{kernel_cols}) / {stride}) + 1"
        )

        output = (
            "CalculatorTool computed the requested "
            "convolution output shape.\n\n"
            f"Formula: {formula}\n\n"
            f"Input shape: {input_rows}x{input_cols}\n"
            f"Kernel shape: {kernel_rows}x{kernel_cols}\n"
            f"Stride: {stride}\n"
            f"Padding: {padding}\n\n"
            f"Output rows = {row_expression} = "
            f"{output_rows}\n"
            f"Output columns = {col_expression} = "
            f"{output_cols}\n\n"
            "Therefore, the verified output shape is "
            f"{output_rows}x{output_cols}."
        )

        legacy_input_size: Any = (
            input_rows
            if input_rows == input_cols
            else input_shape
        )

        legacy_kernel_size: Any = (
            kernel_rows
            if kernel_rows == kernel_cols
            else kernel_shape
        )

        legacy_output_size: Any = (
            output_rows
            if output_rows == output_cols
            else output_shape
        )

        return ToolResult(
            tool_name=self.name,
            status="success",
            success=True,
            output=output,
            evidence=[
                (
                    "The convolution output-size formula is "
                    f"{formula}."
                ),
                (
                    "For input "
                    f"{input_rows}x{input_cols}, kernel "
                    f"{kernel_rows}x{kernel_cols}, stride "
                    f"{stride}, and padding {padding}, the "
                    "verified output shape is "
                    f"{output_rows}x{output_cols}."
                ),
            ],
            references=[
                (
                    "CalculatorTool deterministic convolution "
                    "output-size evaluator"
                )
            ],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "operation": "convolution_output_size",
                "operation_type": "convolution_output_size",
                "formula": formula,
                "input_shape": input_shape,
                "kernel_shape": kernel_shape,
                "stride": stride,
                "padding": padding,
                "output_shape": output_shape,
                # Retain legacy metadata names for current renderers.
                "input_size": legacy_input_size,
                "kernel_size": legacy_kernel_size,
                "output_size": legacy_output_size,
                "registry_tool": True,
            },
        )

    def _extract_convolution_parameters(
        self,
        text: str,
    ) -> Dict[str, Any]:
        normalized = str(text or "").lower()

        input_match = re.search(
            r"\binput(?:\s+(?:size|shape))?"
            r"\s*(?:=|is|of|:)?\s*"
            r"(\d+)\s*x\s*(\d+)",
            normalized,
        )

        kernel_match = re.search(
            r"\b(?:kernel|filter)"
            r"(?:\s+(?:size|shape))?"
            r"\s*(?:=|is|of|:)?\s*"
            r"(\d+)\s*x\s*(\d+)",
            normalized,
        )

        stride_match = re.search(
            r"\bstride\s*(?:=|of|is)?\s*(\d+)",
            normalized,
        )

        padding_match = re.search(
            r"\bpadding\s*(?:=|of|is)?\s*(\d+)",
            normalized,
        )

        result: Dict[str, Any] = {}

        if input_match:
            result["input_shape"] = [
                int(input_match.group(1)),
                int(input_match.group(2)),
            ]

        if kernel_match:
            result["kernel_shape"] = [
                int(kernel_match.group(1)),
                int(kernel_match.group(2)),
            ]

        result["stride"] = (
            int(stride_match.group(1))
            if stride_match
            else 1
        )

        result["padding"] = (
            int(padding_match.group(1))
            if padding_match
            else 0
        )

        return result

    def _extract_scalar_expression(
        self,
        text: str,
    ) -> Optional[str]:
        normalized = str(text or "").lower()

        squared_match = re.search(
            r"\b(-?\d+(?:\.\d+)?)\s+squared\s*"
            r"(plus|\+|minus|-)\s*"
            r"(-?\d+(?:\.\d+)?)\s+squared\b",
            normalized,
        )

        if squared_match:
            left, raw_operator, right = (
                squared_match.groups()
            )

            operator_symbol = (
                "+"
                if raw_operator in {"plus", "+"}
                else "-"
            )

            return (
                f"{left} ** 2 "
                f"{operator_symbol} "
                f"{right} ** 2"
            )

        binary_match = re.search(
            r"\b(-?\d+(?:\.\d+)?)\s*"
            r"(times|multiplied\s+by|\*|plus|\+|"
            r"minus|-|divided\s+by|/)\s*"
            r"(-?\d+(?:\.\d+)?)\b",
            normalized,
        )

        if binary_match:
            left, raw_operator, right = (
                binary_match.groups()
            )

            operator_map = {
                "times": "*",
                "multiplied by": "*",
                "*": "*",
                "plus": "+",
                "+": "+",
                "minus": "-",
                "-": "-",
                "divided by": "/",
                "/": "/",
            }

            key = re.sub(
                r"\s+",
                " ",
                raw_operator.strip(),
            )

            return (
                f"{left} "
                f"{operator_map[key]} "
                f"{right}"
            )

        matrix_context = any(
            term in normalized
            for term in (
                "matrix",
                "matrices",
                "kernel",
                "convolution",
                "shape",
                "dimension",
                "window",
            )
        )

        if not matrix_context:
            x_match = re.search(
                r"\b(-?\d+(?:\.\d+)?)\s*x\s*"
                r"(-?\d+(?:\.\d+)?)\b",
                normalized,
            )

            if x_match:
                return (
                    f"{x_match.group(1)} * "
                    f"{x_match.group(2)}"
                )

        return None

    def _combined_text(
        self,
        request: Any,
        step: Dict[str, Any],
        analyzed_task: Dict[str, Any],
    ) -> str:
        values = [
            getattr(request, "student_question", ""),
            getattr(request, "user_query", ""),
            getattr(request, "query", ""),
            getattr(request, "current_step", ""),
            getattr(request, "step_goal", ""),
            getattr(request, "expected_output", ""),
            step.get("goal", ""),
            step.get("purpose", ""),
            step.get("current_step", ""),
            analyzed_task.get("student_question", ""),
            analyzed_task.get("user_query", ""),
            analyzed_task.get("current_step", ""),
            analyzed_task.get("expected_output", ""),
            analyzed_task.get("goal", ""),
        ]

        return " ".join(
            str(value)
            for value in values
            if value
        ).lower()

    def _is_convolution_output_size_request(
        self,
        text: str,
    ) -> bool:
        return (
            any(
                term in text
                for term in (
                    "convolution",
                    "kernel",
                    "cnn",
                )
            )
            and any(
                phrase in text
                for phrase in (
                    "output size",
                    "output shape",
                    "output dimension",
                    "output dimensions",
                )
            )
        )

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
            cols = int(value[1])

        except (TypeError, ValueError):
            return None

        if rows <= 0 or cols <= 0:
            return None

        return [rows, cols]

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

    def _failed_result(
        self,
        attempt: int,
        reason: str,
        output: str,
        operation: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
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
                "CalculatorTool deterministic input policy"
            ],
            metadata=metadata,
        )

    def _safe_eval(self, expression: str):
        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
        ):
            return node.value

        if isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in _ALLOWED_OPERATORS:
                raise ValueError(
                    "Operator "
                    f"{operator_type.__name__} "
                    "is not allowed."
                )

            return _ALLOWED_OPERATORS[operator_type](
                self._eval_node(node.left),
                self._eval_node(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            operator_type = type(node.op)

            if operator_type not in _ALLOWED_OPERATORS:
                raise ValueError(
                    "Unary operator "
                    f"{operator_type.__name__} "
                    "is not allowed."
                )

            return _ALLOWED_OPERATORS[operator_type](
                self._eval_node(node.operand)
            )

        raise ValueError(
            "Unsupported expression node: "
            f"{type(node).__name__}"
        )
