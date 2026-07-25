"""Semantic validation for Tool Interaction Manager outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.agentic.core.schemas import ValidationReport


class OutputValidator:
    """
    Validate execution success and correspondence between the Analyzer's
    structured operation and the returned tool result.

    This component does not choose tools or execute them.
    """

    CODE_TASK_TYPES = {
        "code_execution",
        "code_help",
        "programming",
        "debugging",
    }

    MATRIX_OPERATIONS = {
        "matrix_compatibility",
        "matrix_multiplication",
        "matrix_entry",
        "dot_product",
        "valid_2d_convolution",
        "convolution_window_count",
        "kernel_fit",
    }

    def validate(
        self,
        request: Any,
        tool_results: List[Any],
        attempt_number: int,
        analyzed_task: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ValidationReport:
        analyzed_task = analyzed_task or {}

        if not tool_results:
            return self._invalid(
                confidence=0.1,
                reason="no_tool_results",
                action="replan_and_re_execute",
            )

        failed_results = [
            result
            for result in tool_results
            if not bool(
                getattr(result, "success", False)
            )
        ]

        if failed_results:
            return self._invalid(
                confidence=0.2,
                reason="tool_execution_failed",
                action="re_execute_relevant_tool",
            )

        operation_type = str(
            analyzed_task.get("operation_type")
            or ""
        )

        parameters = dict(
            analyzed_task.get(
                "operation_parameters"
            )
            or {}
        )

        text = self._request_text(
            request=request,
            analyzed_task=analyzed_task,
        )

        no_code_requested = any(
            phrase in text
            for phrase in (
                "without code",
                "no code",
                "avoid code",
                "do not use code",
                "don't use code",
                "code is confusing",
                "code confusing",
            )
        )

        if (
            no_code_requested
            and self._has_tool(
                tool_results,
                "CodeSandboxTool",
            )
        ):
            return self._invalid(
                confidence=0.2,
                reason=(
                    "explicit_no_code_constraint_violated"
                ),
                action="replan_without_code",
            )

        if self._requires_history_support(text):
            if not self._has_tool(
                tool_results,
                "TraceSearchTool",
            ):
                return self._invalid(
                    confidence=0.4,
                    reason=(
                        "missing_trace_history_support"
                    ),
                    action=(
                        "retry_with_trace_search_tool"
                    ),
                )

        if self._requires_grounding(
            analyzed_task=analyzed_task,
            text=text,
        ):
            rag_results = self._tool_results(
                tool_results,
                "RAGTool",
            )

            grounded = any(
                bool(
                    getattr(result, "evidence", [])
                )
                and bool(
                    getattr(result, "references", [])
                )
                and (
                    (
                        getattr(
                            result,
                            "metadata",
                            {},
                        )
                        or {}
                    ).get("topic_consistent")
                    is True
                )
                for result in rag_results
            )

            if not grounded:
                return self._invalid(
                    confidence=0.42,
                    reason="insufficient_grounding",
                    action="retry_with_grounding_tool",
                )

        if operation_type == "scalar_arithmetic":
            calculator = self._first_result(
                tool_results,
                "CalculatorTool",
            )

            if not calculator:
                return self._invalid(
                    confidence=0.25,
                    reason=(
                        "missing_verified_arithmetic_result"
                    ),
                    action=(
                        "retry_with_calculator_tool"
                    ),
                )

            metadata = self._metadata(calculator)

            expected_expression = parameters.get(
                "expression"
            )

            if (
                metadata.get("operation")
                != "arithmetic"
                or "result" not in metadata
                or not metadata.get("expression")
                or (
                    expected_expression
                    and metadata.get("expression")
                    != expected_expression
                )
            ):
                return self._invalid(
                    confidence=0.25,
                    reason=(
                        "arithmetic_result_mismatch"
                    ),
                    action=(
                        "re_execute_with_analyzed_expression"
                    ),
                )

        elif (
            operation_type
            == "convolution_output_size"
        ):
            calculator = self._first_result(
                tool_results,
                "CalculatorTool",
            )

            if not calculator:
                return self._invalid(
                    confidence=0.3,
                    reason=(
                        "missing_output_size_calculation"
                    ),
                    action=(
                        "retry_with_calculator_tool"
                    ),
                )

            metadata = self._metadata(calculator)

            if (
                metadata.get("operation")
                != "convolution_output_size"
                or not metadata.get("formula")
            ):
                return self._invalid(
                    confidence=0.3,
                    reason=(
                        "output_size_operation_mismatch"
                    ),
                    action=(
                        "re_execute_with_analyzed_parameters"
                    ),
                )

            expected_input = parameters.get(
                "input_shape"
            )

            expected_kernel = parameters.get(
                "kernel_shape"
            )

            if expected_input and expected_kernel:
                if (
                    metadata.get("input_shape")
                    != expected_input
                    or metadata.get("kernel_shape")
                    != expected_kernel
                    or int(
                        metadata.get("stride", 1)
                    )
                    != int(
                        parameters.get("stride", 1)
                    )
                    or int(
                        metadata.get("padding", 0)
                    )
                    != int(
                        parameters.get("padding", 0)
                    )
                    or not metadata.get(
                        "output_shape"
                    )
                ):
                    return self._invalid(
                        confidence=0.25,
                        reason=(
                            "output_size_parameter_mismatch"
                        ),
                        action=(
                            "re_execute_with_analyzed_parameters"
                        ),
                    )

        elif operation_type in self.MATRIX_OPERATIONS:
            matrix_result = self._first_result(
                tool_results,
                "MatrixComputationTool",
            )

            if not matrix_result:
                return self._invalid(
                    confidence=0.3,
                    reason=(
                        "missing_matrix_computation_support"
                    ),
                    action=(
                        "retry_with_matrix_computation_tool"
                    ),
                )

            metadata = self._metadata(
                matrix_result
            )

            if (
                metadata.get("operation")
                != operation_type
            ):
                return self._invalid(
                    confidence=0.25,
                    reason=(
                        "matrix_operation_mismatch"
                    ),
                    action=(
                        "re_execute_requested_matrix_operation"
                    ),
                )

            required_fields = {
                "matrix_compatibility": [
                    "compatible",
                ],
                "matrix_multiplication": [
                    "compatible",
                    "output_matrix",
                ],
                "matrix_entry": [
                    "entry_value",
                ],
                "dot_product": [
                    "result",
                ],
                "valid_2d_convolution": [
                    "output_matrix",
                    "top_left_value",
                ],
                "convolution_window_count": [
                    "window_count",
                ],
                "kernel_fit": [
                    "fits",
                    "window_count",
                ],
            }

            missing_fields = [
                field
                for field in required_fields.get(
                    operation_type,
                    [],
                )
                if field not in metadata
            ]

            if missing_fields:
                return self._invalid(
                    confidence=0.25,
                    reason=(
                        "incomplete_matrix_result_metadata"
                    ),
                    action=(
                        "re_execute_requested_matrix_operation"
                    ),
                )

        if self._requires_matrix_support(
            analyzed_task=analyzed_task,
            text=text,
        ):
            matrix_result = self._first_result(
                tool_results,
                "MatrixComputationTool",
            )

            if not matrix_result:
                return self._invalid(
                    confidence=0.4,
                    reason=(
                        "missing_matrix_computation_support"
                    ),
                    action=(
                        "retry_with_matrix_computation_tool"
                    ),
                )

            matrix_operation = self._metadata(
                matrix_result
            ).get("operation")

            if operation_type in {
                "matrix_concept",
                "convolution_concept",
            } and matrix_operation != operation_type:
                return self._invalid(
                    confidence=0.3,
                    reason=(
                        "matrix_concept_operation_mismatch"
                    ),
                    action=(
                        "re_execute_topic_consistent_matrix_support"
                    ),
                )

        if self._requires_code(
            analyzed_task=analyzed_task,
            text=text,
        ):
            sandbox = self._first_result(
                tool_results,
                "CodeSandboxTool",
            )

            if not sandbox:
                return self._invalid(
                    confidence=0.45,
                    reason=(
                        "missing_code_sandbox_execution"
                    ),
                    action=(
                        "retry_with_code_sandbox_tool"
                    ),
                )

        if self._requires_visual(
            analyzed_task=analyzed_task,
            text=text,
        ):
            visual = self._first_result(
                tool_results,
                "VisualMatrixTool",
            )

            if not visual:
                return self._invalid(
                    confidence=0.5,
                    reason="missing_visual_support",
                    action=(
                        "retry_with_visual_matrix_tool"
                    ),
                )

        return ValidationReport(
            status="valid",
            confidence_score=0.9,
            failure_reason=None,
            recommended_action="accept",
        )

    def _requires_grounding(
        self,
        analyzed_task: Dict[str, Any],
        text: str,
    ) -> bool:
        task_type = str(
            analyzed_task.get("task_type")
            or ""
        )

        operation_type = str(
            analyzed_task.get("operation_type")
            or ""
        )

        if task_type in self.CODE_TASK_TYPES:
            return any(
                phrase in text
                for phrase in (
                    "explain",
                    "explains",
                    "why",
                    "grounded",
                    "course",
                    "evidence",
                    "dimensions before",
                )
            )

        if self._requires_history_support(text):
            return True

        if self._requires_visual(
            analyzed_task,
            text,
        ):
            return True

        if operation_type in {
            "scalar_arithmetic",
            "convolution_output_size",
            *self.MATRIX_OPERATIONS,
        }:
            return False

        return operation_type in {
            "conceptual_explanation",
            "matrix_concept",
            "convolution_concept",
        } or task_type in {
            "conceptual_explanation",
            "personalized_support",
            "visual_explanation",
        }

    def _requires_history_support(
        self,
        text: str,
    ) -> bool:
        return any(
            phrase in text
            for phrase in (
                "previous",
                "earlier",
                "last time",
                "mistake",
                "again",
                "review",
                "personalized",
                "confusing",
                "confused",
                "my difficulty",
                "my difficulties",
            )
        )

    def _requires_code(
        self,
        analyzed_task: Dict[str, Any],
        text: str,
    ) -> bool:
        no_code = any(
            phrase in text
            for phrase in (
                "without code",
                "no code",
                "avoid code",
                "do not use code",
                "don't use code",
                "code is confusing",
                "code confusing",
            )
        )

        if no_code:
            return False

        return (
            str(
                analyzed_task.get("task_type")
                or ""
            )
            in self.CODE_TASK_TYPES
            or bool(
                analyzed_task.get(
                    "needs_code_execution",
                    False,
                )
            )
        )

    def _requires_visual(
        self,
        analyzed_task: Dict[str, Any],
        text: str,
    ) -> bool:
        return bool(
            analyzed_task.get(
                "explicit_visual_requested",
                False,
            )
        )


    def _requires_matrix_support(
        self,
        analyzed_task: Dict[str, Any],
        text: str,
    ) -> bool:
        operation_type = str(
            analyzed_task.get("operation_type")
            or ""
        )

        task_type = str(
            analyzed_task.get("task_type")
            or ""
        )

        if operation_type in self.MATRIX_OPERATIONS:
            return True

        visual = self._requires_visual(
            analyzed_task,
            text,
        )

        if task_type in self.CODE_TASK_TYPES:
            return (
                visual
                and operation_type == "matrix_concept"
                and any(
                    phrase in text
                    for phrase in (
                        "matrix multiplication",
                        "multiply matrices",
                        "row by column",
                        "row-by-column",
                    )
                )
            )

        if operation_type == "matrix_concept":
            return any(
                phrase in text
                for phrase in (
                    "matrix multiplication",
                    "multiply matrices",
                    "multiplying matrices",
                    "row by column",
                    "row-by-column",
                    "dot product",
                )
            )

        if operation_type == "convolution_concept":
            return visual

        return False

    def _request_text(
        self,
        request: Any,
        analyzed_task: Dict[str, Any],
    ) -> str:
        values = [
            getattr(
                request,
                "student_question",
                "",
            ),
            getattr(request, "user_query", ""),
            getattr(request, "query", ""),
            getattr(request, "current_step", ""),
            getattr(request, "step_goal", ""),
            getattr(request, "expected_output", ""),
            analyzed_task.get(
                "student_question",
                "",
            ),
            analyzed_task.get("user_query", ""),
            analyzed_task.get("current_step", ""),
            analyzed_task.get("expected_output", ""),
        ]

        return " ".join(
            str(value)
            for value in values
            if value
        ).lower()

    def _tool_results(
        self,
        results: List[Any],
        tool_name: str,
    ) -> List[Any]:
        return [
            result
            for result in results
            if getattr(
                result,
                "tool_name",
                None,
            ) == tool_name
        ]

    def _first_result(
        self,
        results: List[Any],
        tool_name: str,
    ) -> Optional[Any]:
        matching = self._tool_results(
            results,
            tool_name,
        )

        return matching[0] if matching else None

    def _has_tool(
        self,
        results: List[Any],
        tool_name: str,
    ) -> bool:
        return bool(
            self._tool_results(
                results,
                tool_name,
            )
        )

    def _metadata(
        self,
        result: Any,
    ) -> Dict[str, Any]:
        return dict(
            getattr(result, "metadata", {})
            or {}
        )

    def _invalid(
        self,
        confidence: float,
        reason: str,
        action: str,
    ) -> ValidationReport:
        return ValidationReport(
            status="invalid",
            confidence_score=confidence,
            failure_reason=reason,
            recommended_action=action,
        )
