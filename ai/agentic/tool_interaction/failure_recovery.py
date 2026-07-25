"""Bounded and failure-specific recovery policy."""


class FailureRecovery:
    """
    Translate validation failures into the minimum relevant tools for the
    next bounded attempt.
    """

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def decide(
        self,
        validation_report,
        attempt_number: int,
        analyzed_task=None,
    ):
        analyzed_task = analyzed_task or {}

        failure_reason = (
            validation_report.failure_reason
            or "unknown_failure"
        )

        if attempt_number >= self.max_attempts:
            return {
                "should_retry": False,
                "action": "fallback",
                "reason": "maximum_attempts_reached",
                "failure_reason": failure_reason,
                "required_tools": [],
            }

        operation_type = str(
            analyzed_task.get("operation_type")
            or ""
        )

        task_type = str(
            analyzed_task.get("task_type")
            or ""
        )

        text = " ".join(
            str(
                analyzed_task.get(key, "")
                or ""
            )
            for key in (
                "student_question",
                "user_query",
                "current_step",
                "expected_output",
            )
        ).lower()

        required_tools = []

        def add(tool_name):
            if tool_name not in required_tools:
                required_tools.append(tool_name)

        calculator_failures = {
            "missing_verified_arithmetic_result",
            "arithmetic_result_mismatch",
            "missing_output_size_calculation",
            "output_size_operation_mismatch",
            "output_size_parameter_mismatch",
        }

        matrix_failures = {
            "missing_matrix_computation_support",
            "matrix_operation_mismatch",
            "incomplete_matrix_result_metadata",
            "matrix_concept_operation_mismatch",
        }

        if failure_reason in calculator_failures:
            add("CalculatorTool")
            action = "retry_calculation"

        elif failure_reason in matrix_failures:
            add("MatrixComputationTool")
            action = "retry_matrix_computation"

        elif failure_reason in {
            "insufficient_grounding",
            "weak_grounding",
        }:
            add("RAGTool")
            action = "retry_grounding"

        elif (
            failure_reason
            == "missing_trace_history_support"
        ):
            add("TraceSearchTool")
            action = "retry_trace_retrieval"

        elif failure_reason == "missing_visual_support":
            add("VisualMatrixTool")
            action = "retry_visual_support"

        elif failure_reason in {
            "missing_code_sandbox_execution",
            "code_execution_failed",
            "code_drafting_failed",
        }:
            add("CodeSandboxTool")
            action = "retry_code_execution"

        elif failure_reason in {
            "tool_execution_failed",
            "no_tool_results",
        }:
            action = "retry_primary_operation"

            if operation_type in {
                "scalar_arithmetic",
                "convolution_output_size",
            }:
                add("CalculatorTool")

            elif operation_type in {
                "matrix_compatibility",
                "matrix_multiplication",
                "matrix_entry",
                "dot_product",
                "valid_2d_convolution",
                "convolution_window_count",
                "kernel_fit",
            }:
                add("MatrixComputationTool")

            elif task_type in {
                "code_execution",
                "code_help",
                "programming",
                "debugging",
            }:
                add("CodeSandboxTool")

            else:
                add("RAGTool")

        else:
            action = "replan_relevant_support"

            if operation_type in {
                "scalar_arithmetic",
                "convolution_output_size",
            }:
                add("CalculatorTool")

            elif operation_type in {
                "matrix_compatibility",
                "matrix_multiplication",
                "matrix_entry",
                "dot_product",
                "valid_2d_convolution",
                "convolution_window_count",
                "kernel_fit",
            }:
                add("MatrixComputationTool")

            else:
                add("RAGTool")

        # Preserve the primary execution capability while adding
        # the support requested by validation. Recovery augments the
        # original operation; it does not replace it.
        if operation_type in {
            "scalar_arithmetic",
            "convolution_output_size",
        }:
            add("CalculatorTool")

        elif operation_type in {
            "matrix_compatibility",
            "matrix_multiplication",
            "matrix_entry",
            "dot_product",
            "valid_2d_convolution",
            "convolution_window_count",
            "kernel_fit",
        }:
            add("MatrixComputationTool")

        history_required = any(
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
            )
        )

        if history_required:
            add("TraceSearchTool")

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

        visual_required = bool(
            analyzed_task.get(
                "explicit_visual_requested",
                False,
            )
        )

        code_task = task_type in {
            "code_execution",
            "code_help",
            "programming",
            "debugging",
        }

        if code_task and not no_code:
            add("CodeSandboxTool")

        grounding_required = False

        if code_task:
            grounding_required = any(
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

        elif history_required or visual_required:
            grounding_required = True

        elif operation_type in {
            "conceptual_explanation",
            "matrix_concept",
            "convolution_concept",
        }:
            grounding_required = True

        if grounding_required:
            add("RAGTool")

        if visual_required:
            add("VisualMatrixTool")

            if (
                operation_type
                in {
                    "matrix_concept",
                    "convolution_concept",
                }
                and not code_task
            ):
                add("MatrixComputationTool")

            if (
                code_task
                and operation_type
                == "matrix_concept"
                and "matrix multiplication" in text
            ):
                add("MatrixComputationTool")

        if (
            operation_type == "matrix_concept"
            and not code_task
            and any(
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
        ):
            add("MatrixComputationTool")

        if no_code:
            required_tools = [
                tool_name
                for tool_name in required_tools
                if tool_name != "CodeSandboxTool"
            ]

        return {
            "should_retry": True,
            "action": action,
            "reason": failure_reason,
            "failure_reason": failure_reason,
            "required_tools": required_tools,
        }
