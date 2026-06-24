from ai.agentic.core.schemas import ValidationReport


class OutputValidator:
    """
    Validates tool outputs.

    Rules:
    - all selected tools should execute successfully
    - output must be grounded by RAGTool
    - visual explanations must include VisualMatrixTool
    - convolution visual explanations should include MatrixComputationTool
    """

    def validate(self, request, tool_results, attempt_number: int):
        execution_ok = all(result.success for result in tool_results)

        if not execution_ok:
            return ValidationReport(
                status="invalid",
                confidence_score=0.2,
                failure_reason="tool_execution_failed",
                recommended_action="re_execute",
            )

        rag_results = [
            result for result in tool_results
            if result.tool_name in {"RAGTool", "mock_rag_tool"}
        ]

        rag_has_evidence = any(result.evidence for result in rag_results)
        rag_has_references = any(result.references for result in rag_results)

        if not rag_has_evidence or not rag_has_references:
            return ValidationReport(
                status="invalid",
                confidence_score=0.42,
                failure_reason="insufficient_grounding",
                recommended_action="retry_with_grounding_tools",
            )

        if request.task_type == "visual_explanation":
            visual_results = [
                result for result in tool_results
                if result.tool_name in {"VisualMatrixTool", "mock_visualizer_tool"}
            ]

            if not visual_results:
                return ValidationReport(
                    status="invalid",
                    confidence_score=0.5,
                    failure_reason="missing_visual_support",
                    recommended_action="re_select_or_re_execute",
                )

            question = getattr(request, "student_question", "").lower()
            if "convolution" in question:
                matrix_results = [
                    result for result in tool_results
                    if result.tool_name == "MatrixComputationTool"
                ]

                if not matrix_results:
                    return ValidationReport(
                        status="invalid",
                        confidence_score=0.55,
                        failure_reason="missing_matrix_computation_support",
                        recommended_action="re_select_or_re_execute",
                    )

        return ValidationReport(
            status="valid",
            confidence_score=0.9,
            failure_reason=None,
            recommended_action="accept",
        )
