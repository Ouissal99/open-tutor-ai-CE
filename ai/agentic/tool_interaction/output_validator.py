from ai.agentic.core.schemas import ValidationReport


class OutputValidator:
    """
    Validates tool outputs.

    For this demo, the important rule is:
    the final output must be grounded by the RAG tool.
    Visualizer output alone is not enough.
    """

    def validate(self, request, tool_results, attempt_number: int):
        execution_ok = all(result.success for result in tool_results)

        if not execution_ok:
            return ValidationReport(
                status="invalid",
                confidence_score=0.2,
                failure_reason="tool_execution_failed",
                recommended_action="re_execute"
            )

        # Require grounding from the RAG tool specifically.
        rag_results = [
            result for result in tool_results
            if result.tool_name == "mock_rag_tool"
        ]

        rag_has_evidence = any(result.evidence for result in rag_results)
        rag_has_references = any(result.references for result in rag_results)

        if not rag_has_evidence or not rag_has_references:
            return ValidationReport(
                status="invalid",
                confidence_score=0.42,
                failure_reason="weak_grounding",
                recommended_action="re_query_or_re_execute"
            )

        # Check that the expected visual support exists when requested.
        if request.task_type == "visual_explanation":
            visual_results = [
                result for result in tool_results
                if result.tool_name == "mock_visualizer_tool"
            ]

            if not visual_results:
                return ValidationReport(
                    status="invalid",
                    confidence_score=0.5,
                    failure_reason="missing_visual_support",
                    recommended_action="re_select_or_re_execute"
                )

        return ValidationReport(
            status="valid",
            confidence_score=0.88,
            failure_reason=None,
            recommended_action="accept"
        )
