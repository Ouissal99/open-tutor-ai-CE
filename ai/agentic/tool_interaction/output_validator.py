from ai.agentic.core.schemas import ValidationReport


class OutputValidator:
    """
    Validates tool outputs.

    Important:
    Validation must be step-aware. A global student question may ask for code,
    but an individual scratchpad step may only retrieve grounding evidence.
    CodeSandboxTool is required only when the current step itself is code-related.
    """

    CODE_KEYWORDS = {
        "code",
        "python",
        "numpy",
        "debug",
        "implement",
        "function",
        "script",
        "program",
        "snippet",
        "write a simple python",
    }

    RETRIEVAL_KEYWORDS = {
        "retrieve",
        "retrieval",
        "course-grounded evidence",
        "grounded evidence",
        "grounding",
        "rag",
        "search evidence",
        "collect evidence",
        "course evidence",
    }

    VISUAL_KEYWORDS = {
        "visual",
        "show",
        "illustrate",
        "visualize",
        "matrix",
        "kernel movement",
        "sliding",
    }

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

        if self._is_code_task(request):
            sandbox_results = [
                result for result in tool_results
                if result.tool_name == "CodeSandboxTool"
            ]

            if not sandbox_results:
                return ValidationReport(
                    status="invalid",
                    confidence_score=0.45,
                    failure_reason="missing_code_sandbox_execution",
                    recommended_action="retry_with_code_sandbox_tool",
                )

            if not any(result.success for result in sandbox_results):
                return ValidationReport(
                    status="invalid",
                    confidence_score=0.35,
                    failure_reason="code_execution_failed",
                    recommended_action="retry_with_corrected_code",
                )

        if self._is_visual_task(request):
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

            if "convolution" in self._current_step_text(request):
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

    def _current_step_text(self, request) -> str:
        return " ".join(
            str(getattr(request, attr, "") or "")
            for attr in ["current_step", "expected_output"]
        ).lower()

    def _global_text(self, request) -> str:
        return " ".join(
            str(getattr(request, attr, "") or "")
            for attr in ["student_question", "user_query", "query"]
        ).lower()

    def _is_retrieval_only_step(self, text: str) -> bool:
        has_retrieval = any(keyword in text for keyword in self.RETRIEVAL_KEYWORDS)
        has_code = any(keyword in text for keyword in self.CODE_KEYWORDS)
        has_visual = any(keyword in text for keyword in self.VISUAL_KEYWORDS)

        return has_retrieval and not has_code and not has_visual

    def _is_code_task(self, request) -> bool:
        task_type = getattr(request, "task_type", "") or ""
        step_text = self._current_step_text(request)
        global_text = self._global_text(request)

        if self._is_retrieval_only_step(step_text):
            return False

        if task_type in {"code_execution", "code_help", "programming", "debugging"}:
            return True

        if any(keyword in step_text for keyword in self.CODE_KEYWORDS):
            return True

        # Only use the global question when there is no concrete scratchpad step.
        if not step_text.strip() and any(keyword in global_text for keyword in self.CODE_KEYWORDS):
            return True

        return False

    def _is_visual_task(self, request) -> bool:
        task_type = getattr(request, "task_type", "") or ""
        step_text = self._current_step_text(request)

        if self._is_code_task(request):
            return False

        if self._is_retrieval_only_step(step_text):
            return False

        if task_type == "visual_explanation":
            return True

        return any(keyword in step_text for keyword in self.VISUAL_KEYWORDS)
