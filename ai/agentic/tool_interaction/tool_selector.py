from typing import Any, Dict, List

from ai.agentic.memory.trace_toolkit import TraceToolkit


class ToolSelector:
    """
    Memory-aware tool selector.

    It can reuse previous successful traces, normalize old mock tool names,
    and respect FailureRecovery-required tools.
    """

    TOOL_NAME_ALIASES = {
        "mock_rag_tool": "RAGTool",
        "mock_visualizer_tool": "VisualMatrixTool",
        "mock_validator_support": "TraceSearchTool",
        "mock_matrix_tool": "MatrixComputationTool",
        "mock_calculator_tool": "CalculatorTool",
        "mock_trace_tool": "TraceSearchTool",
    }

    def __init__(self):
        self.trace_toolkit = TraceToolkit()
        self.last_selection_metadata: Dict[str, Any] = {}

    def select_tools(self, analyzed_task: Dict[str, Any], attempt: int = 1) -> List[str]:
        task_type = analyzed_task.get("task_type", "unknown_task")
        query = self._build_query_text(analyzed_task)

        if attempt == 1 and analyzed_task.get("force_recovery_test"):
            selected_tools = ["TraceSearchTool"]

            self.last_selection_metadata = {
                "selection_strategy": "controlled_failure_test",
                "selection_reason": "force_recovery_test_attempt_1_omits_grounding_tools",
                "reference_trace_id": None,
                "reference_trace_path": None,
                "similarity_score": 0,
                "failed_trace_count_considered": 0,
                "selected_tools": selected_tools,
                "attempt": attempt,
            }

            return selected_tools

        recovery_required_tools = analyzed_task.get("recovery_required_tools") or []
        if attempt > 1 and recovery_required_tools:
            selected_tools = self._normalize_selected_tools(
                selected_tools=recovery_required_tools,
                task_type=task_type,
                failed_traces=[],
                analyzed_task=analyzed_task,
            )

            self.last_selection_metadata = {
                "selection_strategy": "failure_recovery_policy",
                "selection_reason": "using_tools_required_by_failure_recovery",
                "reference_trace_id": None,
                "reference_trace_path": None,
                "similarity_score": 0,
                "failed_trace_count_considered": 0,
                "selected_tools": selected_tools,
                "attempt": attempt,
                "recovery_required_tools": recovery_required_tools,
            }

            return selected_tools

        similar_successful_traces = self.trace_toolkit.find_similar_traces(
            query=query,
            task_type=task_type,
            limit=3,
            only_successful=True,
        )

        failed_traces = self.trace_toolkit.find_failed_traces(
            task_type=task_type,
            limit=3,
        )

        if similar_successful_traces:
            best_trace = similar_successful_traces[0]
            selected_tools = best_trace.get("selected_tools", [])

            selected_tools = self._normalize_selected_tools(
                selected_tools=selected_tools,
                task_type=task_type,
                failed_traces=failed_traces,
                analyzed_task=analyzed_task,
            )

            self.last_selection_metadata = {
                "selection_strategy": "trace_guided_reuse",
                "selection_reason": "similar_successful_trace_found",
                "reference_trace_id": best_trace.get("trace_id"),
                "reference_trace_path": best_trace.get("trace_path"),
                "similarity_score": best_trace.get("similarity_score"),
                "failed_trace_count_considered": len(failed_traces),
                "selected_tools": selected_tools,
                "attempt": attempt,
            }

            return selected_tools

        selected_tools = self._default_policy(task_type, analyzed_task)

        selected_tools = self._normalize_selected_tools(
            selected_tools=selected_tools,
            task_type=task_type,
            failed_traces=failed_traces,
            analyzed_task=analyzed_task,
        )

        self.last_selection_metadata = {
            "selection_strategy": "default_policy",
            "selection_reason": "no_similar_successful_trace_found",
            "reference_trace_id": None,
            "reference_trace_path": None,
            "similarity_score": 0,
            "failed_trace_count_considered": len(failed_traces),
            "selected_tools": selected_tools,
            "attempt": attempt,
        }

        return selected_tools

    def select(self, analyzed_task: Dict[str, Any], attempt: int = 1) -> List[str]:
        return self.select_tools(analyzed_task, attempt=attempt)

    def get_last_selection_metadata(self) -> Dict[str, Any]:
        return self.last_selection_metadata

    def _build_query_text(self, analyzed_task: Dict[str, Any]) -> str:
        parts = [
            analyzed_task.get("topic", ""),
            analyzed_task.get("task_type", ""),
            analyzed_task.get("current_step", ""),
            analyzed_task.get("student_question", ""),
            analyzed_task.get("user_query", ""),
            analyzed_task.get("query", ""),
        ]

        context = analyzed_task.get("context", {})
        if isinstance(context, dict):
            parts.extend(str(value) for value in context.values())

        investigation = analyzed_task.get("investigation_result", {})
        if isinstance(investigation, dict):
            parts.append(str(investigation.get("tool_request_goal", "")))
            parts.append(str(investigation.get("initial_tutoring_plan", "")))

        return " ".join(str(part) for part in parts if part)

    def _default_policy(self, task_type: str, analyzed_task: Dict[str, Any]) -> List[str]:
        if task_type == "visual_explanation":
            return [
                "RAGTool",
                "TraceSearchTool",
                "MatrixComputationTool",
                "VisualMatrixTool",
            ]

        if task_type in {"code_help", "programming", "code_execution", "debugging"}:
            return [
                "RAGTool",
                "TraceSearchTool",
                "MatrixComputationTool",
                "CodeSandboxTool",
            ]

        if task_type == "calculation_or_verification":
            return [
                "RAGTool",
                "TraceSearchTool",
                "CalculatorTool",
            ]

        return [
            "RAGTool",
            "TraceSearchTool",
        ]

    def _normalize_selected_tools(
        self,
        selected_tools: List[str],
        task_type: str,
        failed_traces: List[Dict[str, Any]],
        analyzed_task: Dict[str, Any],
    ) -> List[str]:
        normalized = []

        for tool in selected_tools:
            if not isinstance(tool, str):
                continue

            mapped_tool = self.TOOL_NAME_ALIASES.get(tool, tool)

            if mapped_tool not in normalized:
                normalized.append(mapped_tool)

        if not normalized:
            normalized = self._default_policy(task_type, analyzed_task)

        if "RAGTool" not in normalized:
            normalized.insert(0, "RAGTool")

        if "TraceSearchTool" not in normalized:
            normalized.append("TraceSearchTool")

        if task_type == "visual_explanation":
            if "MatrixComputationTool" not in normalized:
                normalized.append("MatrixComputationTool")
            if "VisualMatrixTool" not in normalized:
                normalized.append("VisualMatrixTool")

        if task_type == "calculation_or_verification" and "CalculatorTool" not in normalized:
            normalized.append("CalculatorTool")

        if task_type in {"code_help", "programming", "code_execution"} and "CodeSandboxTool" not in normalized:
            normalized.append("CodeSandboxTool")

        return normalized
