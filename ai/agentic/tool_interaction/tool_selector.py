from typing import Any, Dict, List

from ai.agentic.memory.trace_toolkit import TraceToolkit


class ToolSelector:
    """
    Memory-aware tool selector.

    Instead of selecting tools only from fixed rules, this selector first checks
    previous successful tool-interaction traces.

    Selection strategy:
    1. Read task_type and query/topic from analyzed_task.
    2. Ask TraceToolkit for similar successful traces.
    3. If a successful similar trace exists, reuse its selected tools.
    4. If no useful trace exists, fall back to default policy.
    5. Store selection metadata for trace/debug/evaluation.
    """

    def __init__(self):
        self.trace_toolkit = TraceToolkit()
        self.last_selection_metadata: Dict[str, Any] = {}

    def select_tools(self, analyzed_task: Dict[str, Any], attempt: int = 1) -> List[str]:
        task_type = analyzed_task.get("task_type", "unknown_task")
        query = self._build_query_text(analyzed_task)

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

        selected_tools = self._default_policy(task_type)

        selected_tools = self._normalize_selected_tools(
            selected_tools=selected_tools,
            task_type=task_type,
            failed_traces=failed_traces,
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
        """
        Backward-compatible alias in case old manager code calls selector.select().
        """
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

        return " ".join(str(part) for part in parts if part)

    def _default_policy(self, task_type: str) -> List[str]:
        if task_type == "visual_explanation":
            return [
                "mock_rag_tool",
                "mock_visualizer_tool",
                "mock_validator_support",
            ]

        if task_type == "calculation_or_verification":
            return [
                "mock_rag_tool",
                "mock_validator_support",
            ]

        return [
            "mock_rag_tool",
            "mock_validator_support",
        ]

    def _normalize_selected_tools(
        self,
        selected_tools: List[str],
        task_type: str,
        failed_traces: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Ensures selected tools are valid and complete.

        This avoids returning empty or incomplete tool lists when old traces
        are partially structured.
        """
        normalized = []

        for tool in selected_tools:
            if isinstance(tool, str) and tool not in normalized:
                normalized.append(tool)

        if not normalized:
            normalized = self._default_policy(task_type)

        if task_type == "visual_explanation":
            if "mock_rag_tool" not in normalized:
                normalized.insert(0, "mock_rag_tool")

            if "mock_visualizer_tool" not in normalized:
                normalized.append("mock_visualizer_tool")

        if "mock_validator_support" not in normalized:
            normalized.append("mock_validator_support")

        if failed_traces:
            # If previous failures exist for this task type, keep validator support.
            if "mock_validator_support" not in normalized:
                normalized.append("mock_validator_support")

        return normalized
