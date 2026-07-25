from typing import Any, Dict, List


class ToolSelector:
    """
    Adaptive Tool Selector.

    Important:
    - It does NOT retrieve DPM, SKG, RAG, or previous traces.
    - ContextCollector already retrieved context.
    - ToolPlanner already created the ordered support plan.
    - This selector maps planned capabilities to concrete tools.
    """

    TOOL_NAME_ALIASES = {
        "mock_rag_tool": "RAGTool",
        "mock_visualizer_tool": "VisualMatrixTool",
        "mock_validator_support": "TraceSearchTool",
        "mock_matrix_tool": "MatrixComputationTool",
        "mock_calculator_tool": "CalculatorTool",
        "mock_trace_tool": "TraceSearchTool",
    }

    AVAILABLE_TOOLS = {
        "RAGTool",
        "TraceSearchTool",
        "MatrixComputationTool",
        "VisualMatrixTool",
        "CalculatorTool",
        "CodeSandboxTool",
    }

    CAPABILITY_TO_TOOL = {
        "grounding": "RAGTool",
        "trace_reuse": "TraceSearchTool",
        "matrix_computation": "MatrixComputationTool",
        "visualization": "VisualMatrixTool",
        "calculation": "CalculatorTool",
        "code_execution": "CodeSandboxTool",
        "general_tool_support": "RAGTool",
    }

    TOOL_TO_CAPABILITY = {
        "RAGTool": "grounding",
        "TraceSearchTool": "trace_reuse",
        "MatrixComputationTool": "matrix_computation",
        "VisualMatrixTool": "visualization",
        "CalculatorTool": "calculation",
        "CodeSandboxTool": "code_execution",
    }

    def __init__(self):
        self.last_selection_metadata: Dict[str, Any] = {}

    def select_tools(
        self,
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
        plan: Any = None,
    ) -> List[str]:
        task_type = analyzed_task.get("task_type", "unknown_task")

        # Controlled failure test: keep this behavior for recovery experiments.
        if attempt == 1 and analyzed_task.get("force_recovery_test"):
            selected_tools = ["TraceSearchTool"]

            self.last_selection_metadata = {
                "selection_strategy": "controlled_failure_test",
                "selection_reason": "force_recovery_test_attempt_1_omits_required_support",
                "reference_trace_id": None,
                "reference_trace_path": None,
                "similarity_score": 0,
                "failed_trace_count_considered": 0,
                "selected_tools": selected_tools,
                "attempt": attempt,
            }

            return selected_tools

        # Recovery has priority because validation already detected missing support.
        recovery_required_tools = analyzed_task.get("recovery_required_tools") or []
        if attempt > 1 and recovery_required_tools:
            selected_tools = self._normalize_selected_tools(recovery_required_tools)

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

        # Main article-version behavior:
        # select concrete tools from the plan produced by ToolPlanner.
        plan_required_tools = self._tools_from_plan(plan)
        if plan_required_tools:
            selected_tools = self._normalize_selected_tools(plan_required_tools)

            self.last_selection_metadata = {
                "selection_strategy": "plan_guided_selection",
                "selection_reason": "selected_tools_mapped_from_tool_use_plan",
                "reference_trace_id": None,
                "reference_trace_path": None,
                "similarity_score": 0,
                "failed_trace_count_considered": self._failed_trace_count(analyzed_task),
                "selected_tools": selected_tools,
                "attempt": attempt,
                "plan_capabilities": self._capabilities_from_plan(plan),
            }

            return selected_tools

        # Fallback only: reuse trace-derived tools if the plan is missing or empty.
        trace_tools = self._tools_from_collected_trace_context(analyzed_task)
        if trace_tools:
            selected_tools = self._normalize_selected_tools(trace_tools)

            self.last_selection_metadata = {
                "selection_strategy": "trace_context_fallback",
                "selection_reason": "plan_empty_reused_tools_from_collected_context",
                "reference_trace_id": self._best_trace_id(analyzed_task),
                "reference_trace_path": self._best_trace_path(analyzed_task),
                "similarity_score": self._best_trace_similarity(analyzed_task),
                "failed_trace_count_considered": self._failed_trace_count(analyzed_task),
                "selected_tools": selected_tools,
                "attempt": attempt,
            }

            return selected_tools

        # Final fallback: deterministic default policy.
        selected_tools = self._normalize_selected_tools(
            self._default_policy(task_type, analyzed_task)
        )

        self.last_selection_metadata = {
            "selection_strategy": "default_policy",
            "selection_reason": "no_plan_and_no_trace_context_available",
            "reference_trace_id": None,
            "reference_trace_path": None,
            "similarity_score": 0,
            "failed_trace_count_considered": self._failed_trace_count(analyzed_task),
            "selected_tools": selected_tools,
            "attempt": attempt,
        }

        return selected_tools

    def select(
        self,
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
        plan: Any = None,
    ) -> List[str]:
        return self.select_tools(analyzed_task=analyzed_task, attempt=attempt, plan=plan)

    def get_last_selection_metadata(self) -> Dict[str, Any]:
        return self.last_selection_metadata

    def _tools_from_plan(self, plan: Any) -> List[str]:
        if plan is None:
            return []

        steps = getattr(plan, "steps", []) or []
        selected_tools: List[str] = []

        for step in steps:
            if not isinstance(step, dict):
                continue

            explicit_tool = step.get("tool_name")
            if explicit_tool:
                mapped_tool = self.TOOL_NAME_ALIASES.get(explicit_tool, explicit_tool)
                if mapped_tool in self.AVAILABLE_TOOLS and mapped_tool not in selected_tools:
                    selected_tools.append(mapped_tool)
                continue

            capability = step.get("required_capability")
            tool_name = self.CAPABILITY_TO_TOOL.get(capability)

            if tool_name and tool_name in self.AVAILABLE_TOOLS and tool_name not in selected_tools:
                selected_tools.append(tool_name)

        return selected_tools

    def _capabilities_from_plan(self, plan: Any) -> List[str]:
        if plan is None:
            return []

        steps = getattr(plan, "steps", []) or []
        capabilities: List[str] = []

        for step in steps:
            if not isinstance(step, dict):
                continue

            capability = step.get("required_capability")
            if capability and capability not in capabilities:
                capabilities.append(capability)

        return capabilities

    def _tools_from_collected_trace_context(self, analyzed_task: Dict[str, Any]) -> List[str]:
        collected_context = analyzed_task.get("collected_context", {}) or {}

        traces = (
            collected_context.get("similar_successful_traces")
            or collected_context.get("similar_traces")
            or analyzed_task.get("similar_successful_traces")
            or []
        )

        if not traces:
            return []

        best_trace = traces[0]
        if not isinstance(best_trace, dict):
            return []

        return best_trace.get("selected_tools", []) or best_trace.get("tools", []) or []

    def _best_trace_id(self, analyzed_task: Dict[str, Any]) -> Any:
        best_trace = self._best_trace(analyzed_task)
        return best_trace.get("trace_id") if best_trace else None

    def _best_trace_path(self, analyzed_task: Dict[str, Any]) -> Any:
        best_trace = self._best_trace(analyzed_task)
        return best_trace.get("trace_path") if best_trace else None

    def _best_trace_similarity(self, analyzed_task: Dict[str, Any]) -> Any:
        best_trace = self._best_trace(analyzed_task)
        return best_trace.get("similarity_score") if best_trace else 0

    def _best_trace(self, analyzed_task: Dict[str, Any]) -> Dict[str, Any]:
        collected_context = analyzed_task.get("collected_context", {}) or {}
        traces = (
            collected_context.get("similar_successful_traces")
            or collected_context.get("similar_traces")
            or analyzed_task.get("similar_successful_traces")
            or []
        )

        if traces and isinstance(traces[0], dict):
            return traces[0]

        return {}

    def _failed_trace_count(self, analyzed_task: Dict[str, Any]) -> int:
        collected_context = analyzed_task.get("collected_context", {}) or {}
        summary = collected_context.get("summary", {}) or {}

        if "failed_trace_count" in summary:
            return int(summary.get("failed_trace_count") or 0)

        failed_traces = (
            collected_context.get("failed_traces")
            or analyzed_task.get("failed_traces")
            or []
        )

        return len(failed_traces)

    def _default_policy(self, task_type: str, analyzed_task: Dict[str, Any]) -> List[str]:
        text = self._combined_text(analyzed_task)

        if task_type == "visual_explanation":
            if self._mentions_matrix_or_convolution(text):
                return ["RAGTool", "MatrixComputationTool", "VisualMatrixTool"]
            return ["RAGTool", "VisualMatrixTool"]

        if task_type in {"code_help", "programming", "code_execution", "debugging"}:
            return ["RAGTool", "CodeSandboxTool"]

        if task_type in {"calculation_or_verification", "calculation", "math_verification"}:
            return ["RAGTool", "CalculatorTool"]

        if self._mentions_matrix_or_convolution(text):
            return ["RAGTool", "MatrixComputationTool", "VisualMatrixTool"]

        return ["RAGTool"]

    def _normalize_selected_tools(self, selected_tools: List[str]) -> List[str]:
        normalized: List[str] = []

        for tool in selected_tools:
            if not isinstance(tool, str):
                continue

            mapped_tool = self.TOOL_NAME_ALIASES.get(tool, tool)

            if mapped_tool not in self.AVAILABLE_TOOLS:
                continue

            if mapped_tool not in normalized:
                normalized.append(mapped_tool)

        return normalized

    def _combined_text(self, analyzed_task: Dict[str, Any]) -> str:
        parts = [
            analyzed_task.get("topic", ""),
            analyzed_task.get("task_type", ""),
            analyzed_task.get("current_step", ""),
            analyzed_task.get("student_question", ""),
            analyzed_task.get("user_query", ""),
            analyzed_task.get("query", ""),
            analyzed_task.get("expected_output", ""),
        ]

        investigation = analyzed_task.get("investigation_result", {})
        if isinstance(investigation, dict):
            parts.append(str(investigation.get("tool_request_goal", "")))
            parts.append(str(investigation.get("initial_tutoring_plan", "")))

        return " ".join(str(part) for part in parts if part).lower()

    def _mentions_matrix_or_convolution(self, text: str) -> bool:
        keywords = [
            "matrix",
            "matrice",
            "matrices",
            "multiplication",
            "convolution",
            "kernel",
            "linear algebra",
        ]
        return any(keyword in text for keyword in keywords)