from typing import Any, Dict, List, Optional

from ai.agentic.core.schemas import ToolPlan, new_id


class ToolPlanner:
    """
    Creates an ordered tool-use plan for the centralized Tool Interaction Manager.

    Important:
    - This planner does NOT retrieve DPM, SKG, RAG, or previous traces.
    - Context retrieval is already done by ContextCollector.
    - This planner only reads the analyzed task and the collected_context already present in state.
    - It produces ordered required capabilities.
    - The Adaptive Tool Selector later maps those capabilities to concrete tools.
    """

    def create_plan(
        self,
        analyzed_task: Dict[str, Any],
        collected_context: Optional[Dict[str, Any]] = None,
    ) -> ToolPlan:
        collected_context = collected_context or {}

        task_type = analyzed_task.get("task_type", "unknown_task")
        topic = analyzed_task.get("topic", "general topic")
        current_step = analyzed_task.get("current_step", "unknown step")
        expected_output = analyzed_task.get("expected_output")
        student_question = analyzed_task.get("student_question", "")

        context_summary = collected_context.get("summary", {}) or {}

        learner_level = context_summary.get("learner_level")
        is_weak_topic = bool(context_summary.get("is_weak_topic", False))

        recovery_required_tools = analyzed_task.get("recovery_required_tools") or []
        recovery_reason = analyzed_task.get("recovery_reason")

        steps: List[Dict[str, Any]] = []

        def add_step(goal: str, required_capability: str, reason: str) -> None:
            if any(step.get("required_capability") == required_capability for step in steps):
                return

            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "goal": goal,
                    "required_capability": required_capability,
                    "reason": reason,
                    "topic": topic,
                    "current_step": current_step,
                }
            )

        # 1. Grounding support
        # In tutoring, most answers should be grounded in educational content.
        if self._needs_grounding(task_type, analyzed_task):
            add_step(
                goal="Retrieve grounded educational knowledge",
                required_capability="grounding",
                reason="The tutoring answer should be supported by verified educational content.",
            )

        # 2. Calculation support
        if self._needs_calculation(task_type, analyzed_task):
            add_step(
                goal="Perform deterministic calculation or verification",
                required_capability="calculation",
                reason="The task requires explicit numerical verification.",
            )

        # 3. Matrix computation support
        if self._needs_matrix_computation(task_type, analyzed_task):
            add_step(
                goal="Compute or verify a matrix-based example",
                required_capability="matrix_computation",
                reason="The task involves matrix, convolution, kernel, or linear-algebra operations.",
            )

        # 4. Visual support
        if self._needs_visualization(
            task_type=task_type,
            analyzed_task=analyzed_task,
            learner_level=learner_level,
            is_weak_topic=is_weak_topic,
        ):
            add_step(
                goal="Generate visual or step-by-step support",
                required_capability="visualization",
                reason="The learner benefits from a concrete visual or step-by-step explanation.",
            )

        # 5. Code execution support
        if self._needs_code_execution(task_type, analyzed_task):
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "goal": "Generate and execute a safe code-based verification",
                    "required_capability": "code_execution",
                    "reason": "The task requires executable code support.",
                    "topic": topic,
                    "current_step": current_step,
                    "requires_code": True,
                    "code_generation_instruction": analyzed_task.get(
                        "code_generation_instruction",
                        "Generate a minimal, safe, executable Python snippet that answers the student request. "
                        "The code must be deterministic, beginner-friendly, avoid unsafe imports and file/network operations, "
                        "and print its result. "
                        "When the request specifies matrix and kernel sizes, the generated code must use exactly those dimensions. "
                        "For example, if the request says a 3x3 matrix and a 2x2 kernel, do not create a 3x3 kernel. "
                        "For valid convolution or cross-correlation, compute output_rows = matrix_rows - kernel_rows + 1 and "
                        "output_cols = matrix_cols - kernel_cols + 1. "
                        "Iterate only over the output dimensions. "
                        "Every extracted patch must have the same shape as the kernel before multiplication. "
                        "If a previous execution error mentions broadcasting, operands, or shape mismatch, correct the matrix/kernel dimensions before retrying.",
                    ),
                    "student_question": student_question,
                    "code_source": "CodeDraftAgent",
                }
            )

        # 6. Recovery support
        # FailureRecovery may request concrete tools. The planner converts them back
        # into required capabilities so the selector can map them cleanly.
        for tool_name in recovery_required_tools:
            capability = self._capability_from_tool(tool_name)
            add_step(
                goal=f"Add missing support requested by recovery: {capability}",
                required_capability=capability,
                reason=f"Recovery requested this capability after validation failure: {recovery_reason}",
            )

        # Safety fallback: avoid empty plans.
        if not steps:
            add_step(
                goal="Retrieve grounded educational knowledge",
                required_capability="grounding",
                reason="Fallback planning step for general tutoring support.",
            )

        return ToolPlan(
            plan_id=new_id("PLAN"),
            task_type=task_type,
            selected_tools=[],
            steps=steps,
            metadata={
                "source_component": "ToolPlanner",
                "planning_mode": "capability_first_no_retrieval",
                "topic": topic,
                "current_step": current_step,
                "student_question": student_question,
                "goal": analyzed_task.get("goal"),
                "expected_output": expected_output,
                "learner_level": learner_level,
                "is_weak_topic": is_weak_topic,
                "collected_context": collected_context,
                "collected_context_summary": context_summary,
                "needs_code_execution": analyzed_task.get("needs_code_execution", False),
                "code_generation_required": analyzed_task.get("code_generation_required", False),
                "code_generation_instruction": analyzed_task.get("code_generation_instruction"),
                "recovery_reason": recovery_reason,
                "recovery_required_tools": recovery_required_tools,
                "previous_code_execution_errors": analyzed_task.get("previous_code_execution_errors", []),
            },
        )

    def _needs_grounding(self, task_type: str, analyzed_task: Dict[str, Any]) -> bool:
        # Most tutoring tasks need grounding.
        return task_type not in {"pure_calculation"}

    def _needs_calculation(self, task_type: str, analyzed_task: Dict[str, Any]) -> bool:
        text = self._combined_text(analyzed_task)
        calculation_keywords = [
            "calculate",
            "compute",
            "verify",
            "result",
            "solve",
            "arithmetic",
            "number",
            "equation",
        ]
        return task_type in {"calculation_or_verification", "calculation", "math_verification"} or any(
            keyword in text for keyword in calculation_keywords
        )

    def _needs_matrix_computation(self, task_type: str, analyzed_task: Dict[str, Any]) -> bool:
        text = self._combined_text(analyzed_task)
        matrix_keywords = [
            "matrix",
            "matrice",
            "matrices",
            "multiplication",
            "convolution",
            "kernel",
            "linear algebra",
        ]
        return any(keyword in text for keyword in matrix_keywords)

    def _needs_visualization(
        self,
        task_type: str,
        analyzed_task: Dict[str, Any],
        learner_level: Optional[str],
        is_weak_topic: bool,
    ) -> bool:
        text = self._combined_text(analyzed_task)
        visual_keywords = [
            "visual",
            "diagram",
            "show",
            "step-by-step",
            "simple example",
            "explain simply",
            "beginner",
        ]

        return (
            task_type == "visual_explanation"
            or learner_level == "beginner"
            or is_weak_topic
            or self._needs_matrix_computation(task_type, analyzed_task)
            or any(keyword in text for keyword in visual_keywords)
        )

    def _needs_code_execution(self, task_type: str, analyzed_task: Dict[str, Any]) -> bool:
        return task_type in {"code_help", "programming", "code_execution", "debugging"} or bool(
            analyzed_task.get("needs_code_execution", False)
        )

    def _combined_text(self, analyzed_task: Dict[str, Any]) -> str:
        parts = [
            analyzed_task.get("topic", ""),
            analyzed_task.get("task_type", ""),
            analyzed_task.get("current_step", ""),
            analyzed_task.get("expected_output", ""),
            analyzed_task.get("student_question", ""),
            analyzed_task.get("user_query", ""),
            analyzed_task.get("query", ""),
        ]

        investigation = analyzed_task.get("investigation_result", {})
        if isinstance(investigation, dict):
            parts.append(str(investigation.get("tool_request_goal", "")))
            parts.append(str(investigation.get("initial_tutoring_plan", "")))

        return " ".join(str(part) for part in parts if part).lower()

    def _capability_from_tool(self, tool_name: str) -> str:
        mapping = {
            "RAGTool": "grounding",
            "TraceSearchTool": "trace_reuse",
            "MatrixComputationTool": "matrix_computation",
            "VisualMatrixTool": "visualization",
            "CalculatorTool": "calculation",
            "CodeSandboxTool": "code_execution",
        }
        return mapping.get(tool_name, "general_tool_support")