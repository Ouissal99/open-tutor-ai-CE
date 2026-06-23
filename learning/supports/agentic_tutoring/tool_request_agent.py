from typing import Any, Dict, List, Optional

from ai.agentic.core.schemas import ToolRequest, new_id


class ToolRequestAgent:
    """
    Lightweight proxy between the tutoring workflow and the centralized
    Tool Interaction Manager.

    It does not execute tools directly.
    It only creates a structured ToolRequest.
    """

    def create_request(
        self,
        student_question: str,
        step_goal: str,
        learner_id: str = "demo_user",
        context: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        expected_output: Optional[str] = None,
        suggested_tools: Optional[List[str]] = None,
        step_id: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> ToolRequest:
        resolved_task_type = task_type or self._infer_task_type(student_question, step_goal)
        resolved_expected_output = expected_output or self._infer_expected_output(resolved_task_type)

        metadata = {
            "source_component": "ToolRequestAgent",
            "learner_id": learner_id,
            "current_step": step_goal,
            "plan_step_id": step_id,
            "step_reason": reason,
            "suggested_tools": suggested_tools or [],
        }

        if context:
            metadata.update(context)

        return ToolRequest(
            request_id=new_id("REQ"),
            student_question=student_question,
            step_goal=step_goal,
            task_type=resolved_task_type,
            expected_output=resolved_expected_output,
            workflow_source="personalized_problem_tutoring",
            metadata=metadata,
        )

    def _infer_task_type(self, student_question: str, step_goal: str) -> str:
        text = f"{student_question} {step_goal}".lower()

        if "visual" in text or "image" in text or "kernel" in text or "convolution" in text:
            return "visual_explanation"

        if "calculate" in text or "compute" in text or "solve" in text:
            return "calculation_or_verification"

        return "conceptual_explanation"

    def _infer_expected_output(self, task_type: str) -> str:
        if task_type == "visual_explanation":
            return "visual explanation with grounded evidence"

        if task_type == "calculation_or_verification":
            return "verified calculation with evidence"

        if task_type in {"code_help", "programming", "code_execution"}:
            return "grounded code help with execution evidence"

        return "grounded tutoring explanation"
