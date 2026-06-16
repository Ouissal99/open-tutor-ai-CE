from ai.agentic.core.schemas import ToolRequest, new_id


class ToolRequestAgent:
    """
    Lightweight proxy between the tutoring workflow and the centralized
    Tool Interaction Manager.

    It does not execute tools directly.
    It only creates a structured ToolRequest.
    """

    def create_request(self, student_question: str, step_goal: str) -> ToolRequest:
        task_type = self._infer_task_type(student_question, step_goal)

        return ToolRequest(
            request_id=new_id("REQ"),
            student_question=student_question,
            step_goal=step_goal,
            task_type=task_type,
            expected_output="visual explanation with grounded evidence",
            workflow_source="personalized_problem_tutoring",
            metadata={
                "source_component": "ToolRequestAgent",
                "current_step": step_goal,
            },
        )

    def _infer_task_type(self, student_question: str, step_goal: str) -> str:
        text = f"{student_question} {step_goal}".lower()

        if "visual" in text or "image" in text or "kernel" in text or "convolution" in text:
            return "visual_explanation"

        if "calculate" in text or "compute" in text or "solve" in text:
            return "calculation_or_verification"

        return "concept_explanation"
