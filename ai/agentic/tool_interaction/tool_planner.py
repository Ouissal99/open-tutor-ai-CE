from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolPlan, new_id


class ToolPlanner:
    """
    Creates an ordered plan of tool calls for the centralized Tool Interaction Manager.

    For code-related tasks, the planner marks the CodeSandboxTool step as
    requiring an LLM-generated code draft before execution.
    """

    def create_plan(self, selected_tools: List[str], analyzed_task: Dict[str, Any]) -> ToolPlan:
        task_type = analyzed_task.get("task_type", "unknown_task")
        topic = analyzed_task.get("topic", "general topic")
        current_step = analyzed_task.get("current_step", "unknown step")

        steps = []

        for index, tool_name in enumerate(selected_tools, start=1):
            step = {
                "step_id": index,
                "tool_name": tool_name,
                "purpose": f"Use {tool_name} for {task_type}",
                "topic": topic,
                "current_step": current_step,
            }

            if tool_name == "CodeSandboxTool":
                step["requires_code"] = True
                step["code_generation_instruction"] = analyzed_task.get(
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
                )
                step["student_question"] = analyzed_task.get("student_question", "")
                step["code_source"] = "CodeDraftAgent"

            steps.append(step)

        return ToolPlan(
            plan_id=new_id("PLAN"),
            task_type=task_type,
            selected_tools=selected_tools,
            steps=steps,
            metadata={
                "source_component": "ToolPlanner",
                "topic": topic,
                "current_step": current_step,
                "student_question": analyzed_task.get("student_question", ""),
                "goal": analyzed_task.get("goal"),
                "expected_output": analyzed_task.get("expected_output"),
                "tool_selection": analyzed_task.get("tool_selection", {}),
                "collected_context": analyzed_task.get("collected_context", {}),
                "needs_code_execution": analyzed_task.get("needs_code_execution", False),
                "code_generation_required": analyzed_task.get("code_generation_required", False),
                "code_generation_instruction": analyzed_task.get("code_generation_instruction"),
                "recovery_reason": analyzed_task.get("recovery_reason"),
                "previous_code_execution_errors": analyzed_task.get("previous_code_execution_errors", []),
            },
        )
