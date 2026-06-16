from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolPlan, new_id


class ToolPlanner:
    """
    Creates an ordered plan of tool calls for the centralized Tool Interaction Manager.

    Input:
    - selected_tools: tools chosen by the ToolSelector
    - analyzed_task: structured task/context returned by TaskContextAnalyzer

    Output:
    - ToolPlan with task_type, selected_tools, and ordered execution steps
    """

    def create_plan(self, selected_tools: List[str], analyzed_task: Dict[str, Any]) -> ToolPlan:
        task_type = analyzed_task.get("task_type", "unknown_task")
        topic = analyzed_task.get("topic", "general topic")
        current_step = analyzed_task.get("current_step", "unknown step")

        steps = []

        for index, tool_name in enumerate(selected_tools, start=1):
            steps.append(
                {
                    "step_id": index,
                    "tool_name": tool_name,
                    "purpose": f"Use {tool_name} for {task_type}",
                    "topic": topic,
                    "current_step": current_step,
                }
            )

        return ToolPlan(
            plan_id=new_id("PLAN"),
            task_type=task_type,
            selected_tools=selected_tools,
            steps=steps,
            metadata={
                "source_component": "ToolPlanner",
                "topic": topic,
                "current_step": current_step,
            },
        )
