from ai.agentic.core.schemas import ToolPlan, new_id


class ToolPlanner:
    """
    Orders selected tools into an execution sequence.
    """

    def create_plan(self, selected_tools, analyzed_task):
        steps = []
        for index, tool_name in enumerate(selected_tools, start=1):
            steps.append({
                "step": index,
                "tool_name": tool_name,
                "purpose": self._purpose(tool_name)
            })

        return ToolPlan(
            plan_id=new_id("PLAN"),
            selected_tools=selected_tools,
            steps=steps
        )

    def _purpose(self, tool_name: str) -> str:
        purposes = {
            "mock_rag_tool": "retrieve course evidence",
            "mock_visualizer_tool": "generate visual support",
            "mock_validator_support": "support validation"
        }
        return purposes.get(tool_name, "execute tool")
