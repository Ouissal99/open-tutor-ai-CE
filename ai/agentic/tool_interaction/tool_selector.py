class ToolSelector:
    """
    Memory-aware selection will be added later.
    For the demo, selection is rule-based.
    """

    def select(self, analyzed_task, attempt_number: int):
        task_type = analyzed_task["task_type"]
        needs_visual = analyzed_task.get("needs_visual_support", False)

        tools = ["mock_rag_tool"]

        if needs_visual or task_type == "visual_explanation":
            tools.append("mock_visualizer_tool")

        tools.append("mock_validator_support")

        return tools
