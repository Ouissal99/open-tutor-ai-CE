"""Centralized Tool Interaction Manager.

This public manager is now backed by a LangGraph orchestration engine.
The external API remains the same:

    package, trace_path = ToolInteractionManager().run(request)
"""

from ai.agentic.tool_interaction.tool_interaction_graph import ToolInteractionGraph


class ToolInteractionManager:
    """Centralized T2-Based Adaptive Tool Interaction Manager."""

    def __init__(self):
        self.graph = ToolInteractionGraph(max_attempts=3)

    def run(self, request):
        print("\n[4.0] LangGraph ToolInteractionGraph started")

        state = self.graph.invoke(request)

        print("    node_history:", state.get("node_history"))
        print("    final_status:", state.get("final_status"))

        package = state.get("package")
        trace_path = state.get("trace_path")

        if package is None:
            raise RuntimeError("ToolInteractionGraph finished without building an OutputPackage.")

        if trace_path is None:
            raise RuntimeError("ToolInteractionGraph finished without saving a trace.")

        return package, trace_path
