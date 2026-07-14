"""Centralized Tool Interaction Manager.

This public manager is now backed by a LangGraph orchestration engine.
The external API remains the same:

    package, trace_path = ToolInteractionManager().run(request)
"""

import os

from ai.agentic.tool_interaction.tool_interaction_graph import ToolInteractionGraph


class ToolInteractionManager:
    """Centralized T2-Based Adaptive Tool Interaction Manager."""

    def __init__(self):
        # Normal condition: max_attempts=3 with FailureRecovery enabled.
        # Baseline condition: NO_RECOVERY_BASELINE=1 forces max_attempts=1,
        # which prevents retry/recovery after an invalid first attempt.
        max_attempts = 1 if os.getenv("NO_RECOVERY_BASELINE") == "1" else 3
        self.graph = ToolInteractionGraph(max_attempts=max_attempts)

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
