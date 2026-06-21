"""LangGraph implementation of the centralized Tool Interaction Manager.

This graph preserves the exact working logic of the original manager:

analyze → collect context → select tools → plan tools → execute tools
→ validate output → conditional route:
    valid → build success package
    invalid + retry allowed → recovery → next attempt
    invalid + no retry → build fallback package

LangGraph is used only as the orchestration engine.
The domain logic stays inside the OpenTutorAI agentic components.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from ai.agentic.tool_interaction.tool_interaction_state import ToolInteractionState
from ai.agentic.tool_interaction.task_context_analyzer import TaskContextAnalyzer
from ai.agentic.tool_interaction.context_collector import ContextCollector
from ai.agentic.tool_interaction.tool_selector import ToolSelector
from ai.agentic.tool_interaction.tool_planner import ToolPlanner
from ai.agentic.tool_interaction.tool_executor import ToolExecutor
from ai.agentic.tool_interaction.output_validator import OutputValidator
from ai.agentic.tool_interaction.failure_recovery import FailureRecovery
from ai.agentic.tool_interaction.output_package_builder import OutputPackageBuilder
from ai.agentic.tool_interaction.trace_event_bus import TraceEventBus


class ToolInteractionGraph:
    """Stateful LangGraph orchestration for the centralized tool manager."""

    def __init__(
        self,
        analyzer: Optional[TaskContextAnalyzer] = None,
        context_collector: Optional[ContextCollector] = None,
        selector: Optional[ToolSelector] = None,
        planner: Optional[ToolPlanner] = None,
        executor: Optional[ToolExecutor] = None,
        validator: Optional[OutputValidator] = None,
        recovery: Optional[FailureRecovery] = None,
        package_builder: Optional[OutputPackageBuilder] = None,
        max_attempts: int = 3,
    ):
        self.analyzer = analyzer or TaskContextAnalyzer()
        self.context_collector = context_collector or ContextCollector()
        self.selector = selector or ToolSelector()
        self.planner = planner or ToolPlanner()
        self.executor = executor or ToolExecutor()
        self.validator = validator or OutputValidator()
        self.recovery = recovery or FailureRecovery(max_attempts=max_attempts)
        self.package_builder = package_builder or OutputPackageBuilder()

        self.max_attempts = max_attempts
        self.graph = self._build_graph()

    def invoke(self, request: Any) -> ToolInteractionState:
        """Run the full graph and return final state."""
        trace_bus = TraceEventBus()
        trace_bus.emit("tool_request_received", request)

        initial_state: ToolInteractionState = {
            "request": request,
            "trace_bus": trace_bus,
            "trace_id": trace_bus.trace_id,
            "attempt": 1,
            "max_attempts": self.max_attempts,
            "last_results": [],
            "last_report": None,
            "recovery_used": False,
            "node_history": [],
            "trace_events": [],
            "final_status": "running",
        }

        return self.graph.invoke(initial_state)

    def _build_graph(self):
        workflow = StateGraph(ToolInteractionState)

        workflow.add_node("analyze_task", self._analyze_task_node)
        workflow.add_node("collect_context", self._collect_context_node)
        workflow.add_node("select_tools", self._select_tools_node)
        workflow.add_node("plan_tools", self._plan_tools_node)
        workflow.add_node("execute_tools", self._execute_tools_node)
        workflow.add_node("validate_output", self._validate_output_node)
        workflow.add_node("recover_failure", self._recover_failure_node)
        workflow.add_node("build_success_package", self._build_success_package_node)
        workflow.add_node("build_fallback_package", self._build_fallback_package_node)

        workflow.set_entry_point("analyze_task")

        workflow.add_edge("analyze_task", "collect_context")
        workflow.add_edge("collect_context", "select_tools")
        workflow.add_edge("select_tools", "plan_tools")
        workflow.add_edge("plan_tools", "execute_tools")
        workflow.add_edge("execute_tools", "validate_output")

        workflow.add_conditional_edges(
            "validate_output",
            self._route_after_validation,
            {
                "success": "build_success_package",
                "retry": "recover_failure",
                "fallback": "build_fallback_package",
            },
        )

        workflow.add_edge("recover_failure", "select_tools")
        workflow.add_edge("build_success_package", END)
        workflow.add_edge("build_fallback_package", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _analyze_task_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "analyze_task")

        request = state["request"]
        trace_bus = state["trace_bus"]

        analyzed_task = self.analyzer.analyze(request)

        state["analyzed_task"] = analyzed_task
        trace_bus.emit("task_analyzed", analyzed_task)
        self._add_trace_event(state, "analyze_task", {"status": "completed"})

        return state

    def _collect_context_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "collect_context")

        request = state["request"]
        trace_bus = state["trace_bus"]
        analyzed_task = state["analyzed_task"]

        analyzed_task = self.context_collector.enrich_task(
            request=request,
            analyzed_task=analyzed_task,
        )

        collected_context = analyzed_task.get("collected_context", {})
        context_summary = collected_context.get("summary", {})

        state["analyzed_task"] = analyzed_task
        state["collected_context"] = collected_context
        state["context_summary"] = context_summary

        print("\n[4.0] LangGraph ContextCollector enriched task context")
        print(f"    learner_id: {collected_context.get('learner_id')}")
        print(f"    learner_level: {context_summary.get('learner_level')}")
        print(f"    is_weak_topic: {context_summary.get('is_weak_topic')}")
        print(f"    retrieved_knowledge_chunks: {context_summary.get('retrieved_knowledge_chunks')}")
        print(f"    similar_successful_traces: {context_summary.get('similar_successful_trace_count')}")
        print(f"    failed_traces: {context_summary.get('failed_trace_count')}")

        trace_bus.emit("task_context_analyzed", analyzed_task)

        self._add_trace_event(
            state,
            "collect_context",
            {
                "status": "completed",
                "learner_id": collected_context.get("learner_id"),
                "learner_level": context_summary.get("learner_level"),
                "retrieved_knowledge_chunks": context_summary.get("retrieved_knowledge_chunks"),
            },
        )

        return state

    def _select_tools_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "select_tools")

        analyzed_task = state["analyzed_task"]
        attempt = state["attempt"]
        trace_bus = state["trace_bus"]

        selected_tools = self.selector.select(analyzed_task, attempt)

        selection_metadata: Dict[str, Any] = {}
        if hasattr(self.selector, "get_last_selection_metadata"):
            selection_metadata = self.selector.get_last_selection_metadata()
            analyzed_task["tool_selection"] = selection_metadata

            print("\n[4.1] LangGraph memory-aware tool selection")
            print(f"    strategy: {selection_metadata.get('selection_strategy')}")
            print(f"    reason: {selection_metadata.get('selection_reason')}")
            print(f"    reference_trace_id: {selection_metadata.get('reference_trace_id')}")
            print(f"    selected_tools: {selection_metadata.get('selected_tools')}")

        state["selected_tools"] = selected_tools
        state["selection_metadata"] = selection_metadata
        state["analyzed_task"] = analyzed_task

        trace_bus.emit("tools_selected", selected_tools)

        self._add_trace_event(
            state,
            "select_tools",
            {
                "status": "completed",
                "attempt": attempt,
                "selected_tools": selected_tools,
                "selection_metadata": selection_metadata,
            },
        )

        return state

    def _plan_tools_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "plan_tools")

        selected_tools = state["selected_tools"]
        analyzed_task = state["analyzed_task"]
        trace_bus = state["trace_bus"]

        plan = self.planner.create_plan(selected_tools, analyzed_task)

        state["plan"] = plan
        trace_bus.emit("tool_plan_created", plan)

        self._add_trace_event(
            state,
            "plan_tools",
            {
                "status": "completed",
                "attempt": state["attempt"],
            },
        )

        return state

    def _execute_tools_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "execute_tools")

        request = state["request"]
        plan = state["plan"]
        attempt = state["attempt"]
        trace_bus = state["trace_bus"]

        results = self.executor.execute(plan, request, attempt)

        state["results"] = results
        state["last_results"] = results

        trace_bus.emit("tools_executed", results)

        self._add_trace_event(
            state,
            "execute_tools",
            {
                "status": "completed",
                "attempt": attempt,
                "tool_result_count": len(results),
            },
        )

        return state

    def _validate_output_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "validate_output")

        request = state["request"]
        results = state["results"]
        attempt = state["attempt"]
        trace_bus = state["trace_bus"]

        report = self.validator.validate(request, results, attempt)

        state["report"] = report
        state["last_report"] = report

        trace_bus.emit("output_validated", report)

        if report.status == "valid":
            state["route_decision"] = "success"
            state["final_status"] = "validated"
        else:
            recovery_decision = self.recovery.decide(report, attempt)
            state["recovery_decision"] = recovery_decision
            trace_bus.emit("recovery_decision", recovery_decision)

            if recovery_decision.get("should_retry"):
                state["route_decision"] = "retry"
                state["final_status"] = "retrying"
            else:
                state["route_decision"] = "fallback"
                state["final_status"] = "fallback"

        self._add_trace_event(
            state,
            "validate_output",
            {
                "status": report.status,
                "attempt": attempt,
                "route_decision": state.get("route_decision"),
            },
        )

        return state

    def _recover_failure_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "recover_failure")

        previous_attempt = state["attempt"]
        state["attempt"] = previous_attempt + 1
        state["recovery_used"] = True

        self._add_trace_event(
            state,
            "recover_failure",
            {
                "status": "completed",
                "previous_attempt": previous_attempt,
                "next_attempt": state["attempt"],
                "recovery_decision": state.get("recovery_decision", {}),
            },
        )

        return state

    def _build_success_package_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "build_success_package")

        request = state["request"]
        results = state["results"]
        report = state["report"]
        trace_bus = state["trace_bus"]

        package = self.package_builder.build_success(
            request=request,
            tool_results=results,
            validation_report=report,
            trace_id=trace_bus.trace_id,
        )

        trace_bus.emit("output_package_built", package)
        trace_path = trace_bus.save(package)

        state["package"] = package
        state["trace_path"] = trace_path
        state["final_status"] = "validated"

        self._add_trace_event(
            state,
            "build_success_package",
            {
                "status": "completed",
                "trace_path": trace_path,
            },
        )

        return state

    def _build_fallback_package_node(self, state: ToolInteractionState) -> ToolInteractionState:
        self._mark_node(state, "build_fallback_package")

        request = state["request"]
        results = state.get("last_results", [])
        report = state.get("last_report")
        trace_bus = state["trace_bus"]

        package = self.package_builder.build_fallback(
            request=request,
            tool_results=results,
            validation_report=report,
            trace_id=trace_bus.trace_id,
        )

        trace_bus.emit("fallback_package_built", package)
        trace_path = trace_bus.save(package)

        state["package"] = package
        state["trace_path"] = trace_path
        state["final_status"] = "fallback"

        self._add_trace_event(
            state,
            "build_fallback_package",
            {
                "status": "completed",
                "trace_path": trace_path,
            },
        )

        return state

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_validation(self, state: ToolInteractionState) -> str:
        return state.get("route_decision", "fallback")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mark_node(self, state: ToolInteractionState, node_name: str) -> None:
        state.setdefault("node_history", [])
        state["node_history"].append(node_name)

    def _add_trace_event(
        self,
        state: ToolInteractionState,
        node_name: str,
        payload: Dict[str, Any],
    ) -> None:
        state.setdefault("trace_events", [])
        state["trace_events"].append(
            {
                "node": node_name,
                **payload,
            }
        )
