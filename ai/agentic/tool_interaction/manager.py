from ai.agentic.tool_interaction.task_context_analyzer import TaskContextAnalyzer
from ai.agentic.tool_interaction.tool_selector import ToolSelector
from ai.agentic.tool_interaction.tool_planner import ToolPlanner
from ai.agentic.tool_interaction.tool_executor import ToolExecutor
from ai.agentic.tool_interaction.output_validator import OutputValidator
from ai.agentic.tool_interaction.failure_recovery import FailureRecovery
from ai.agentic.tool_interaction.output_package_builder import OutputPackageBuilder
from ai.agentic.tool_interaction.trace_event_bus import TraceEventBus


class ToolInteractionManager:
    """
    Centralized T2-Based Adaptive Tool Interaction Manager.
    """

    def __init__(self):
        self.analyzer = TaskContextAnalyzer()
        self.selector = ToolSelector()
        self.planner = ToolPlanner()
        self.executor = ToolExecutor()
        self.validator = OutputValidator()
        self.recovery = FailureRecovery(max_attempts=3)
        self.package_builder = OutputPackageBuilder()

    def run(self, request):
        trace = TraceEventBus()
        trace.emit("tool_request_received", request)

        analyzed_task = self.analyzer.analyze(request)
        trace.emit("task_context_analyzed", analyzed_task)

        last_results = []
        last_report = None

        for attempt in range(1, self.recovery.max_attempts + 1):
            trace.emit("attempt_started", {"attempt_number": attempt})

            selected_tools = self.selector.select(analyzed_task, attempt)
            trace.emit("tools_selected", selected_tools)

            plan = self.planner.create_plan(selected_tools, analyzed_task)
            trace.emit("tool_plan_created", plan)

            results = self.executor.execute(plan, request, attempt)
            last_results = results
            trace.emit("tools_executed", results)

            report = self.validator.validate(request, results, attempt)
            last_report = report
            trace.emit("output_validated", report)

            if report.status == "valid":
                package = self.package_builder.build_success(
                    request=request,
                    tool_results=results,
                    validation_report=report,
                    trace_id=trace.trace_id
                )
                trace.emit("output_package_built", package)
                trace_path = trace.save(package)
                return package, trace_path

            recovery_decision = self.recovery.decide(report, attempt)
            trace.emit("recovery_decision", recovery_decision)

            if not recovery_decision["should_retry"]:
                package = self.package_builder.build_fallback(
                    request=request,
                    tool_results=last_results,
                    validation_report=last_report,
                    trace_id=trace.trace_id
                )
                trace.emit("fallback_package_built", package)
                trace_path = trace.save(package)
                return package, trace_path

        package = self.package_builder.build_fallback(
            request=request,
            tool_results=last_results,
            validation_report=last_report,
            trace_id=trace.trace_id
        )
        trace_path = trace.save(package)
        return package, trace_path
