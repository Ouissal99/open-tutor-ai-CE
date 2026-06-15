from ai.agentic.core.schemas import OutputPackage, new_id


class OutputPackageBuilder:
    """
    Builds success or fallback packages returned to the calling workflow.
    """

    def build_success(self, request, tool_results, validation_report, trace_id):
        evidence = []
        references = []
        content_parts = []

        for result in tool_results:
            content_parts.append(result.output)
            evidence.extend(result.evidence)
            references.extend(result.references)

        return OutputPackage(
            package_id=new_id("PKG"),
            status="validated",
            content=" ".join(content_parts),
            evidence=evidence,
            references=references,
            validation_report=validation_report,
            trace_id=trace_id,
            metadata={
                "workflow_source": request.workflow_source,
                "task_type": request.task_type,
                "current_step": request.current_step
            }
        )

    def build_fallback(self, request, tool_results, validation_report, trace_id):
        safe_content = "The system could not produce a fully validated tool output. Returning safe partial support."

        evidence = []
        references = []
        for result in tool_results:
            evidence.extend(result.evidence)
            references.extend(result.references)

        return OutputPackage(
            package_id=new_id("PKG"),
            status="fallback",
            content=safe_content,
            evidence=evidence,
            references=references,
            validation_report=validation_report,
            trace_id=trace_id,
            metadata={
                "workflow_source": request.workflow_source,
                "task_type": request.task_type,
                "current_step": request.current_step,
                "fallback_reason": validation_report.failure_reason
            }
        )
