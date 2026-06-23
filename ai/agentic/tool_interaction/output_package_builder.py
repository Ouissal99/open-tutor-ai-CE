from collections import OrderedDict
from typing import Any, Dict, List

from ai.agentic.core.schemas import OutputPackage, new_id


class OutputPackageBuilder:
    """
    Builds success or fallback packages returned to the calling workflow.

    Phase 11B:
    - Preserve each tool output as a separate section.
    - Deduplicate evidence and references.
    - Store structured tool outputs in metadata.
    """

    def build_success(self, request, tool_results, validation_report, trace_id):
        evidence = []
        references = []
        tool_outputs = OrderedDict()
        tool_metadata = OrderedDict()

        for result in tool_results:
            tool_name = result.tool_name or "UnknownTool"
            output = (result.output or "").strip()

            if output:
                tool_outputs.setdefault(tool_name, [])
                tool_outputs[tool_name].append(output)

            evidence.extend(result.evidence or [])
            references.extend(result.references or [])

            metadata = result.metadata or {}
            if metadata:
                tool_metadata.setdefault(tool_name, [])
                tool_metadata[tool_name].append(metadata)

        content = self._format_tool_outputs(tool_outputs)

        return OutputPackage(
            package_id=new_id("PKG"),
            status="validated",
            content=content,
            evidence=self._dedupe(evidence),
            references=self._dedupe(references),
            validation_report=validation_report,
            trace_id=trace_id,
            metadata={
                "workflow_source": request.workflow_source,
                "task_type": request.task_type,
                "current_step": request.current_step,
                "tool_outputs": dict(tool_outputs),
                "tool_metadata": dict(tool_metadata),
                "tool_names": list(tool_outputs.keys()),
                "content_format": "structured_tool_sections",
            },
        )

    def build_fallback(self, request, tool_results, validation_report, trace_id):
        evidence = []
        references = []
        tool_outputs = OrderedDict()
        tool_metadata = OrderedDict()

        for result in tool_results:
            tool_name = result.tool_name or "UnknownTool"
            output = (result.output or "").strip()

            if output:
                tool_outputs.setdefault(tool_name, [])
                tool_outputs[tool_name].append(output)

            evidence.extend(result.evidence or [])
            references.extend(result.references or [])

            metadata = result.metadata or {}
            if metadata:
                tool_metadata.setdefault(tool_name, [])
                tool_metadata[tool_name].append(metadata)

        partial_content = self._format_tool_outputs(tool_outputs)
        safe_content = (
            "The system could not produce a fully validated tool output. "
            "Returning safe partial support."
        )

        if partial_content:
            safe_content += "\n\n" + partial_content

        return OutputPackage(
            package_id=new_id("PKG"),
            status="fallback",
            content=safe_content,
            evidence=self._dedupe(evidence),
            references=self._dedupe(references),
            validation_report=validation_report,
            trace_id=trace_id,
            metadata={
                "workflow_source": request.workflow_source,
                "task_type": request.task_type,
                "current_step": request.current_step,
                "fallback_reason": validation_report.failure_reason,
                "tool_outputs": dict(tool_outputs),
                "tool_metadata": dict(tool_metadata),
                "tool_names": list(tool_outputs.keys()),
                "content_format": "structured_tool_sections",
            },
        )

    def _format_tool_outputs(self, tool_outputs: OrderedDict) -> str:
        sections = []

        for tool_name, outputs in tool_outputs.items():
            clean_outputs = [
                output.strip()
                for output in outputs
                if output and output.strip()
            ]

            if not clean_outputs:
                continue

            section_body = "\n\n".join(clean_outputs)
            sections.append(f"## {tool_name}\n{section_body}")

        return "\n\n".join(sections)

    def _dedupe(self, items: List[Any]) -> List[Any]:
        deduped = []
        seen = set()

        for item in items:
            if item is None:
                continue

            key = str(item).strip()

            if not key or key in seen:
                continue

            seen.add(key)
            deduped.append(item)

        return deduped
