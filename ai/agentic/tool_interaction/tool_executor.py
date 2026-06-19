from typing import Any, Dict, List

from ai.agentic.core.schemas import ToolResult


class ToolExecutor:
    """
    Executes selected mock tools.

    In Phase 5B, mock_rag_tool uses SKG snippets from collected_context.
    This makes the prototype less hard-coded and closer to the DeepTutor-style
    grounding flow.
    """

    def execute(self, plan: Any, request: Any, attempt: int = 1) -> List[ToolResult]:
        results = []

        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
        metadata = plan_dict.get("metadata", {})
        collected_context = metadata.get("collected_context", {})

        for step in getattr(plan, "steps", []):
            tool_name = step.get("tool_name")

            if tool_name == "mock_rag_tool":
                results.append(self._execute_rag_tool(tool_name, collected_context, attempt))

            elif tool_name == "mock_visualizer_tool":
                results.append(self._execute_visualizer_tool(tool_name, collected_context, attempt))

            elif tool_name == "mock_validator_support":
                results.append(self._execute_validator_support(tool_name, collected_context, attempt))

            else:
                results.append(
                    ToolResult(
                        tool_name=tool_name,
                        success=True,
                        output=f"Mock execution completed for {tool_name}.",
                        evidence=[],
                        references=[],
                        metadata={
                            "attempt": attempt,
                            "source_component": "ToolExecutor",
                        },
                    )
                )

        return results

    def _execute_rag_tool(
        self,
        tool_name: str,
        collected_context: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        skg = collected_context.get("skg", {})
        snippets = skg.get("snippets", [])
        grounding_text = skg.get("grounding_text", "")

        if snippets:
            evidence = [item.get("content", "") for item in snippets if item.get("content")]
            references = [item.get("source", "") for item in snippets if item.get("source")]
            output = "Course evidence retrieved from SKG:\n" + grounding_text
        else:
            evidence = ["No SKG snippet found; fallback evidence used."]
            references = ["Fallback mock knowledge source"]
            output = "Fallback RAG result: no SKG snippet was retrieved."

        return ToolResult(
            tool_name=tool_name,
            success=True,
            output=output,
            evidence=evidence,
            references=references,
            metadata={
                "attempt": attempt,
                "source_component": "ToolExecutor",
                "grounding_source": "StaticKnowledgeGrounding",
                "retrieved_knowledge_chunks": len(snippets),
            },
        )

    def _execute_visualizer_tool(
        self,
        tool_name: str,
        collected_context: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        dpm = collected_context.get("dpm", {})
        learner_level = dpm.get("learner_level", "unknown")
        preferred_examples = dpm.get("preferred_examples", [])

        output = (
            "Generated visual explanation: use a 2x2 kernel sliding over a 3x3 matrix. "
            f"The explanation is adapted for learner_level={learner_level}."
        )

        if preferred_examples:
            output += f" Preferred example style: {', '.join(preferred_examples)}."

        return ToolResult(
            tool_name=tool_name,
            success=True,
            output=output,
            evidence=[
                "Visualizer output describes kernel movement step by step.",
                "The example is adapted using learner preferences from DPM.",
            ],
            references=["Mock visualizer output"],
            metadata={
                "attempt": attempt,
                "source_component": "ToolExecutor",
                "personalization_source": "DynamicPersonalMemory",
                "learner_level": learner_level,
            },
        )

    def _execute_validator_support(
        self,
        tool_name: str,
        collected_context: Dict[str, Any],
        attempt: int,
    ) -> ToolResult:
        summary = collected_context.get("summary", {})
        retrieved_chunks = summary.get("retrieved_knowledge_chunks", 0)

        return ToolResult(
            tool_name=tool_name,
            success=True,
            output=(
                "Validator support tool ready. "
                f"Grounding chunks available: {retrieved_chunks}."
            ),
            evidence=[
                "Validation support confirms that grounded evidence and learner context are available."
            ],
            references=["Mock validator support"],
            metadata={
                "attempt": attempt,
                "source_component": "ToolExecutor",
                "retrieved_knowledge_chunks": retrieved_chunks,
            },
        )
