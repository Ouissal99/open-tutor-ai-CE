from typing import Any, Dict, List, Optional

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tool_interaction.tool_registry import ToolRegistry
from ai.agentic.tool_interaction.code_draft_agent import CodeDraftAgent


class ToolExecutor:
    """
    Executes external tools through the ToolRegistry.

    For CodeSandboxTool, the executor first calls the internal CodeDraftAgent
    to generate candidate code using the LLM, then sends that code to the
    sandbox for execution.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        code_draft_agent: Optional[CodeDraftAgent] = None,
    ):
        self.registry = registry or ToolRegistry()
        self.code_draft_agent = code_draft_agent or CodeDraftAgent()

    def execute(self, plan: Any, request: Any, attempt: int = 1) -> List[ToolResult]:
        results: List[ToolResult] = []

        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
        metadata = plan_dict.get("metadata", {})
        collected_context = metadata.get("collected_context", {})

        analyzed_task = {
            "task_type": plan_dict.get("task_type"),
            "selected_tools": plan_dict.get("selected_tools", []),
            "tool_selection": metadata.get("tool_selection", {}),
            "collected_context": collected_context,
            "topic": metadata.get("topic"),
            "current_step": metadata.get("current_step"),
            "student_question": metadata.get("student_question"),
            "goal": metadata.get("goal"),
            "expected_output": metadata.get("expected_output"),
            "needs_code_execution": metadata.get("needs_code_execution", False),
            "code_generation_required": metadata.get("code_generation_required", False),
            "code_generation_instruction": metadata.get("code_generation_instruction"),
            "recovery_reason": metadata.get("recovery_reason"),
            "previous_code_execution_errors": metadata.get("previous_code_execution_errors", []),
        }

        for original_step in getattr(plan, "steps", []):
            step = dict(original_step)
            tool_name = step.get("tool_name")

            try:
                if tool_name == "CodeSandboxTool" and step.get("requires_code") and not step.get("code"):
                    try:
                        code_draft = self.code_draft_agent.generate(
                            request=request,
                            step=step,
                            collected_context=collected_context,
                            analyzed_task=analyzed_task,
                            attempt=attempt,
                        )
                        step["code"] = code_draft["code"]
                        step["code_draft"] = code_draft
                        step["code_source"] = "CodeDraftAgent"

                    except Exception as exc:
                        results.append(
                            ToolResult(
                                tool_name="CodeSandboxTool",
                                status="failed",
                                success=False,
                                output=f"Code drafting failed before sandbox execution: {exc}",
                                evidence=[],
                                references=[],
                                metadata={
                                    "attempt": attempt,
                                    "source_component": "ToolExecutor",
                                    "failure_reason": "code_drafting_failed",
                                    "error": str(exc),
                                    "registry_tool": True,
                                },
                            )
                        )
                        continue

                tool = self.registry.get(tool_name)
                result = tool.run(
                    request=request,
                    step=step,
                    collected_context=collected_context,
                    analyzed_task=analyzed_task,
                    attempt=attempt,
                )
                results.append(result)

            except Exception as exc:
                results.append(
                    ToolResult(
                        tool_name=tool_name or "UnknownTool",
                        status="failed",
                        success=False,
                        output=f"Tool execution failed for {tool_name}: {exc}",
                        evidence=[],
                        references=[],
                        metadata={
                            "attempt": attempt,
                            "source_component": "ToolExecutor",
                            "failure_reason": "tool_registry_execution_failed",
                            "error": str(exc),
                            "available_tools": self.registry.list_tools(),
                        },
                    )
                )

        return results
