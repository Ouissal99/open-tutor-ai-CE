import json
from pathlib import Path
from typing import Any, Dict, Optional

from ai.agentic.llm.llm_adapter import MockLLMAdapter
from ai.agentic.tutoring.scratchpad import StepBasedScratchpad
from ai.agentic.tutoring.tool_request_agent import ToolRequestAgent
from ai.agentic.tool_interaction.manager import ToolInteractionManager


class PersonalizedTutoringWorkflow:
    """
    DeepTutor-inspired personalized problem tutoring workflow.

    This workflow is the first complete vertical slice of the implementation:
    student question -> solving plan -> ToolRequestAgent -> centralized
    ToolInteractionManager -> output package -> scratchpad -> final answer.

    Phase 5C:
    The final answer is generated through a deterministic MockLLMAdapter using:
    - Dynamic Personal Memory
    - Static Knowledge Grounding
    - validated OutputPackage
    """

    def __init__(self):
        self.scratchpad = StepBasedScratchpad()
        self.tool_request_agent = ToolRequestAgent()
        self.tool_manager = ToolInteractionManager()
        self.llm_adapter = MockLLMAdapter()

    def run(
        self,
        student_question: str,
        learner_id: str = "demo_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}

        print("\n[1] Student question received")
        print(f"    {student_question}")

        print("\n[2] Creating simplified solving plan")
        solving_plan = [
            {
                "step_goal": "Explain the concept using course-grounded evidence",
                "needs_tool": False,
            },
            {
                "step_goal": "Generate visual support for kernel movement",
                "needs_tool": True,
            },
        ]

        final_package = None
        trace_path = None
        final_request = None

        for step in solving_plan:
            round_item = self.scratchpad.add_round(
                step_goal=step["step_goal"],
                analysis="The tutor prepares the current solving step.",
            )

            if step["needs_tool"]:
                print("\n[3] Tool Request Agent creates a structured ToolRequest")
                request = self.tool_request_agent.create_request(
                    student_question=student_question,
                    step_goal=step["step_goal"],
                    learner_id=learner_id,
                    context={
                        "workflow_source": "personalized_problem_tutoring",
                    },
                )
                final_request = request

                print(f"    request_id: {request.request_id}")
                print(f"    task_type: {request.task_type}")

                print("\n[4] Central Tool Interaction Manager runs")
                package, trace_path = self.tool_manager.run(request)
                final_package = package

                print("\n[5] OutputPackage received from central manager")
                print(f"    status: {package.status}")
                print(f"    confidence: {package.validation_report.confidence_score}")
                print(f"    trace_id: {package.trace_id}")

                self.scratchpad.update_round_with_package(
                    round_id=round_item.round_id,
                    request_summary=f"{request.task_type} / {request.expected_output}",
                    package=package,
                )
                print("\n[6] Scratchpad updated with validated package")

        final_answer = self._compose_answer(
            question=student_question,
            package=final_package,
            learner_id=learner_id,
        )

        print("\n[7] Final answer generated with MockLLMAdapter")
        print(final_answer)

        print("\n[8] Scratchpad state")
        print(json.dumps(self.scratchpad.to_dict(), indent=2))

        if trace_path:
            print("\n[9] Tool interaction trace saved")
            print(trace_path)

        package_dict = final_package.to_dict() if final_package else {}
        trace_metadata = self._extract_trace_metadata(trace_path)

        return {
            # Main result
            "answer": final_answer,
            "final_answer": final_answer,

            # Request information
            "learner_id": learner_id,
            "student_question": student_question,
            "request_id": final_request.request_id if final_request else None,
            "task_type": final_request.task_type if final_request else None,

            # Package / validation information
            "status": package_dict.get("status", "no_package"),
            "confidence": package_dict.get("validation_report", {}).get("confidence_score"),
            "trace_id": package_dict.get("trace_id"),
            "trace_path": str(trace_path) if trace_path else None,

            # Execution metadata
            "attempts": trace_metadata.get("attempts", 1),
            "selected_tools": trace_metadata.get("selected_tools", []),
            "recovery_used": trace_metadata.get("recovery_used", False),

            # Full objects for debugging, evaluation, and later UI/API
            "output_package": package_dict,
            "scratchpad": self.scratchpad.to_dict(),
            "metadata": {
                "workflow": "personalized_problem_tutoring",
                "source": "terminal_backend_prototype",
                "final_answer_generator": "MockLLMAdapter",
                **metadata,
            },
        }

    def _compose_answer(self, question: str, package: Any, learner_id: str) -> str:
        if not package:
            return "No validated tool package was generated, so the tutor cannot produce a grounded answer."

        return self.llm_adapter.generate_personalized_tutoring_answer(
            student_question=question,
            learner_id=learner_id,
            output_package=package,
            scratchpad=self.scratchpad.to_dict(),
        )

    def _extract_trace_metadata(self, trace_path) -> Dict[str, Any]:
        """
        Extract metadata for the current run only.

        Important:
        The trace may contain previous successful/failed traces inside
        collected_context.trace_memory. Those previous traces must not be counted
        as attempts for the current execution.
        """
        metadata = {
            "attempts": 1,
            "selected_tools": [],
            "recovery_used": False,
        }

        if not trace_path:
            return metadata

        path = Path(trace_path)
        if not path.exists():
            return metadata

        try:
            with path.open("r", encoding="utf-8") as f:
                trace_data = json.load(f)
        except Exception:
            return metadata

        attempt_numbers = []
        selected_tools = set()
        recovery_used = False

        def walk(obj, inside_previous_trace_memory=False):
            nonlocal recovery_used

            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Skip previous trace summaries stored in context memory.
                    if key in {
                        "similar_successful_traces",
                        "failed_traces",
                        "trace_memory",
                    }:
                        continue

                    if not inside_previous_trace_memory:
                        if key in {"attempt", "attempt_number", "current_attempt"} and isinstance(value, int):
                            attempt_numbers.append(value)

                        if key in {"selected_tools", "tools", "tool_names"} and isinstance(value, list):
                            for tool in value:
                                if isinstance(tool, str):
                                    selected_tools.add(tool)

                        if key in {"tool_name", "selected_tool"} and isinstance(value, str):
                            selected_tools.add(value)

                        if isinstance(value, str) and "recovery" in value.lower():
                            recovery_used = True

                    walk(value, inside_previous_trace_memory=inside_previous_trace_memory)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item, inside_previous_trace_memory=inside_previous_trace_memory)

        walk(trace_data)

        if attempt_numbers:
            metadata["attempts"] = max(attempt_numbers)

        metadata["selected_tools"] = sorted(selected_tools)
        metadata["recovery_used"] = recovery_used

        return metadata
