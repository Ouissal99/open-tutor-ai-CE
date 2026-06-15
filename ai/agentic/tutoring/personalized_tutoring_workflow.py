from ai.agentic.tutoring.scratchpad import StepBasedScratchpad
from ai.agentic.tutoring.tool_request_agent import ToolRequestAgent
from ai.agentic.tool_interaction.manager import ToolInteractionManager


class PersonalizedTutoringWorkflow:
    """
    DeepTutor-inspired left-side workflow.
    For the demo, it only implements a minimal vertical slice.
    """

    def __init__(self):
        self.scratchpad = StepBasedScratchpad()
        self.tool_request_agent = ToolRequestAgent()
        self.tool_manager = ToolInteractionManager()

    def run(self, student_question: str):
        print("\n[1] Student question received")
        print(f"    {student_question}")

        print("\n[2] Creating simplified solving plan")
        solving_plan = [
            {
                "step_goal": "Explain the concept using course-grounded evidence",
                "needs_tool": False
            },
            {
                "step_goal": "Generate visual support for kernel movement",
                "needs_tool": True
            }
        ]

        final_package = None
        trace_path = None

        for step in solving_plan:
            round_item = self.scratchpad.add_round(
                step_goal=step["step_goal"],
                analysis="The tutor prepares the current solving step."
            )

            if step["needs_tool"]:
                print("\n[3] Tool Request Agent creates a structured ToolRequest")
                request = self.tool_request_agent.create_request(
                    student_question=student_question,
                    step_goal=step["step_goal"]
                )
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
                    package=package
                )
                print("\n[6] Scratchpad updated with validated package")

        final_answer = self._compose_answer(student_question, final_package)

        return {
            "final_answer": final_answer,
            "scratchpad": self.scratchpad.to_dict(),
            "output_package": final_package.to_dict() if final_package else None,
            "trace_path": trace_path
        }

    def _compose_answer(self, question, package):
        if not package:
            return "No tool package was generated."

        return (
            "Final personalized answer draft:\n"
            f"For the question: '{question}', the system retrieved grounded support about convolution. "
            "A convolution kernel slides over local regions of an input matrix. "
            "At each position, the system multiplies the kernel values with the corresponding input patch, "
            "then sums the products to produce one output value. "
            "The validated package also includes visual support for kernel movement."
        )
