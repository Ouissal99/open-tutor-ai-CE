from ai.agentic.core.schemas import ToolResult


class ToolExecutor:
    """
    Executes tools.
    In this demo, tools are mocked. Later this will call ai/tools/registry.py.
    """

    def execute(self, plan, request, attempt_number: int):
        results = []

        for step in plan.steps:
            tool_name = step["tool_name"]

            if tool_name == "mock_rag_tool":
                # Force first attempt to fail to demonstrate recovery.
                if request.context.get("force_first_failure") and attempt_number == 1:
                    results.append(ToolResult(
                        tool_name=tool_name,
                        success=True,
                        output="Weak retrieved content: not enough grounding.",
                        evidence=[],
                        references=[]
                    ))
                else:
                    results.append(ToolResult(
                        tool_name=tool_name,
                        success=True,
                        output="Course evidence retrieved about convolution and kernel movement.",
                        evidence=[
                            "Convolution applies a kernel over an input matrix by sliding it over local regions.",
                            "Each output value is obtained by multiplying kernel values with the corresponding input patch and summing the result."
                        ],
                        references=["OpenTutorAI mock lesson: Image Processing / Convolution"]
                    ))

            elif tool_name == "mock_visualizer_tool":
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=True,
                    output="Generated visual explanation: a 2x2 kernel slides over a 3x3 matrix.",
                    evidence=["Visualizer output describes kernel movement step by step."],
                    references=["Mock visualizer output"]
                ))

            elif tool_name == "mock_validator_support":
                results.append(ToolResult(
                    tool_name=tool_name,
                    success=True,
                    output="Validator support tool ready.",
                    evidence=[],
                    references=[]
                ))

        return results
