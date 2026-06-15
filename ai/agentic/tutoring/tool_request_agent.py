from ai.agentic.core.schemas import ToolRequest, new_id


class ToolRequestAgent:
    """
    Lightweight Tool Request Agent / Tool Proxy.
    It does not execute tools locally.
    It sends structured requests to the centralized Tool Interaction Manager.
    """

    def create_request(self, student_question: str, step_goal: str, learner_level: str = "beginner"):
        return ToolRequest(
            request_id=new_id("REQ"),
            workflow_source="personalized_problem_tutoring",
            task_type="visual_explanation",
            user_query=student_question,
            current_step=step_goal,
            expected_output="visual explanation with grounded evidence",
            context={
                "learner_level": learner_level,
                "force_first_failure": True
            },
            constraints={
                "must_be_grounded": True,
                "max_attempts": 3
            }
        )
