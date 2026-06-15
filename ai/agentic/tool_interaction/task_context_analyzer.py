class TaskContextAnalyzer:
    """
    Simplified first version.
    Later this will connect to OpenTutorAI sessions, RAG context, and learner memory.
    """

    def analyze(self, request):
        topic = "convolution" if "convolution" in request.user_query.lower() else "general topic"

        return {
            "request_id": request.request_id,
            "workflow_source": request.workflow_source,
            "task_type": request.task_type,
            "topic": topic,
            "current_step": request.current_step,
            "expected_output": request.expected_output,
            "learner_level": request.context.get("learner_level", "beginner"),
            "needs_visual_support": "visual" in request.expected_output.lower()
        }
