class FailureRecovery:
    """
    Bounded recovery controller.

    It decides whether to retry and which tools must be forced
    into the next attempt.
    """

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def decide(self, validation_report, attempt_number: int):
        failure_reason = validation_report.failure_reason or "unknown_failure"

        if attempt_number >= self.max_attempts:
            return {
                "should_retry": False,
                "action": "fallback",
                "reason": "maximum_attempts_reached",
                "failure_reason": failure_reason,
                "required_tools": [],
            }

        if failure_reason in {"insufficient_grounding", "weak_grounding"}:
            return {
                "should_retry": True,
                "action": "add_grounding_tools",
                "reason": "insufficient_grounding_detected",
                "failure_reason": failure_reason,
                "required_tools": ["RAGTool", "TraceSearchTool"],
            }

        if failure_reason == "missing_visual_support":
            return {
                "should_retry": True,
                "action": "add_visual_support",
                "reason": "visual_support_missing",
                "failure_reason": failure_reason,
                "required_tools": ["RAGTool", "TraceSearchTool", "VisualMatrixTool"],
            }

        if failure_reason == "missing_matrix_computation_support":
            return {
                "should_retry": True,
                "action": "add_matrix_computation_support",
                "reason": "matrix_computation_missing",
                "failure_reason": failure_reason,
                "required_tools": [
                    "RAGTool",
                    "TraceSearchTool",
                    "VisualMatrixTool",
                    "MatrixComputationTool",
                ],
            }

        return {
            "should_retry": True,
            "action": "re_execute_with_default_tools",
            "reason": failure_reason,
            "failure_reason": failure_reason,
            "required_tools": ["RAGTool", "TraceSearchTool"],
        }
