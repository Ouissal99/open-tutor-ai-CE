class FailureRecovery:
    """
    Bounded recovery controller.
    Stops after max_attempts.
    """

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def decide(self, validation_report, attempt_number: int):
        if attempt_number >= self.max_attempts:
            return {
                "should_retry": False,
                "action": "fallback",
                "reason": "maximum_attempts_reached"
            }

        if validation_report.failure_reason == "weak_grounding":
            return {
                "should_retry": True,
                "action": "re_query_or_re_execute",
                "reason": "weak grounding detected"
            }

        return {
            "should_retry": True,
            "action": "re_execute",
            "reason": validation_report.failure_reason or "unknown_failure"
        }
