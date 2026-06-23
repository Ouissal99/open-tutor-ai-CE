"""LLM error types."""


class LLMProviderUnavailableError(RuntimeError):
    """Raised when the configured LLM provider cannot be reached."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message)
        self.original_error = original_error
