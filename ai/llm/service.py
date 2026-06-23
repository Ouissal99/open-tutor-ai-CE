"""LLM service for coordinating completions with provider retry handling."""

import asyncio
import os

from ai.llm.errors import LLMProviderUnavailableError
from ai.llm.schemas import LLMRequest, LLMResponse
from ai.llm.transports.base import LLMTransport


class LLMService:
    """Service for LLM operations.

    Strict provider mode:
    - The configured LLM provider must complete the LLM stage.
    - If the provider fails, retry a limited number of times.
    - If all attempts fail, raise a clean provider-unavailable error.
    - Do not generate a fake or deterministic LLM answer.
    """

    def __init__(self, transport: LLMTransport):
        self.transport = transport
        self.max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
        self.retry_delay_seconds = float(os.getenv("LLM_RETRY_DELAY_SECONDS", "2.0"))

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Get completion from the configured LLM provider."""
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await self.transport.complete(request)
            except LLMProviderUnavailableError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc

            if attempt < self.max_attempts:
                await asyncio.sleep(self.retry_delay_seconds * attempt)

        raise LLMProviderUnavailableError(
            "The configured Groq/OpenAI-compatible LLM provider is currently "
            f"unavailable after {self.max_attempts} attempt(s). "
            "Please check your internet connection, API key, provider status, "
            "or try again later.",
            original_error=last_error,
        )
