"""LLM service for coordinating completions."""

from ai.llm.errors import LLMProviderUnavailableError
from ai.llm.schemas import LLMRequest, LLMResponse
from ai.llm.transports.base import LLMTransport


class LLMService:
    """Service for LLM operations."""

    def __init__(self, transport: LLMTransport):
        self.transport = transport

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Get completion from LLM."""
        try:
            return await self.transport.complete(request)
        except LLMProviderUnavailableError:
            raise
        except Exception as exc:
            raise LLMProviderUnavailableError(
                "The configured LLM provider is currently unavailable. "
                "Please check your internet connection, API key, provider status, "
                "or try again later.",
                original_error=exc,
            ) from exc
