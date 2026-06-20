"""OpenAI-compatible LLM transport for OpenTutorAI.

This is provider-neutral. It works with any provider that exposes an
OpenAI-compatible /chat/completions endpoint, such as:
- Groq
- OpenRouter
- OpenAI
- LMStudio
- Ollama /v1-compatible endpoint
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from ai.llm.schemas import LLMRequest, LLMResponse
from ai.llm.transports.base import LLMTransport
from ai.providers.proxy import proxy_json


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


class OpenAICompatibleTransport(LLMTransport):
    """Transport that calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        self.base_url = (
            base_url
            or os.getenv("LLM_API_BASE_URL")
            or os.getenv("OPENAI_API_BASE_URL")
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")

        self.api_key = (
            api_key
            if api_key is not None
            else (
                os.getenv("LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("GROQ_API_KEY")
                or ""
            )
        )

        self.default_model = (
            default_model
            or os.getenv("AGENTIC_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "llama-3.1-8b-instant"
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model

        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature

        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens

        if request.top_p is not None:
            body["top_p"] = request.top_p

        data = await proxy_json(
            base_url=self.base_url,
            key=self.api_key,
            method="POST",
            path="chat/completions",
            body=body,
            timeout=60.0,
        )

        choices: List[Dict[str, Any]] = data.get("choices", [])

        if not choices:
            return LLMResponse(
                model=data.get("model", model),
                completion="",
                stop_reason="no_choices",
                usage=data.get("usage"),
            )

        first_choice = choices[0]
        message = first_choice.get("message", {}) or {}
        completion = message.get("content", "") or ""

        return LLMResponse(
            model=data.get("model", model),
            completion=completion,
            stop_reason=first_choice.get("finish_reason"),
            usage=data.get("usage"),
        )
