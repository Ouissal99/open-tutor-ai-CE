"""CodeDraftAgent for LLM-generated executable code snippets.

This is an internal Tool Interaction Manager component, not an external tool.
It drafts small safe Python snippets that are then executed by CodeSandboxTool.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, Optional

from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport


class CodeDraftAgent:
    """
    Generates candidate Python code using the configured LLM provider.

    The generated code is not trusted directly. It must be executed by
    CodeSandboxTool and validated by OutputValidator before it can be used
    in the final tutoring answer.
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        model: Optional[str] = None,
    ):
        self.llm_service = llm_service or LLMService(OpenAICompatibleTransport())
        self.model = model or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant"

    def generate(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> Dict[str, Any]:
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                return self._run_async_in_thread(
                    self.generate_async(
                        request=request,
                        step=step,
                        collected_context=collected_context,
                        analyzed_task=analyzed_task,
                        attempt=attempt,
                    )
                )
        except RuntimeError:
            pass

        return asyncio.run(
            self.generate_async(
                request=request,
                step=step,
                collected_context=collected_context,
                analyzed_task=analyzed_task,
                attempt=attempt,
            )
        )

    async def generate_async(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> Dict[str, Any]:
        payload = self._build_payload(
            request=request,
            step=step,
            collected_context=collected_context,
            analyzed_task=analyzed_task,
            attempt=attempt,
        )

        system_prompt = """
You are CodeDraftAgent inside a centralized Tool Interaction Manager.

Your task:
Generate a small executable Python snippet for an educational tutoring request.

Return ONLY valid JSON with this schema:
{
  "code": "...",
  "explanation": "...",
  "assumptions": ["..."],
  "safety_notes": ["..."]
}

Rules:
1. The code must be executable by a restricted educational sandbox.
2. The code must print its result\n
MATRIX_SHAPE_SAFETY_RULES:
- If the student request specifies matrix and kernel dimensions, use exactly those dimensions.
- Do not replace a requested 2x2 kernel with a 3x3 kernel.
- For valid convolution or cross-correlation, output shape is:
  output_rows = matrix_rows - kernel_rows + 1
  output_cols = matrix_cols - kernel_cols + 1
- Iterate only over output_rows and output_cols.
- Each extracted patch must have the same shape as the kernel before multiplication.
- If a previous error mentions broadcasting, operands, or shape mismatch, fix the dimensions before retrying.
.
3. Prefer short, deterministic examples.
4. Do not use file operations, network operations, subprocesses, operating-system access, eval, exec, input, or dynamic imports.
5. Use only safe beginner-friendly code.
6. You may use pure Python. If the student specifically asks for NumPy, you may use `import numpy as np`.
7. Do not include markdown fences in the JSON value.
8. If this is a retry, use the previous error information to correct the code.
9. The code is only a candidate. It will be verified by CodeSandboxTool before final answer generation.
""".strip()

        user_prompt = (
            "Generate candidate executable code from this manager context:\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        response = await self.llm_service.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                temperature=0.0,
                max_tokens=900,
            )
        )

        parsed = self._parse_response(response.completion)

        code = (parsed.get("code") or "").strip()
        if not code:
            raise ValueError("CodeDraftAgent did not return executable code.")

        return {
            "code": code,
            "explanation": parsed.get("explanation", ""),
            "assumptions": parsed.get("assumptions", []),
            "safety_notes": parsed.get("safety_notes", []),
            "attempt": attempt,
            "model": response.model,
            "source_component": "CodeDraftAgent",
            "generation_strategy": "llm_code_drafting",
        }

    def _build_payload(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int,
    ) -> Dict[str, Any]:
        dpm = collected_context.get("dpm", {}) or {}
        skg = collected_context.get("skg", {}) or {}
        trace_memory = collected_context.get("trace_memory", {}) or {}

        return {
            "attempt": attempt,
            "student_question": (
                analyzed_task.get("student_question")
                or getattr(request, "student_question", None)
                or getattr(request, "user_query", None)
                or ""
            ),
            "task_type": analyzed_task.get("task_type"),
            "topic": analyzed_task.get("topic"),
            "goal": analyzed_task.get("goal") or analyzed_task.get("tool_request_goal"),
            "current_step": analyzed_task.get("current_step"),
            "expected_output": analyzed_task.get("expected_output"),
            "code_generation_instruction": (
                step.get("code_generation_instruction")
                or analyzed_task.get("code_generation_instruction")
                or "Generate a minimal safe Python snippet and print the result."
            ),
            "learner_profile": {
                "learner_level": dpm.get("learner_level"),
                "preferred_explanation_style": dpm.get("preferred_explanation_style"),
                "preferred_examples": dpm.get("preferred_examples"),
                "weak_topics": dpm.get("weak_topics"),
                "is_weak_topic": dpm.get("is_weak_topic"),
            },
            "skg_grounding": {
                "snippets": (skg.get("snippets") or [])[:3],
                "sources": (skg.get("sources") or [])[:3],
            },
            "previous_tool_traces": {
                "similar_successful_traces": trace_memory.get("similar_successful_traces", [])[:3],
                "failed_traces": trace_memory.get("failed_traces", [])[:3],
            },
            "previous_code_execution_errors": analyzed_task.get("previous_code_execution_errors", []),
            "recovery_reason": analyzed_task.get("recovery_reason"),
        }

    def _parse_response(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()

        try:
            return self._parse_json_object(text)
        except Exception:
            code = self._extract_code_block(text)
            if code:
                return {
                    "code": code,
                    "explanation": "Code extracted from LLM response.",
                    "assumptions": [],
                    "safety_notes": [],
                }
            raise

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM code draft response was not a JSON object.")
        return data

    def _extract_code_block(self, text: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _run_async_in_thread(self, coroutine):
        result_box = {}
        error_box = {}

        def runner():
            try:
                result_box["result"] = asyncio.run(coroutine)
            except Exception as exc:
                error_box["error"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if "error" in error_box:
            raise error_box["error"]

        return result_box.get("result")
