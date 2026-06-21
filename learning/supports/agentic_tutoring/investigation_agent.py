"""LLM-based investigation agent for personalized tutoring.

This agent performs the first DeepTutor-style step:
- understand the student question
- decompose it into meta-questions
- identify required context
- create an initial tutoring plan

It uses OpenTutorAI's LLM layer:
InvestigationAgent
→ LLMService
→ OpenAICompatibleTransport
→ provider such as Groq/OpenAI/Ollama
"""

import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, Optional

from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport


class InvestigationAgent:
    """LLM-powered investigation agent for the tutoring workflow."""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        model: Optional[str] = None,
    ):
        self.llm_service = llm_service or LLMService(OpenAICompatibleTransport())
        self.model = model or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant"

    def investigate(
        self,
        student_question: str,
        learner_id: str = "demo_user",
    ) -> Dict[str, Any]:
        """Synchronous wrapper for the current terminal prototype."""
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                return self._run_async_in_thread(
                    self.investigate_async(
                        student_question=student_question,
                        learner_id=learner_id,
                    )
                )
        except RuntimeError:
            pass

        return asyncio.run(
            self.investigate_async(
                student_question=student_question,
                learner_id=learner_id,
            )
        )

    async def investigate_async(
        self,
        student_question: str,
        learner_id: str = "demo_user",
    ) -> Dict[str, Any]:
        system_prompt = """
You are an InvestigationAgent inside a personalized agentic tutoring system inspired by DeepTutor.

Your job is NOT to answer the student directly.
Your job is to analyze the student question and prepare a tutoring investigation plan.

Return ONLY valid JSON with this exact schema:
{
  "topic": "short topic name",
  "task_type": "visual_explanation | conceptual_explanation | problem_solving | code_help | general_tutoring",
  "student_intent": "what the student wants",
  "difficulty_estimate": "beginner | intermediate | advanced",
  "meta_questions": [
    "question the tutor should investigate first",
    "question the tutor should investigate next"
  ],
  "required_context": [
    "DPM",
    "SKG",
    "TraceToolkit"
  ],
  "initial_tutoring_plan": [
    {
      "step_id": 1,
      "step_goal": "clear tutoring step",
      "needs_tool": true,
      "suggested_tools": ["RAGTool", "TraceSearchTool"],
      "reason": "why this step is needed"
    }
  ],
  "tool_request_goal": "short goal to send to the centralized ToolInteractionManager"
}

Rules:
- Always include DPM, SKG, and TraceToolkit in required_context.
- Use the exact key name initial_tutoring_plan, not initial_tutorial_plan.
- If the student asks for a simple example, include a concrete or visual example step.
- If the topic is convolution, prefer a small matrix/kernel example.
- For course-grounded evidence, suggest RAGTool.
- For previous trace guidance, suggest TraceSearchTool.
- For visual matrix/kernel examples, suggest VisualMatrixTool.
- Do not answer the student.
- Do not write markdown.
- Return JSON only.
""".strip()

        user_prompt = {
            "learner_id": learner_id,
            "student_question": student_question,
            "instruction": "Investigate this tutoring request and create a plan.",
        }

        response = await self.llm_service.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=json.dumps(user_prompt, ensure_ascii=False)),
                ],
                temperature=0.1,
                max_tokens=900,
            )
        )

        raw = response.completion.strip()

        parsed = self._parse_json_or_fallback(raw, student_question)
        parsed = self._normalize_result(parsed, student_question)

        parsed["learner_id"] = learner_id
        parsed["student_question"] = student_question
        parsed["llm_model"] = response.model
        parsed["raw_response"] = raw

        return parsed

    def _normalize_result(self, result: Dict[str, Any], student_question: str) -> Dict[str, Any]:
        """Normalize real LLM output so the workflow receives stable fields."""
        if not isinstance(result, dict):
            result = {}

        topic = str(result.get("topic") or self._infer_topic(student_question)).strip().lower()
        result["topic"] = topic

        allowed_task_types = {
            "visual_explanation",
            "conceptual_explanation",
            "problem_solving",
            "code_help",
            "general_tutoring",
        }

        task_type = result.get("task_type")
        if task_type not in allowed_task_types:
            if topic == "convolution" or "example" in student_question.lower():
                task_type = "visual_explanation"
            else:
                task_type = "general_tutoring"

        result["task_type"] = task_type

        result.setdefault(
            "student_intent",
            "The student wants a clear personalized tutoring explanation.",
        )

        result.setdefault("difficulty_estimate", "beginner")

        required_context = result.get("required_context") or []
        if not isinstance(required_context, list):
            required_context = []

        # Keep only supported context systems for this workflow.
        allowed_context = ["DPM", "SKG", "TraceToolkit"]
        normalized_context = []

        for source in allowed_context:
            if source in required_context or source not in normalized_context:
                normalized_context.append(source)

        result["required_context"] = normalized_context

        # Fix common LLM schema drift.
        # Some models return initial_tutorial_plan instead of initial_tutoring_plan.
        if "initial_tutoring_plan" not in result and "initial_tutorial_plan" in result:
            result["initial_tutoring_plan"] = result.pop("initial_tutorial_plan")

        plan = result.get("initial_tutoring_plan") or []
        if not isinstance(plan, list):
            plan = []

        normalized_plan = []

        for index, step in enumerate(plan, start=1):
            if not isinstance(step, dict):
                continue

            step_goal = step.get("step_goal") or f"Tutoring step {index}"
            suggested_tools = step.get("suggested_tools") or []

            if not isinstance(suggested_tools, list):
                suggested_tools = [str(suggested_tools)]

            step_goal_lower = step_goal.lower()

            # Evidence/definition/explanation steps need RAG.
            if any(word in step_goal_lower for word in ["define", "explain", "evidence", "course", "grounded"]):
                if "RAGTool" not in suggested_tools:
                    suggested_tools.append("RAGTool")

            # Visual/matrix/kernel/example steps need the visual tool.
            if any(word in step_goal_lower for word in ["visual", "matrix", "kernel", "example", "slide"]):
                if "VisualMatrixTool" not in suggested_tools:
                    suggested_tools.append("VisualMatrixTool")

            # Personalized agentic workflow can use traces when tools are needed.
            if suggested_tools and "TraceSearchTool" not in suggested_tools:
                suggested_tools.append("TraceSearchTool")

            normalized_plan.append(
                {
                    "step_id": step.get("step_id", index),
                    "step_goal": step_goal,
                    "needs_tool": bool(suggested_tools) or bool(step.get("needs_tool", False)),
                    "suggested_tools": suggested_tools,
                    "reason": step.get("reason", "Needed for personalized tutoring."),
                }
            )

        # Strong default for convolution/simple-example requests.
        if topic == "convolution":
            all_tools = {
                tool
                for step in normalized_plan
                for tool in step.get("suggested_tools", [])
            }

            if "RAGTool" not in all_tools:
                normalized_plan.insert(
                    0,
                    {
                        "step_id": 1,
                        "step_goal": "Retrieve course-grounded evidence about convolution",
                        "needs_tool": True,
                        "suggested_tools": ["RAGTool", "TraceSearchTool"],
                        "reason": "The explanation should be grounded in SKG and previous traces.",
                    },
                )

            if "VisualMatrixTool" not in all_tools:
                normalized_plan.append(
                    {
                        "step_id": len(normalized_plan) + 1,
                        "step_goal": "Generate visual matrix support for kernel movement",
                        "needs_tool": True,
                        "suggested_tools": ["VisualMatrixTool", "TraceSearchTool"],
                        "reason": "A small matrix example helps beginner understanding.",
                    },
                )

        if not normalized_plan:
            normalized_plan = [
                {
                    "step_id": 1,
                    "step_goal": "Retrieve grounded tutoring context",
                    "needs_tool": True,
                    "suggested_tools": ["RAGTool", "TraceSearchTool"],
                    "reason": "The answer should use course knowledge and learner history.",
                },
                {
                    "step_id": 2,
                    "step_goal": "Generate a concrete beginner-friendly example",
                    "needs_tool": True,
                    "suggested_tools": ["VisualMatrixTool"],
                    "reason": "The learner requested a simple explanation.",
                },
            ]

        for index, step in enumerate(normalized_plan, start=1):
            step["step_id"] = index

        result["initial_tutoring_plan"] = normalized_plan

        meta_questions = result.get("meta_questions") or []
        if not isinstance(meta_questions, list) or not meta_questions:
            meta_questions = [
                f"What is the core idea of {topic}?",
                "What simple example can make it understandable?",
                "What learner context should influence the answer?",
            ]

        result["meta_questions"] = meta_questions

        result.setdefault(
            "tool_request_goal",
            "Generate a personalized explanation with grounded evidence and a simple example.",
        )

        return result

    def _parse_json_or_fallback(self, raw: str, student_question: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Sometimes models wrap JSON in ```json fences.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        topic = self._infer_topic(student_question)

        return {
            "topic": topic,
            "task_type": "visual_explanation" if topic == "convolution" else "general_tutoring",
            "student_intent": "The student wants a clear personalized explanation.",
            "difficulty_estimate": "beginner",
            "meta_questions": [
                f"What is the core concept of {topic}?",
                "What simple example can make it understandable?",
                "What prior learner context should influence the explanation?",
            ],
            "required_context": ["DPM", "SKG", "TraceToolkit"],
            "initial_tutoring_plan": [
                {
                    "step_id": 1,
                    "step_goal": "Retrieve course-grounded evidence",
                    "needs_tool": True,
                    "suggested_tools": ["RAGTool", "TraceSearchTool"],
                    "reason": "The answer should be grounded in SKG and previous traces.",
                },
                {
                    "step_id": 2,
                    "step_goal": "Generate visual support for kernel movement",
                    "needs_tool": True,
                    "suggested_tools": ["VisualMatrixTool"],
                    "reason": "The learner asked for a simple example.",
                },
            ],
            "tool_request_goal": "Generate a personalized explanation with grounded evidence and a simple example.",
            "fallback_used": True,
        }

    def _infer_topic(self, text: str) -> str:
        lower = text.lower()

        if "convolution" in lower or "kernel" in lower:
            return "convolution"

        if "overfitting" in lower:
            return "overfitting"

        if "gradient" in lower:
            return "gradient descent"

        if "matrix" in lower:
            return "matrices"

        return "general topic"

    def _run_async_in_thread(self, coro) -> Dict[str, Any]:
        result = {"value": None, "error": None}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if result["error"]:
            raise result["error"]

        return result["value"]
