"""Real LLM answer writer for the agentic tutoring workflow."""

import asyncio
import json
import os
import threading
from typing import Any, Dict, List, Optional

from ai.agentic.memory.dynamic_personal_memory import DynamicPersonalMemory
from ai.agentic.memory.static_knowledge_grounding import StaticKnowledgeGrounding
from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport


class TutoringAnswerWriter:
    """
    Generates the final personalized tutoring answer using a real LLM.

    OpenTutorAI-native path:
    TutoringAnswerWriter
    → LLMService
    → OpenAICompatibleTransport
    → ai.providers.proxy
    → OpenAI-compatible provider such as Groq/OpenAI/Ollama
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        dpm: Optional[DynamicPersonalMemory] = None,
        skg: Optional[StaticKnowledgeGrounding] = None,
        model: Optional[str] = None,
    ):
        self.llm_service = llm_service or LLMService(OpenAICompatibleTransport())
        self.dpm = dpm or DynamicPersonalMemory()
        self.skg = skg or StaticKnowledgeGrounding()
        self.model = model or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant"

    def generate(
        self,
        student_question: str,
        learner_id: str,
        output_package: Any,
        scratchpad: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Synchronous wrapper for the current terminal prototype.
        """
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                return self._run_async_in_thread(
                    self.generate_async(
                        student_question=student_question,
                        learner_id=learner_id,
                        output_package=output_package,
                        scratchpad=scratchpad,
                    )
                )
        except RuntimeError:
            pass

        return asyncio.run(
            self.generate_async(
                student_question=student_question,
                learner_id=learner_id,
                output_package=output_package,
                scratchpad=scratchpad,
            )
        )

    async def generate_async(
        self,
        student_question: str,
        learner_id: str,
        output_package: Any,
        scratchpad: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        topic = self._infer_topic(student_question)

        package_dict = self._to_dict(output_package)
        scratchpad_dict = self._to_dict(scratchpad or [])

        task_type = (package_dict.get("metadata", {}) or {}).get(
            "task_type",
            "visual_explanation",
        )

        dpm_context = self.dpm.build_personalization_context(
            learner_id=learner_id,
            topic=topic,
            task_type=task_type,
        )

        skg_context = self.skg.build_grounding_context(
            query=student_question,
            topic=topic,
            limit=3,
        )

        payload = {
            "student_question": student_question,
            "learner_id": learner_id,
            "topic": topic,
            "dynamic_personal_memory": {
                "learner_level": dpm_context.get("learner_level"),
                "preferred_explanation_style": dpm_context.get("preferred_explanation_style"),
                "preferred_examples": dpm_context.get("preferred_examples"),
                "avoid": dpm_context.get("avoid"),
                "known_topics": dpm_context.get("known_topics"),
                "weak_topics": dpm_context.get("weak_topics"),
                "is_weak_topic": dpm_context.get("is_weak_topic"),
                "recent_memory": dpm_context.get("recent_memory"),
                "tutoring_memory": dpm_context.get("tutoring_memory"),
            },
            "static_knowledge_grounding": {
                "kb_name": skg_context.get("kb_name"),
                "retrieved_snippets": skg_context.get("snippets", []),
                "sources": skg_context.get("sources", []),
            },
            "validated_output_package": {
                "status": package_dict.get("status"),
                "content": package_dict.get("content"),
                "evidence": package_dict.get("evidence", []),
                "references": package_dict.get("references", []),
                "validation_report": package_dict.get("validation_report", {}),
                "trace_id": package_dict.get("trace_id"),
            },
            "scratchpad": scratchpad_dict,
        }

        system_prompt = """
You are the final answer writer inside an agentic personalized tutoring system.

Use only the provided context:
- Dynamic Personal Memory for learner adaptation.
- Static Knowledge Grounding for course/domain knowledge.
- Validated OutputPackage for tool evidence.
- Scratchpad for workflow state.

Rules:
1. Do not invent sources.
2. Do not mention internal implementation words like mock, hard-coded, or fake.
3. If learner_level is beginner, explain simply and step by step.
4. If the topic is a weak topic, use an intuitive example before formal explanation.
5. Use the provided SKG snippets and validated package evidence.
6. Include a short "Evidence used" section.
7. Include a short "References" section using only provided references.
8. Keep the answer clear, pedagogical, and suitable for a tutoring platform.
""".strip()

        user_prompt = (
            "Generate the final personalized tutoring answer from this JSON context:\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        response = await self.llm_service.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                temperature=0.2,
                max_tokens=900,
            )
        )

        return response.completion.strip()

    def _infer_topic(self, text: str) -> str:
        lower = text.lower()

        if "convolution" in lower or "kernel" in lower:
            return "convolution"

        if "overfitting" in lower:
            return "overfitting"

        if "gradient" in lower:
            return "gradient descent"

        return "general topic"

    def _to_dict(self, obj: Any) -> Any:
        if obj is None:
            return {}

        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj

        if hasattr(obj, "model_dump"):
            return obj.model_dump()

        if hasattr(obj, "dict"):
            return obj.dict()

        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return str(obj)

    def _run_async_in_thread(self, coro) -> str:
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
