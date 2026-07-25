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
from learning.supports.agentic_tutoring.grounded_answer_guard import GroundedAnswerGuard


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
        self.grounding_guard = GroundedAnswerGuard()

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

        deterministic_output_size_answer = (
            self._render_verified_output_size_explanation(
                student_question=student_question,
                package_dict=package_dict,
            )
        )

        if deterministic_output_size_answer:
            return deterministic_output_size_answer

        deterministic_calculation_answer = (
            self._render_verified_scalar_calculation(
                student_question=student_question,
                package_dict=package_dict,
            )
        )

        if deterministic_calculation_answer:
            return deterministic_calculation_answer
        scratchpad_dict = self._to_dict(scratchpad or [])
        scratchpad_summary = self._compact_scratchpad_summary(scratchpad_dict)

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
                "recent_memory": self._truncate(dpm_context.get("recent_memory"), 500),
                "tutoring_memory": self._truncate(dpm_context.get("tutoring_memory"), 500),
            },
            "static_knowledge_grounding": {
                "kb_name": skg_context.get("kb_name"),
                "retrieved_snippets": self._truncate_list(skg_context.get("snippets", []), max_items=3, max_chars=700),
                "sources": skg_context.get("sources", [])[:3],
            },
            "validated_output_package": {
                "status": package_dict.get("status"),
                "content": self._truncate(package_dict.get("content"), 2500),
                "evidence": self._truncate_list(package_dict.get("evidence", []), max_items=4, max_chars=700),
                "references": package_dict.get("references", [])[:4],
                "validation_report": package_dict.get("validation_report", {}),
                "trace_id": package_dict.get("trace_id"),
            },
            "scratchpad_summary": scratchpad_summary,
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
6. Include exactly one short "Evidence used" section.
7. Include exactly one short "References" section using only validated_output_package.references.
8. Do not add extra references from Static Knowledge Grounding if they duplicate or are not in validated_output_package.references.
9. Keep the answer clear, pedagogical, and suitable for a tutoring platform.
10. For numerical, matrix, code, or formula results, use only results already present in the validated OutputPackage.
11. Do not invent extra formulas, matrix values, code blocks, code outputs, or execution results.
12. If the student asks for code, include code only when the validated OutputPackage contains CodeSandboxTool output or executed code.
13. If no CodeSandboxTool output is present, say that executable code was not validated in the current tool step instead of inventing code.
14. If the tool output gives a verified matrix/result/stdout, state it exactly and do not recalculate it differently.
15. Do not claim that convolution output has the same dimensions as the input unless that exact result is explicitly verified in the validated OutputPackage.
16. If MatrixComputationTool or CalculatorTool did not validate output dimensions, do not invent an output matrix shape.
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
                temperature=0.0,
                max_tokens=900,
            )
        )

        answer = response.completion.strip()

        return self.grounding_guard.validate_or_fallback(
            answer=answer,
            student_question=student_question,
            topic=topic,
            package_dict=package_dict,
            dpm_context=dpm_context,
        )

    def _render_verified_output_size_explanation(
        self,
        student_question: str,
        package_dict: Dict[str, Any],
    ) -> Optional[str]:
        package_metadata = (
            package_dict.get("metadata", {})
            or {}
        )
        tool_metadata = (
            package_metadata.get(
                "tool_metadata",
                {},
            )
            or {}
        )

        calculator_entries = (
            tool_metadata.get(
                "CalculatorTool",
                [],
            )
            or []
        )

        if isinstance(calculator_entries, dict):
            calculator_entries = [
                calculator_entries
            ]

        verified_entry = None

        for entry in calculator_entries:
            if (
                isinstance(entry, dict)
                and entry.get("operation")
                == "convolution_output_size"
                and entry.get("formula")
                and "output_size" in entry
            ):
                verified_entry = entry
                break

        if not verified_entry:
            return None

        formula = verified_entry["formula"]
        input_size = verified_entry.get("input_size")
        kernel_size = verified_entry.get("kernel_size")
        padding = verified_entry.get("padding")
        stride = verified_entry.get("stride")
        output_size = verified_entry.get("output_size")

        evidence = (
            package_dict.get("evidence", [])
            or []
        )
        references = (
            package_dict.get("references", [])
            or []
        )

        evidence_lines = "\n".join(
            f"- {item}"
            for item in evidence[:5]
        ) or "- CalculatorTool verified the formula and result."

        reference_lines = "\n".join(
            f"- {item}"
            for item in references[:4]
        ) or "- CalculatorTool"

        return (
            "**Convolution output size explained again**\n\n"
            "The important correction is that a valid convolution "
            "does not automatically preserve the input size. The "
            "output depends on the input size, kernel, padding, and "
            "stride.\n\n"
            "**Formula**\n\n"
            f"`{formula}`\n\n"
            "Where:\n\n"
            "- `N` is the input size.\n"
            "- `K` is the kernel size.\n"
            "- `P` is the padding.\n"
            "- `S` is the stride.\n\n"
            "**Verified example**\n\n"
            f"- Input size: `N = {input_size}`\n"
            f"- Kernel size: `K = {kernel_size}`\n"
            f"- Padding: `P = {padding}`\n"
            f"- Stride: `S = {stride}`\n\n"
            f"`O = floor(({input_size} + 2({padding}) - "
            f"{kernel_size}) / {stride}) + 1`\n\n"
            f"`O = {output_size}`\n\n"
            f"Therefore, the verified spatial output is "
            f"**{output_size} × {output_size}**.\n\n"
            "A common mistake is to assume that the output keeps the "
            "same size as the input. That happens only under specific "
            "padding and stride settings.\n\n"
            "**Evidence used**\n\n"
            f"{evidence_lines}\n\n"
            "**References**\n\n"
            f"{reference_lines}"
        )

    def _render_verified_scalar_calculation(
        self,
        student_question: str,
        package_dict: Dict[str, Any],
    ) -> Optional[str]:
        package_metadata = (
            package_dict.get("metadata", {})
            or {}
        )
        tool_metadata = (
            package_metadata.get(
                "tool_metadata",
                {},
            )
            or {}
        )
        calculator_entries = (
            tool_metadata.get(
                "CalculatorTool",
                [],
            )
            or []
        )

        if isinstance(calculator_entries, dict):
            calculator_entries = [
                calculator_entries
            ]

        verified_entry = None

        for entry in calculator_entries:
            if (
                isinstance(entry, dict)
                and entry.get("operation")
                == "arithmetic"
                and entry.get("expression")
                and "result" in entry
            ):
                verified_entry = entry
                break

        if not verified_entry:
            return None

        expression = str(
            verified_entry["expression"]
        )
        result = verified_entry["result"]

        display_expression = (
            expression.replace("*", "×")
        )

        evidence = (
            package_dict.get("evidence", [])
            or []
        )
        references = (
            package_dict.get("references", [])
            or []
        )

        evidence_line = (
            str(evidence[0])
            if evidence
            else (
                "CalculatorTool verified the "
                "arithmetic operation."
            )
        )
        reference_line = (
            str(references[0])
            if references
            else "CalculatorTool"
        )

        return (
            "**Verified calculation**\n\n"
            f"{display_expression} = **{result}**.\n\n"
            "The result was computed by the calculator "
            "rather than estimated by the language model.\n\n"
            "**Evidence used**\n\n"
            f"- {evidence_line}\n\n"
            "**References**\n\n"
            f"- {reference_line}"
        )

    def _truncate(self, value: Any, max_chars: int) -> Any:
        if value is None:
            return None

        text = str(value)

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "... [truncated]"

    def _truncate_list(
        self,
        values: Any,
        max_items: int = 4,
        max_chars: int = 700,
    ) -> List[Any]:
        if not isinstance(values, list):
            return []

        compact = []

        for item in values[:max_items]:
            compact.append(self._truncate(item, max_chars))

        return compact

    def _compact_scratchpad_summary(self, scratchpad: Any) -> List[Dict[str, Any]]:
        if not isinstance(scratchpad, list):
            return []

        summary = []

        for item in scratchpad:
            if not isinstance(item, dict):
                continue

            package = item.get("output_package") or item.get("package") or {}
            package = package if isinstance(package, dict) else {}

            validation = package.get("validation_report", {}) or {}
            metadata = package.get("metadata", {}) or {}

            summary.append(
                {
                    "step_id": item.get("step_id"),
                    "step_goal": item.get("step_goal") or item.get("goal"),
                    "status": item.get("status") or package.get("status"),
                    "task_type": metadata.get("task_type"),
                    "tool_names": metadata.get("tool_names", []),
                    "confidence_score": validation.get("confidence_score"),
                    "failure_reason": validation.get("failure_reason"),
                    "trace_id": package.get("trace_id"),
                }
            )

        return summary

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
