from typing import Any, Dict, List, Optional

from ai.agentic.memory.dynamic_personal_memory import DynamicPersonalMemory
from ai.agentic.memory.static_knowledge_grounding import StaticKnowledgeGrounding


class MockLLMAdapter:
    """
    Deterministic LLM adapter used for the research prototype.

    This is not a real external LLM call.
    It provides a clean interface that can later be replaced by a real LLM provider.

    Inputs:
    - student question
    - learner context from DPM
    - grounded knowledge from SKG
    - validated OutputPackage from the ToolInteractionManager

    Output:
    - personalized final tutoring answer
    """

    def __init__(
        self,
        dpm: Optional[DynamicPersonalMemory] = None,
        skg: Optional[StaticKnowledgeGrounding] = None,
    ):
        self.dpm = dpm or DynamicPersonalMemory()
        self.skg = skg or StaticKnowledgeGrounding()

    def generate_personalized_tutoring_answer(
        self,
        student_question: str,
        learner_id: str,
        output_package: Any,
        scratchpad: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        topic = self._infer_topic(student_question)

        package_dict = output_package.to_dict() if hasattr(output_package, "to_dict") else {}
        package_content = package_dict.get("content", "")
        package_evidence = package_dict.get("evidence", [])
        package_references = package_dict.get("references", [])

        dpm_context = self.dpm.build_personalization_context(
            learner_id=learner_id,
            topic=topic,
            task_type=package_dict.get("metadata", {}).get("task_type", "visual_explanation"),
        )

        skg_context = self.skg.build_grounding_context(
            query=student_question,
            topic=topic,
            limit=3,
        )

        learner_level = dpm_context.get("learner_level", "unknown")
        explanation_style = dpm_context.get("preferred_explanation_style", "unknown")
        preferred_examples = dpm_context.get("preferred_examples", [])
        is_weak_topic = dpm_context.get("is_weak_topic", False)

        skg_snippets = skg_context.get("snippets", [])
        grounding_points = [
            snippet.get("content", "")
            for snippet in skg_snippets
            if snippet.get("content")
        ]

        answer_parts = []

        answer_parts.append("Personalized tutoring answer:")
        answer_parts.append("")
        answer_parts.append(f"Question: {student_question}")
        answer_parts.append("")

        answer_parts.append("Because your current learner profile is marked as "
                            f"{learner_level}, I will explain it in a simple step-by-step way.")

        if is_weak_topic:
            answer_parts.append(
                f"This topic appears in your weak topics, so I will use a concrete example before giving a formal explanation."
            )

        if preferred_examples:
            answer_parts.append(
                "I will use your preferred example style: "
                + ", ".join(preferred_examples)
                + "."
            )

        answer_parts.append("")
        answer_parts.append("1. Main idea")
        if grounding_points:
            answer_parts.append(grounding_points[0])
        else:
            answer_parts.append("Convolution means applying a small filter over parts of an input.")

        answer_parts.append("")
        answer_parts.append("2. How it works")
        if len(grounding_points) > 1:
            answer_parts.append(grounding_points[1])
        else:
            answer_parts.append(
                "The kernel looks at one small region at a time, multiplies matching values, and sums them."
            )

        answer_parts.append("")
        answer_parts.append("3. Visual intuition")
        visual_point = self._find_visual_point(grounding_points)
        if visual_point:
            answer_parts.append(visual_point)
        else:
            answer_parts.append(
                "Imagine a 2x2 kernel sliding over a 3x3 matrix. Each position creates one output value."
            )

        answer_parts.append("")
        answer_parts.append("4. Validated tool support")
        if package_content:
            answer_parts.append(
                "The centralized Tool Interaction Manager validated the following support:"
            )
            answer_parts.append(package_content)
        else:
            answer_parts.append("No validated package content was available.")

        if package_evidence:
            answer_parts.append("")
            answer_parts.append("Evidence used:")
            for idx, evidence in enumerate(package_evidence[:4], start=1):
                answer_parts.append(f"- Evidence {idx}: {evidence}")

        if package_references:
            unique_refs = []
            for ref in package_references:
                if ref and ref not in unique_refs:
                    unique_refs.append(ref)

            answer_parts.append("")
            answer_parts.append("References:")
            for ref in unique_refs[:4]:
                answer_parts.append(f"- {ref}")

        answer_parts.append("")
        answer_parts.append(
            "In short: convolution slides a kernel over the input, combines each local patch with the kernel, "
            "and produces output values that describe local patterns."
        )

        return "\n".join(answer_parts)

    def _infer_topic(self, text: str) -> str:
        lower_text = text.lower()

        if "convolution" in lower_text or "kernel" in lower_text:
            return "convolution"

        if "overfitting" in lower_text:
            return "overfitting"

        if "gradient" in lower_text:
            return "gradient descent"

        return "general topic"

    def _find_visual_point(self, grounding_points: List[str]) -> Optional[str]:
        """
        Prefer the most concrete visual/matrix example.
        """
        priority_terms = ["2x2", "3x3", "visual explanation", "matrix", "local patches"]

        for term in priority_terms:
            for point in grounding_points:
                if term in point.lower():
                    return point

        for point in grounding_points:
            lower = point.lower()
            if "visual" in lower or "sliding" in lower:
                return point

        return None
