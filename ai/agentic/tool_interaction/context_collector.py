from typing import Any, Dict, Optional

from ai.agentic.memory.dynamic_personal_memory import DynamicPersonalMemory
from ai.agentic.memory.static_knowledge_grounding import StaticKnowledgeGrounding
from ai.agentic.memory.trace_toolkit import TraceToolkit


class ContextCollector:
    """
    Collects the personalization and grounding context required by the
    centralized Tool Interaction Manager.

    DeepTutor-inspired role:
    - DPM supplies learner context.
    - SKG supplies course/domain knowledge.
    - TraceToolkit supplies previous tool-interaction experience.

    This component creates the enriched context used by:
    - ToolSelector
    - ToolPlanner
    - ToolExecutor
    - OutputPackageBuilder
    """

    def __init__(
        self,
        dpm: Optional[DynamicPersonalMemory] = None,
        skg: Optional[StaticKnowledgeGrounding] = None,
        trace_toolkit: Optional[TraceToolkit] = None,
    ):
        self.dpm = dpm or DynamicPersonalMemory()
        self.skg = skg or StaticKnowledgeGrounding()
        self.trace_toolkit = trace_toolkit or TraceToolkit()

    def collect(self, request: Any, analyzed_task: Dict[str, Any]) -> Dict[str, Any]:
        learner_id = self._get_learner_id(request)
        student_question = self._get_student_question(request)
        task_type = analyzed_task.get("task_type", getattr(request, "task_type", "unknown_task"))
        topic = analyzed_task.get("topic", self._infer_topic(student_question))
        current_step = analyzed_task.get("current_step", getattr(request, "current_step", ""))

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

        similar_successful_traces = self.trace_toolkit.find_similar_traces(
            query=student_question,
            task_type=task_type,
            limit=3,
            only_successful=True,
            learner_id=learner_id,
        )

        failed_traces = self.trace_toolkit.find_failed_traces(
            task_type=task_type,
            limit=3,
            learner_id=learner_id,
        )

        return {
            "learner_id": learner_id,
            "student_question": student_question,
            "task_type": task_type,
            "topic": topic,
            "current_step": current_step,
            "dpm": dpm_context,
            "skg": skg_context,
            "trace_memory": {
                "similar_successful_traces": similar_successful_traces,
                "failed_traces": failed_traces,
            },
            "summary": {
                "learner_level": dpm_context.get("learner_level"),
                "preferred_explanation_style": dpm_context.get("preferred_explanation_style"),
                "is_weak_topic": dpm_context.get("is_weak_topic"),
                "retrieved_knowledge_chunks": len(skg_context.get("snippets", [])),
                "similar_successful_trace_count": len(similar_successful_traces),
                "failed_trace_count": len(failed_traces),
            },
        }

    def enrich_task(self, request: Any, analyzed_task: Dict[str, Any]) -> Dict[str, Any]:
        collected_context = self.collect(request, analyzed_task)

        enriched_task = dict(analyzed_task)
        enriched_task["learner_id"] = collected_context["learner_id"]
        enriched_task["student_question"] = collected_context["student_question"]
        enriched_task["query"] = collected_context["student_question"]
        enriched_task["topic"] = collected_context["topic"]
        enriched_task["task_type"] = collected_context["task_type"]
        enriched_task["current_step"] = collected_context["current_step"]
        enriched_task["collected_context"] = collected_context

        # Shortcuts for components that do not need the full nested object.
        enriched_task["learner_level"] = collected_context["summary"]["learner_level"]
        enriched_task["is_weak_topic"] = collected_context["summary"]["is_weak_topic"]
        enriched_task["knowledge_snippets"] = collected_context["skg"].get("snippets", [])
        enriched_task["grounding_text"] = collected_context["skg"].get("grounding_text", "")
        enriched_task["similar_successful_traces"] = collected_context["trace_memory"].get(
            "similar_successful_traces",
            [],
        )
        enriched_task["failed_traces"] = collected_context["trace_memory"].get(
            "failed_traces",
            [],
        )

        return enriched_task

    def _get_learner_id(self, request: Any) -> str:
        metadata = getattr(request, "metadata", {}) or {}
        context = getattr(request, "context", {}) or {}

        return (
            metadata.get("learner_id")
            or context.get("learner_id")
            or getattr(request, "learner_id", None)
            or "demo_user"
        )

    def _get_student_question(self, request: Any) -> str:
        return (
            getattr(request, "student_question", None)
            or getattr(request, "user_query", None)
            or getattr(request, "query", None)
            or ""
        )

    def _infer_topic(self, text: str) -> str:
        lower_text = text.lower()

        if "convolution" in lower_text or "kernel" in lower_text:
            return "convolution"

        if "overfitting" in lower_text:
            return "overfitting"

        if "gradient" in lower_text:
            return "gradient descent"

        return "general topic"
