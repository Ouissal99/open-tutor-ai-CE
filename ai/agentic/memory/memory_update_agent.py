"""MemoryUpdateAgent for Dynamic Personal Memory.

This agent updates the learner-specific DPM after a validated tutoring interaction.

It is intentionally file-based and resource-safe:
- no vector DB
- no heavy model
- no extra install

It updates:
- L1: interaction trace summaries
- L2: tutoring observations
- L3: recent learner memory and profile metadata
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.agentic.memory.dynamic_personal_memory import DynamicPersonalMemory


class MemoryUpdateAgent:
    """Updates DPM after a validated personalized tutoring interaction."""

    def __init__(
        self,
        dpm: Optional[DynamicPersonalMemory] = None,
        base_dir: str = "var/agentic_memory/dpm",
    ):
        self.dpm = dpm or DynamicPersonalMemory(base_dir=base_dir)
        self.base_dir = Path(base_dir)

    def update_after_interaction(
        self,
        learner_id: str,
        student_question: str,
        investigation_result: Dict[str, Any],
        request: Any,
        package: Any,
        trace_path: Any = None,
    ) -> Dict[str, Any]:
        """Update L1, L2, and L3 memory after one tool-supported interaction."""
        package_dict = package.to_dict() if hasattr(package, "to_dict") else {}
        validation_report = package_dict.get("validation_report", {}) or {}
        package_metadata = package_dict.get("metadata", {}) or {}

        topic = investigation_result.get("topic") or package_metadata.get("topic") or "unknown_topic"
        task_type = getattr(request, "task_type", None) or investigation_result.get("task_type")
        status = package_dict.get("status")
        confidence_score = validation_report.get("confidence_score")
        failure_reason = validation_report.get("failure_reason")
        selected_tools = package_metadata.get("tool_names", [])

        timestamp = datetime.now(timezone.utc).isoformat()

        memory_update_summary = self._build_memory_update_summary(
            learner_id=learner_id,
            student_question=student_question,
            topic=topic,
            task_type=task_type,
            status=status,
            confidence_score=confidence_score,
            failure_reason=failure_reason,
            selected_tools=selected_tools,
            investigation_result=investigation_result,
            package_metadata=package_metadata,
            timestamp=timestamp,
        )

        self._update_l1(
            learner_id=learner_id,
            student_question=student_question,
            investigation_result=investigation_result,
            request=request,
            package_dict=package_dict,
            trace_path=trace_path,
            memory_update_summary=memory_update_summary,
        )

        self._update_l2_tutoring_notes(
            learner_id=learner_id,
            memory_update_summary=memory_update_summary,
        )

        self._update_l3_recent_memory(
            learner_id=learner_id,
            memory_update_summary=memory_update_summary,
        )

        self._update_l3_profile_metadata(
            learner_id=learner_id,
            memory_update_summary=memory_update_summary,
        )

        return memory_update_summary

    def _build_memory_update_summary(
        self,
        learner_id: str,
        student_question: str,
        topic: str,
        task_type: str,
        status: str,
        confidence_score: Any,
        failure_reason: Any,
        selected_tools: List[str],
        investigation_result: Dict[str, Any],
        package_metadata: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        diagnosed_gap = self._diagnose_gap(topic=topic, investigation_result=investigation_result)
        recommended_strategy = self._recommend_strategy(topic=topic, selected_tools=selected_tools)
        new_weak_topic = self._infer_new_weak_topic(topic=topic, diagnosed_gap=diagnosed_gap)

        return {
            "memory_update_type": "post_tool_interaction_update",
            "memory_update_agent": "MemoryUpdateAgent",
            "updated_at": timestamp,
            "learner_id": learner_id,
            "student_question": student_question,
            "topic": topic,
            "task_type": task_type,
            "status": status,
            "confidence_score": confidence_score,
            "failure_reason": failure_reason,
            "selected_tools": selected_tools,
            "diagnosed_gap": diagnosed_gap,
            "recommended_future_strategy": recommended_strategy,
            "new_weak_topic": new_weak_topic,
            "content_format": package_metadata.get("content_format"),
            "retrieval_mode": self._extract_rag_retrieval_mode(package_metadata),
            "trace_memory_used": self._extract_trace_memory_used(package_metadata),
            "profile_update_policy": "conservative_metadata_update",
        }

    def _update_l1(
        self,
        learner_id: str,
        student_question: str,
        investigation_result: Dict[str, Any],
        request: Any,
        package_dict: Dict[str, Any],
        trace_path: Any,
        memory_update_summary: Dict[str, Any],
    ) -> None:
        validation_report = package_dict.get("validation_report", {}) or {}
        package_metadata = package_dict.get("metadata", {}) or {}

        trace_summary = {
            "trace_type": "tool_interaction_summary",
            "stored_from": "MemoryUpdateAgent",
            "timestamp": memory_update_summary["updated_at"],
            "learner_id": learner_id,
            "student_question": student_question,
            "topic": memory_update_summary.get("topic"),
            "task_type": getattr(request, "task_type", None),
            "step_goal": getattr(request, "step_goal", None),
            "expected_output": getattr(request, "expected_output", None),
            "request_id": getattr(request, "request_id", None),
            "trace_id": package_dict.get("trace_id"),
            "trace_path": str(trace_path) if trace_path else None,
            "status": package_dict.get("status"),
            "confidence_score": validation_report.get("confidence_score"),
            "failure_reason": validation_report.get("failure_reason"),
            "selected_tools": package_metadata.get("tool_names", []),
            "content_format": package_metadata.get("content_format"),
            "memory_update_summary": memory_update_summary,
        }

        self.dpm.append_l1_trace_summary(
            learner_id=learner_id,
            trace_summary=trace_summary,
        )

    def _update_l2_tutoring_notes(
        self,
        learner_id: str,
        memory_update_summary: Dict[str, Any],
    ) -> None:
        path = self.base_dir / learner_id / "L2" / "tutoring.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = path.read_text(encoding="utf-8") if path.exists() else "# Tutoring Memory\n"

        entry = f"""

## MemoryUpdateAgent observation — {memory_update_summary.get("updated_at")}

- Topic: {memory_update_summary.get("topic")}
- Task type: {memory_update_summary.get("task_type")}
- Status: {memory_update_summary.get("status")}
- Confidence: {memory_update_summary.get("confidence_score")}
- Diagnosed gap: {memory_update_summary.get("diagnosed_gap")}
- Recommended future strategy: {memory_update_summary.get("recommended_future_strategy")}
- Selected tools: {", ".join(memory_update_summary.get("selected_tools", []))}
"""

        path.write_text(existing.rstrip() + entry + "\n", encoding="utf-8")

    def _update_l3_recent_memory(
        self,
        learner_id: str,
        memory_update_summary: Dict[str, Any],
    ) -> None:
        path = self.base_dir / learner_id / "L3" / "recent.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        text = f"""# Recent Memory

Last updated: {memory_update_summary.get("updated_at")}

The learner recently worked on: {memory_update_summary.get("topic")}.
Observed learning gap: {memory_update_summary.get("diagnosed_gap")}.
Recommended next tutoring strategy: {memory_update_summary.get("recommended_future_strategy")}.
Last task type: {memory_update_summary.get("task_type")}.
Last selected tools: {", ".join(memory_update_summary.get("selected_tools", []))}.
"""

        path.write_text(text, encoding="utf-8")

    def _update_l3_profile_metadata(
        self,
        learner_id: str,
        memory_update_summary: Dict[str, Any],
    ) -> None:
        path = self.base_dir / learner_id / "L3" / "profile.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                profile = {}
        else:
            profile = {}

        profile.setdefault("learner_id", learner_id)
        profile.setdefault("level", "unknown")
        profile.setdefault("known_topics", [])
        profile.setdefault("weak_topics", [])
        profile.setdefault("learning_goals", [])

        topic = memory_update_summary.get("topic")
        new_weak_topic = memory_update_summary.get("new_weak_topic")

        if topic:
            recent_topics = profile.get("recent_topics", [])
            if topic not in recent_topics:
                recent_topics.append(topic)
            profile["recent_topics"] = recent_topics[-10:]

        if new_weak_topic:
            weak_topics = profile.get("weak_topics", [])
            if new_weak_topic not in weak_topics:
                weak_topics.append(new_weak_topic)
            profile["weak_topics"] = weak_topics

        profile["last_memory_update"] = {
            "updated_at": memory_update_summary.get("updated_at"),
            "topic": topic,
            "diagnosed_gap": memory_update_summary.get("diagnosed_gap"),
            "recommended_future_strategy": memory_update_summary.get("recommended_future_strategy"),
            "confidence_score": memory_update_summary.get("confidence_score"),
            "status": memory_update_summary.get("status"),
        }

        path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _diagnose_gap(
        self,
        topic: str,
        investigation_result: Dict[str, Any],
    ) -> str:
        topic_lower = str(topic or "").lower()
        raw_plan = investigation_result.get("initial_tutoring_plan", [])

        plan_text = json.dumps(raw_plan, ensure_ascii=False).lower()

        if "kernel" in plan_text or "kernel" in topic_lower:
            return "kernel movement"

        if "feature map" in plan_text:
            return "feature map interpretation"

        if "matrix" in plan_text or topic_lower == "convolution":
            return "matrix-to-output transformation"

        return f"needs more support with {topic}"

    def _recommend_strategy(
        self,
        topic: str,
        selected_tools: List[str],
    ) -> str:
        if "VisualMatrixTool" in selected_tools or str(topic).lower() == "convolution":
            return "use small matrix examples with step-by-step visual explanation"

        if "CodeSandboxTool" in selected_tools:
            return "use executable code examples with short explanations"

        if "CalculatorTool" in selected_tools:
            return "show verified calculations step by step"

        return "use simple step-by-step explanation grounded in course evidence"

    def _infer_new_weak_topic(
        self,
        topic: str,
        diagnosed_gap: str,
    ) -> Optional[str]:
        if str(topic).lower() == "convolution" and diagnosed_gap:
            return diagnosed_gap

        return None

    def _extract_rag_retrieval_mode(self, package_metadata: Dict[str, Any]) -> Optional[str]:
        tool_metadata = package_metadata.get("tool_metadata", {}) or {}
        rag_entries = tool_metadata.get("RAGTool", [])

        if rag_entries and isinstance(rag_entries[0], dict):
            return rag_entries[0].get("retrieval_mode")

        return None

    def _extract_trace_memory_used(self, package_metadata: Dict[str, Any]) -> Dict[str, Any]:
        tool_metadata = package_metadata.get("tool_metadata", {}) or {}
        trace_entries = tool_metadata.get("TraceSearchTool", [])

        if trace_entries and isinstance(trace_entries[0], dict):
            return {
                "similar_successful_trace_count": trace_entries[0].get("similar_successful_trace_count"),
                "failed_trace_count": trace_entries[0].get("failed_trace_count"),
                "reference_trace_id": trace_entries[0].get("reference_trace_id"),
            }

        return {}
