import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TraceEventBus:
    """
    Tool Trace Collector & Logger.

    Phase 13D:
    Saves both raw events and a full lifecycle trace schema for thesis/evaluation use.
    """

    def __init__(self, trace_dir: str = "var/agentic_traces"):
        self.trace_id = f"TT-{uuid4().hex[:8]}"
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def _serialize(self, value: Any) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return self._serialize(value.to_dict())

        if is_dataclass(value):
            return self._serialize(asdict(value))

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, tuple):
            return [self._serialize(v) for v in value]

        if isinstance(value, list):
            return [self._serialize(v) for v in value]

        if isinstance(value, dict):
            return {str(k): self._serialize(v) for k, v in value.items()}

        return value

    def emit(self, event_type: str, payload: Any) -> None:
        self.events.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event_type": event_type,
                "payload": self._serialize(payload),
            }
        )

    def save(self, final_package: Any) -> str:
        path = self.trace_dir / f"{self.trace_id}.json"
        final_package_dict = self._serialize(final_package)

        trace = self._build_full_lifecycle_trace(final_package_dict)
        trace["trace_path"] = str(path)

        path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return str(path)

    def _build_full_lifecycle_trace(self, final_package: Dict[str, Any]) -> Dict[str, Any]:
        request = self._first_payload("tool_request_received") or {}
        analyzed_task = self._last_payload("task_analyzed") or {}
        context_task = self._last_payload("task_context_analyzed") or {}

        collected_context = (
            context_task.get("collected_context")
            or analyzed_task.get("collected_context")
            or {}
        )

        context_summary = (
            collected_context.get("summary")
            or context_task.get("summary")
            or {}
        )

        dpm_snapshot = collected_context.get("dpm", {})
        skg_context = collected_context.get("skg", {})
        trace_memory = collected_context.get("trace_memory", {})

        investigation_result = (
            request.get("metadata", {}).get("investigation_result")
            or analyzed_task.get("request_metadata", {}).get("investigation_result")
            or {}
        )

        tool_plans = self._payloads("tool_plan_created")
        tool_results = self._payloads("tools_executed")
        validation_reports = self._payloads("output_validated")

        recovery_decisions = self._payloads("recovery_decision")
        failure_recovery_steps = self._dedupe_jsonable(
            self._payloads("failure_recovery_applied")
        )

        selection_history = self._extract_selection_history(tool_plans)
        final_selection = selection_history[-1] if selection_history else {}

        skg_retrieved_chunks = (
            skg_context.get("retrieved_chunks")
            or skg_context.get("chunks")
            or self._find_first_key(tool_results, "retrieved_chunks")
            or []
        )

        if not skg_retrieved_chunks:
            skg_retrieved_chunks = self._extract_rag_chunks_from_tool_results(tool_results)

        similar_successful_traces = (
            trace_memory.get("similar_successful_traces")
            or self._find_first_key(tool_results, "similar_successful_traces")
            or []
        )

        failed_traces = (
            trace_memory.get("failed_traces")
            or self._find_first_key(tool_results, "failed_traces")
            or []
        )

        attempts = self._extract_attempt_count(
            validation_reports=validation_reports,
            failure_recovery_steps=failure_recovery_steps,
        )

        recovery_used = bool(failure_recovery_steps)

        llm_calls_metadata = {
            "investigation_agent": {
                "model": investigation_result.get("llm_model"),
                "raw_response_available": bool(investigation_result.get("raw_response")),
            },
            "final_answer_generator": {
                "status": "not_generated_yet",
                "component": "TutoringAnswerWriter",
            },
        }

        return {
            "trace_schema_version": "phase_13d_full_lifecycle_v1",
            "trace_id": self.trace_id,
            "created_at": datetime.utcnow().isoformat() + "Z",

            "request_id": request.get("request_id"),
            "learner_id": request.get("metadata", {}).get("learner_id"),
            "student_question": request.get("student_question"),
            "task_type": request.get("task_type"),
            "workflow_source": request.get("workflow_source"),
            "current_step": request.get("step_goal"),
            "expected_output": request.get("expected_output"),

            "investigation_result": investigation_result,
            "analyzed_task": analyzed_task,

            "collected_context_summary": context_summary,
            "dpm_profile_snapshot": dpm_snapshot,
            "skg_context": skg_context,
            "skg_retrieved_chunks": skg_retrieved_chunks,
            "similar_successful_traces": similar_successful_traces,
            "failed_traces": failed_traces,

            "tool_selection": final_selection,
            "selection_history": selection_history,
            "selection_strategy": final_selection.get("selection_strategy"),
            "reference_trace_id": final_selection.get("reference_trace_id"),

            "tool_plan": tool_plans[-1] if tool_plans else {},
            "tool_plan_history": tool_plans,
            "tool_results": tool_results[-1] if tool_results else [],
            "tool_results_history": tool_results,

            "validation_report": validation_reports[-1] if validation_reports else {},
            "validation_reports": validation_reports,

            "recovery_used": recovery_used,
            "attempts": attempts,
            "recovery_decisions": recovery_decisions,
            "failure_recovery_steps": failure_recovery_steps,

            "output_package": final_package,
            "final_package": final_package,

            "final_answer_status": "not_generated_yet",
            "final_answer": None,
            "provider_failure": None,
            "llm_calls_metadata": llm_calls_metadata,

            "memory_update_summary": None,
            "memory_update_status": "not_updated_yet",

            "events": self.events,
        }

    def _payloads(self, event_type: str) -> List[Any]:
        return [
            event.get("payload")
            for event in self.events
            if isinstance(event, dict) and event.get("event_type") == event_type
        ]

    def _first_payload(self, event_type: str) -> Optional[Any]:
        payloads = self._payloads(event_type)
        return payloads[0] if payloads else None

    def _last_payload(self, event_type: str) -> Optional[Any]:
        payloads = self._payloads(event_type)
        return payloads[-1] if payloads else None

    def _extract_selection_history(self, tool_plans: List[Any]) -> List[Dict[str, Any]]:
        history = []

        for plan in tool_plans:
            if not isinstance(plan, dict):
                continue

            metadata = plan.get("metadata", {}) or {}
            selection = metadata.get("tool_selection", {}) or {}

            if selection:
                history.append(selection)

        return history

    def _extract_attempt_count(
        self,
        validation_reports: List[Any],
        failure_recovery_steps: List[Any],
    ) -> int:
        attempts = []

        for event in self.events:
            payload = event.get("payload") if isinstance(event, dict) else None
            self._collect_attempts(payload, attempts)

        for report in validation_reports:
            self._collect_attempts(report, attempts)

        for step in failure_recovery_steps:
            self._collect_attempts(step, attempts)

        if attempts:
            return max(attempts)

        if failure_recovery_steps:
            return 2

        return 1

    def _collect_attempts(self, obj: Any, attempts: List[int]) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"attempt", "attempt_number", "current_attempt", "next_attempt"} and isinstance(value, int):
                    attempts.append(value)
                else:
                    self._collect_attempts(value, attempts)

        elif isinstance(obj, list):
            for item in obj:
                self._collect_attempts(item, attempts)

    def _find_first_key(self, obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]

            for value in obj.values():
                found = self._find_first_key(value, key)
                if found is not None:
                    return found

        elif isinstance(obj, list):
            for item in obj:
                found = self._find_first_key(item, key)
                if found is not None:
                    return found

        return None

    def _extract_rag_chunks_from_tool_results(self, tool_results_history: List[Any]) -> List[Dict[str, Any]]:
        chunks = []

        for result_group in tool_results_history:
            if not isinstance(result_group, list):
                continue

            for result in result_group:
                if not isinstance(result, dict):
                    continue

                if result.get("tool_name") != "RAGTool":
                    continue

                metadata = result.get("metadata", {}) or {}
                evidence = result.get("evidence", []) or []
                references = result.get("references", []) or []

                chunks.append(
                    {
                        "tool_name": "RAGTool",
                        "retrieval_mode": metadata.get("retrieval_mode"),
                        "scores": metadata.get("scores"),
                        "chunk_ids": metadata.get("chunk_ids"),
                        "matched_terms": metadata.get("matched_terms"),
                        "score_breakdown": metadata.get("score_breakdown"),
                        "evidence": evidence,
                        "references": references,
                    }
                )

        return chunks

    def _dedupe_jsonable(self, items: List[Any]) -> List[Any]:
        deduped = []
        seen = set()

        for item in items:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)

            if key in seen:
                continue

            seen.add(key)
            deduped.append(item)

        return deduped

    @staticmethod
    def update_trace_file(trace_path: Any, updates: Dict[str, Any]) -> None:
        if not trace_path:
            return

        path = Path(trace_path)

        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        data.update(updates)
        data["updated_at"] = datetime.utcnow().isoformat() + "Z"

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
