"""Run one reproducible OpenTutorAI-Agentic tutoring demo and save evidence.

This is implementation evidence, not a formal evaluation script.

It records:
- user question
- learner id
- workflow status
- selected tools
- trace id/path
- output package
- SKG/RAG metadata
- TraceToolkit metadata
- DPM/context metadata
- Groq/provider availability
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.agentic.core.agentic_core_service import AgenticCoreService
from ai.agentic.memory.trace_store import TraceStore
from ai.llm.errors import LLMProviderUnavailableError


TRACE_DIR = Path("var/agentic_traces")
REPORT_DIR = Path("var/agentic_demo_runs")


def list_trace_files() -> set[str]:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return {str(path.resolve()) for path in TRACE_DIR.glob("*.json")}


def newest_new_trace(before: set[str]) -> Optional[Path]:
    after_paths = [path for path in TRACE_DIR.glob("*.json") if str(path.resolve()) not in before]

    if not after_paths:
        return None

    return sorted(after_paths, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def event_payload(trace: Dict[str, Any], event_type: str) -> Dict[str, Any]:
    events = trace.get("events", [])
    if not isinstance(events, list):
        return {}

    for event in reversed(events):
        if isinstance(event, dict) and event.get("event_type") == event_type:
            payload = event.get("payload", {})
            return payload if isinstance(payload, dict) else {}

    return {}


def find_first_key(obj: Any, key_name: str) -> Any:
    if isinstance(obj, dict):
        if key_name in obj:
            return obj[key_name]

        for value in obj.values():
            found = find_first_key(value, key_name)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key(item, key_name)
            if found is not None:
                return found

    return None


def extract_package(response: Optional[Dict[str, Any]], trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if response and isinstance(response.get("output_package"), dict):
        return response["output_package"]

    if trace and isinstance(trace.get("final_package"), dict):
        return trace["final_package"]

    package_event = event_payload(trace or {}, "output_package_built")
    return package_event if package_event else {}


def extract_tool_metadata(package: Dict[str, Any], tool_name: str) -> List[Dict[str, Any]]:
    metadata = package.get("metadata", {}) or {}
    tool_metadata = metadata.get("tool_metadata", {}) or {}

    value = tool_metadata.get(tool_name, [])
    return value if isinstance(value, list) else []


def compact_report(
    question: str,
    learner_id: str,
    response: Optional[Dict[str, Any]],
    trace: Optional[Dict[str, Any]],
    trace_summary: Optional[Dict[str, Any]],
    provider_status: str,
    provider_error: Optional[str],
) -> Dict[str, Any]:
    package = extract_package(response=response, trace=trace)

    rag_metadata = extract_tool_metadata(package, "RAGTool")
    trace_tool_metadata = extract_tool_metadata(package, "TraceSearchTool")
    visual_metadata = extract_tool_metadata(package, "VisualMatrixTool")
    matrix_metadata = extract_tool_metadata(package, "MatrixComputationTool")

    context_payload = event_payload(trace or {}, "task_context_analyzed")
    request_payload = event_payload(trace or {}, "tool_request_received")
    selection_payload = event_payload(trace or {}, "tools_selected")
    validation_payload = event_payload(trace or {}, "output_validated")

    package_metadata = package.get("metadata", {}) or {}
    validation_report = package.get("validation_report", {}) or {}


    selected_tools_from_event = []
    if isinstance(selection_payload, dict):
        selected_tools_from_event = (
            selection_payload.get("selected_tools")
            or selection_payload.get("tools")
            or []
        )
    elif isinstance(selection_payload, list):
        selected_tools_from_event = selection_payload

    effective_trace_path = (
        (response or {}).get("trace_path")
        or (trace_summary or {}).get("trace_path")
        or (trace or {}).get("trace_path")
    )

    recovery_metadata = extract_recovery_metadata_from_trace(effective_trace_path)

    summary = {
        "question": question,
        "learner_id": learner_id,
        "provider_status": provider_status,
        "workflow_status": (response or {}).get("status") or package.get("status"),
        "trace_id": (response or {}).get("trace_id") or (trace_summary or {}).get("trace_id") or package.get("trace_id"),
        "trace_path": effective_trace_path,
        "confidence": (response or {}).get("confidence") or validation_report.get("confidence_score"),
        "selected_tools": (trace_summary or {}).get("selected_tools") or selected_tools_from_event or package_metadata.get("tool_names", []) or (response or {}).get("selected_tools") or [],
        "recovery_used": recovery_metadata.get("recovery_used", False),
        "attempts": recovery_metadata.get("attempts", 1),
        "recovery_steps": recovery_metadata.get("recovery_steps", []),
        "content_format": package_metadata.get("content_format"),
        "rag_retrieval_mode": rag_metadata[0].get("retrieval_mode") if rag_metadata else None,
        "rag_grounding_source": rag_metadata[0].get("grounding_source") if rag_metadata else None,
        "rag_scores": rag_metadata[0].get("scores") if rag_metadata else None,
        "rag_matched_terms": rag_metadata[0].get("matched_terms") if rag_metadata else None,
        "trace_memory_similar_count": trace_tool_metadata[0].get("similar_successful_trace_count") if trace_tool_metadata else None,
        "trace_memory_failed_count": trace_tool_metadata[0].get("failed_trace_count") if trace_tool_metadata else None,
        "trace_memory_reference_trace": trace_tool_metadata[0].get("reference_trace_id") if trace_tool_metadata else None,
        "dpm_learner_level": find_first_key(context_payload, "learner_level"),
        "dpm_is_weak_topic": find_first_key(context_payload, "is_weak_topic"),
        "dpm_recent_trace_count": find_first_key(context_payload, "recent_trace_count"),
        "provider_error": provider_error,
    }

    return {
        "report_type": "implementation_demo_evidence",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "inputs": {
            "question": question,
            "learner_id": learner_id,
        },
        "response": response,
        "trace_summary": trace_summary,
        "trace_events": {
            "tool_request_received": request_payload,
            "task_context_analyzed": context_payload,
            "tools_selected": selection_payload,
            "output_validated": validation_payload,
        },
        "output_package": package,
        "tool_metadata": {
            "MatrixComputationTool": matrix_metadata,
            "RAGTool": rag_metadata,
            "TraceSearchTool": trace_tool_metadata,
            "VisualMatrixTool": visual_metadata,
        },
        "provider": {
            "status": provider_status,
            "error": provider_error,
        },
    }


def save_report(report: Dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    trace_id = report.get("summary", {}).get("trace_id") or "no_trace"
    path = REPORT_DIR / f"agentic_demo_{timestamp}_{trace_id}.json"

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return path



def extract_recovery_metadata_from_trace(trace_path):
    """Extract attempts and recovery usage directly from a saved trace file."""
    metadata = {
        "attempts": 1,
        "recovery_used": False,
        "recovery_steps": [],
    }

    if not trace_path:
        return metadata

    path = Path(trace_path)
    if not path.exists():
        return metadata

    try:
        trace_data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return metadata

    trace_text = json.dumps(trace_data, ensure_ascii=False).lower()

    if (
        "recover_failure" in trace_text
        or "failure_recovery_applied" in trace_text
        or "recovery_decision" in trace_text
        or "using_tools_required_by_failure_recovery" in trace_text
        or "controlled_failure_test" in trace_text
    ):
        metadata["recovery_used"] = True

    attempts = []

    def walk(obj):
        if isinstance(obj, dict):
            event_type = obj.get("event_type") or obj.get("node") or obj.get("event")
            payload = obj.get("payload", {})

            if event_type in {"failure_recovery_applied", "recover_failure"}:
                metadata["recovery_used"] = True
                metadata["recovery_steps"].append(obj)

            if isinstance(payload, dict):
                if payload.get("recovery_used") is True:
                    metadata["recovery_used"] = True
                    metadata["recovery_steps"].append(payload)

                if "previous_attempt" in payload and "next_attempt" in payload:
                    metadata["recovery_used"] = True
                    metadata["recovery_steps"].append(payload)

            for key, value in obj.items():
                if key in {"attempt", "attempt_number", "current_attempt", "next_attempt"} and isinstance(value, int):
                    attempts.append(value)
                walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(trace_data)

    if attempts:
        metadata["attempts"] = max(attempts)
    elif metadata["recovery_used"]:
        metadata["attempts"] = 2

    return metadata

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="Explain convolution with a simple example.",
        help="Tutoring question to run.",
    )
    parser.add_argument(
        "--learner-id",
        default="demo_user",
        help="Learner id used for DPM personalization.",
    )
    parser.add_argument(
        "--force-recovery-test",
        action="store_true",
        help="Force a weak first attempt to prove FailureRecovery.",
    )
    args = parser.parse_args()

    before_traces = list_trace_files()

    response = None
    provider_status = "available"
    provider_error = None

    service = AgenticCoreService()

    try:
        response = service.handle_request(
            {
                "request_type": "personalized_tutoring",
                "query": args.question,
                "learner_id": args.learner_id,
                "metadata": {
                    "source": "run_agentic_tutoring_demo",
                    "force_recovery_test": args.force_recovery_test,
                },
            }
        )
    except LLMProviderUnavailableError as exc:
        provider_status = "unavailable"
        provider_error = str(exc)
    except Exception as exc:
        provider_status = "error"
        provider_error = f"{type(exc).__name__}: {exc}"

    trace_store = TraceStore()

    trace_path = None
    if response and response.get("trace_path"):
        trace_path = Path(response["trace_path"])
    else:
        trace_path = newest_new_trace(before_traces)

    trace = None
    trace_summary = None

    if trace_path:
        trace = trace_store.load_trace(str(trace_path))
        if trace:
            trace_summary = trace_store.summarize_trace(trace)

    report = compact_report(
        question=args.question,
        learner_id=args.learner_id,
        response=response,
        trace=trace,
        trace_summary=trace_summary,
        provider_status=provider_status,
        provider_error=provider_error,
    )

    report_path = save_report(report)

    print("\n" + "=" * 80)
    print("AGENTIC DEMO EVIDENCE SAVED")
    print("=" * 80)
    print(f"Report path: {report_path}")
    print(f"Provider status: {provider_status}")
    print(f"Workflow status: {report['summary'].get('workflow_status')}")
    print(f"Trace ID: {report['summary'].get('trace_id')}")
    print(f"Trace path: {report['summary'].get('trace_path')}")
    print(f"Selected tools: {report['summary'].get('selected_tools')}")
    print(f"Confidence: {report['summary'].get('confidence')}")
    print(f"RAG mode: {report['summary'].get('rag_retrieval_mode')}")
    print(f"Similar traces: {report['summary'].get('trace_memory_similar_count')}")
    print(f"Failed traces: {report['summary'].get('trace_memory_failed_count')}")
    print(f"DPM learner level: {report['summary'].get('dpm_learner_level')}")
    print(f"DPM weak topic: {report['summary'].get('dpm_is_weak_topic')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
