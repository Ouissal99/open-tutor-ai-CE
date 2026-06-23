import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TraceStore:
    """
    File-based trace memory store.

    This component is responsible for:
    - reading saved tool interaction traces
    - listing available traces
    - loading one trace by ID or path
    - extracting useful metadata from traces

    In this prototype, traces are stored as JSON files in:
    var/agentic_traces/
    """

    TOOL_NAME_ALIASES = {
        "mock_rag_tool": "RAGTool",
        "mock_visualizer_tool": "VisualMatrixTool",
        "mock_validator_support": "TraceSearchTool",
        "mock_matrix_tool": "MatrixComputationTool",
        "mock_calculator_tool": "CalculatorTool",
        "mock_trace_tool": "TraceSearchTool",
    }

    def __init__(self, base_dir: str = "var/agentic_traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_trace_files(self) -> List[Path]:
        """Return all JSON trace files, newest first."""
        files = list(self.base_dir.glob("*.json"))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    def load_trace(self, trace_id_or_path: str) -> Optional[Dict[str, Any]]:
        """
        Load one trace using either:
        - trace ID: TT-xxxx
        - path: var/agentic_traces/TT-xxxx.json
        """
        path = Path(trace_id_or_path)

        if not path.exists():
            if not trace_id_or_path.endswith(".json"):
                path = self.base_dir / f"{trace_id_or_path}.json"
            else:
                path = self.base_dir / trace_id_or_path

        if not path.exists():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception:
            return None

        trace["_trace_path"] = str(path)
        trace["_trace_id"] = self.extract_trace_id(trace, path)
        return trace

    def list_traces(self) -> List[Dict[str, Any]]:
        """Load and return all traces."""
        traces = []

        for file_path in self.list_trace_files():
            trace = self.load_trace(str(file_path))
            if trace:
                traces.append(trace)

        return traces

    def get_latest_trace(self) -> Optional[Dict[str, Any]]:
        """Return the newest trace."""
        files = self.list_trace_files()
        if not files:
            return None

        return self.load_trace(str(files[0]))

    def save_trace(self, trace_id: str, trace_data: Dict[str, Any]) -> str:
        """Save a trace dictionary manually."""
        path = self.base_dir / f"{trace_id}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        return str(path)

    def summarize_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Produce compact metadata for display, search, and reuse."""
        trace_id = self.extract_trace_id(trace)
        task_type = self.extract_task_type(trace)
        status = self.extract_status(trace)
        selected_tools = self.extract_selected_tools(trace)
        attempts = self.extract_attempts(trace)

        return {
            "trace_id": trace_id,
            "trace_path": trace.get("_trace_path"),
            "created_at": trace.get("created_at"),
            "task_type": task_type,
            "status": status,
            "student_question": self.extract_student_question(trace),
            "step_goal": self.extract_step_goal(trace),
            "selected_tools": selected_tools,
            "attempts": attempts,
            "confidence_score": self.extract_confidence_score(trace),
            "failure_reason": self.extract_failure_reason(trace),
            "recovery_used": self.extract_recovery_used(trace),
            "content_preview": self.extract_content_preview(trace),
        }

    def extract_trace_id(self, trace: Dict[str, Any], path: Optional[Path] = None) -> str:
        if "trace_id" in trace:
            return trace["trace_id"]

        if "_trace_id" in trace:
            return trace["_trace_id"]

        found = self._find_first_value(trace, "trace_id")
        if found:
            return found

        if path:
            return path.stem

        return "unknown_trace"

    def extract_task_type(self, trace: Dict[str, Any]) -> str:
        request_payload = self._event_payload(trace, "tool_request_received")
        if request_payload.get("task_type"):
            return request_payload["task_type"]

        found = self._find_first_value(trace, "task_type")
        return found or "unknown_task"

    def extract_student_question(self, trace: Dict[str, Any]) -> str:
        request_payload = self._event_payload(trace, "tool_request_received")
        if request_payload.get("student_question"):
            return request_payload["student_question"]

        found = self._find_first_value(trace, "student_question")
        return found or ""

    def extract_step_goal(self, trace: Dict[str, Any]) -> str:
        request_payload = self._event_payload(trace, "tool_request_received")
        if request_payload.get("step_goal"):
            return request_payload["step_goal"]

        found = self._find_first_value(trace, "step_goal")
        return found or ""

    def extract_status(self, trace: Dict[str, Any]) -> str:
        final_package = trace.get("final_package")
        if isinstance(final_package, dict) and final_package.get("status"):
            return str(final_package["status"])

        package_payload = self._event_payload(trace, "output_package_built")
        if package_payload.get("status"):
            return str(package_payload["status"])

        validation_payload = self._event_payload(trace, "output_validated")
        if validation_payload.get("status"):
            return str(validation_payload["status"])

        found = self._find_first_value(trace, "status")
        return str(found) if found is not None else "unknown"

    def extract_confidence_score(self, trace: Dict[str, Any]) -> Optional[float]:
        final_package = trace.get("final_package")
        if isinstance(final_package, dict):
            report = final_package.get("validation_report", {}) or {}
            if report.get("confidence_score") is not None:
                return report.get("confidence_score")

        package_payload = self._event_payload(trace, "output_package_built")
        report = package_payload.get("validation_report", {}) or {}
        if report.get("confidence_score") is not None:
            return report.get("confidence_score")

        found = self._find_first_value(trace, "confidence_score")
        return found if isinstance(found, (int, float)) else None

    def extract_failure_reason(self, trace: Dict[str, Any]) -> Optional[str]:
        final_package = trace.get("final_package")
        if isinstance(final_package, dict):
            report = final_package.get("validation_report", {}) or {}
            if report.get("failure_reason"):
                return str(report["failure_reason"])

        package_payload = self._event_payload(trace, "output_package_built")
        report = package_payload.get("validation_report", {}) or {}
        if report.get("failure_reason"):
            return str(report["failure_reason"])

        found = self._find_first_value(trace, "failure_reason")
        return str(found) if found else None

    def extract_recovery_used(self, trace: Dict[str, Any]) -> bool:
        found = self._find_first_value(trace, "recovery_used")
        return bool(found) if found is not None else False

    def extract_selected_tools(self, trace: Dict[str, Any]) -> List[str]:
        """
        Extract tool names from trace structures and normalize old prototype names.
        """
        tools = []

        def add_tool(tool_name: str):
            mapped = self.TOOL_NAME_ALIASES.get(tool_name, tool_name)
            if mapped and mapped not in tools:
                tools.append(mapped)

        tools_selected_payload = self._event_payload(trace, "tools_selected")
        selected = tools_selected_payload.get("selected_tools")
        if isinstance(selected, list):
            for item in selected:
                if isinstance(item, str):
                    add_tool(item)

        package_payload = self._event_payload(trace, "output_package_built")
        metadata = package_payload.get("metadata", {}) or {}
        tool_names = metadata.get("tool_names")
        if isinstance(tool_names, list):
            for item in tool_names:
                if isinstance(item, str):
                    add_tool(item)

        final_package = trace.get("final_package")
        if isinstance(final_package, dict):
            metadata = final_package.get("metadata", {}) or {}
            tool_names = metadata.get("tool_names")
            if isinstance(tool_names, list):
                for item in tool_names:
                    if isinstance(item, str):
                        add_tool(item)

        if tools:
            return tools

        def walk(obj: Any):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {"selected_tools", "tools", "tool_names"} and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                add_tool(item)

                    if key in {"tool_name", "selected_tool"} and isinstance(value, str):
                        add_tool(value)

                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(trace)
        return tools

    def extract_attempts(self, trace: Dict[str, Any]) -> int:
        attempts = []

        def walk(obj: Any):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {"attempt", "attempt_number", "attempts"} and isinstance(value, int):
                        attempts.append(value)
                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(trace)

        if not attempts:
            return 1

        return max(attempts)

    def extract_content_preview(self, trace: Dict[str, Any], max_chars: int = 500) -> str:
        final_package = trace.get("final_package")
        if isinstance(final_package, dict) and final_package.get("content"):
            return str(final_package["content"])[:max_chars]

        package_payload = self._event_payload(trace, "output_package_built")
        if package_payload.get("content"):
            return str(package_payload["content"])[:max_chars]

        return ""

    def _event_payload(self, trace: Dict[str, Any], event_type: str) -> Dict[str, Any]:
        events = trace.get("events", [])
        if not isinstance(events, list):
            return {}

        for event in reversed(events):
            if not isinstance(event, dict):
                continue

            if event.get("event_type") == event_type:
                payload = event.get("payload", {})
                return payload if isinstance(payload, dict) else {}

        return {}

    def _find_first_value(self, obj: Any, target_key: str) -> Optional[Any]:
        """Recursively find the first value for a key in nested dict/list data."""
        if isinstance(obj, dict):
            if target_key in obj:
                return obj[target_key]

            for value in obj.values():
                found = self._find_first_value(value, target_key)
                if found is not None:
                    return found

        elif isinstance(obj, list):
            for item in obj:
                found = self._find_first_value(item, target_key)
                if found is not None:
                    return found

        return None
