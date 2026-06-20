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
    - extracting basic metadata from traces

    In this prototype, traces are stored as JSON files in:
    var/agentic_traces/
    """

    def __init__(self, base_dir: str = "var/agentic_traces"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_trace_files(self) -> List[Path]:
        """
        Return all JSON trace files, newest first.
        """
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
        """
        Load and return all traces.
        """
        traces = []

        for file_path in self.list_trace_files():
            trace = self.load_trace(str(file_path))
            if trace:
                traces.append(trace)

        return traces

    def get_latest_trace(self) -> Optional[Dict[str, Any]]:
        """
        Return the newest trace.
        """
        files = self.list_trace_files()
        if not files:
            return None

        return self.load_trace(str(files[0]))

    def save_trace(self, trace_id: str, trace_data: Dict[str, Any]) -> str:
        """
        Save a trace dictionary manually.
        This is useful later for tests or custom trace logging.
        """
        path = self.base_dir / f"{trace_id}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)

        return str(path)

    def extract_trace_id(self, trace: Dict[str, Any], path: Optional[Path] = None) -> str:
        """
        Extract trace ID from common locations.
        """
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
        found = self._find_first_value(trace, "task_type")
        return found or "unknown_task"

    def extract_status(self, trace: Dict[str, Any]) -> str:
        """
        Extract a general status from the trace.
        """
        validation_status = self._find_first_value(trace, "status")

        if validation_status:
            return validation_status

        return "unknown"

    def extract_selected_tools(self, trace: Dict[str, Any]) -> List[str]:
        """
        Extract tool names from different possible trace structures.
        """
        tools = set()

        def walk(obj: Any):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in {"selected_tools", "tools", "tool_names"} and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                tools.add(item)

                    if key in {"tool_name", "selected_tool"} and isinstance(value, str):
                        tools.add(value)

                    walk(value)

            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(trace)
        return sorted(tools)

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

    def summarize_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce compact metadata for display, search, and evaluation.
        """
        trace_id = self.extract_trace_id(trace)
        task_type = self.extract_task_type(trace)
        status = self.extract_status(trace)
        selected_tools = self.extract_selected_tools(trace)
        attempts = self.extract_attempts(trace)

        return {
            "trace_id": trace_id,
            "task_type": task_type,
            "status": status,
            "selected_tools": selected_tools,
            "attempts": attempts,
            "trace_path": trace.get("_trace_path"),
        }

    def _find_first_value(self, obj: Any, target_key: str) -> Optional[Any]:
        """
        Recursively find the first value for a key in nested dict/list data.
        """
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
