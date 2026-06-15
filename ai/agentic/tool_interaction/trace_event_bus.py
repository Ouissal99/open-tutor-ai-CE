import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4


class TraceEventBus:
    """
    Tool Trace Collector & Logger.
    It records the full tool interaction process and saves it as JSON.
    """

    def __init__(self, trace_dir: str = "var/agentic_traces"):
        self.trace_id = f"TT-{uuid4().hex[:8]}"
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, list):
            return [self._serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        return value

    def emit(self, event_type: str, payload: Any) -> None:
        self.events.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "payload": self._serialize(payload)
        })

    def save(self, final_package: Any) -> str:
        path = self.trace_dir / f"{self.trace_id}.json"
        trace = {
            "trace_id": self.trace_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "events": self.events,
            "final_package": self._serialize(final_package)
        }
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)
