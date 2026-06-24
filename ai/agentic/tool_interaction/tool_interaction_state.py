"""State schema for the LangGraph-based Tool Interaction Manager."""

from typing import Any, Dict, List, Optional, TypedDict


class ToolInteractionState(TypedDict, total=False):
    """Shared state passed between ToolInteractionGraph nodes."""

    request: Any

    trace_bus: Any
    trace_id: str
    trace_path: str

    analyzed_task: Dict[str, Any]
    collected_context: Dict[str, Any]
    context_summary: Dict[str, Any]

    attempt: int
    max_attempts: int

    selected_tools: List[str]
    selection_metadata: Dict[str, Any]
    plan: Any
    results: List[Any]
    last_results: List[Any]

    report: Any
    last_report: Any

    recovery_decision: Dict[str, Any]
    recovery_steps: List[Dict[str, Any]]
    recovery_used: bool

    package: Any
    final_status: str
    route_decision: str

    node_history: List[str]
    trace_events: List[Dict[str, Any]]
