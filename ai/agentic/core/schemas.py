from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


@dataclass
class ToolRequest:
    request_id: str
    workflow_source: str
    task_type: str
    user_query: str
    current_step: str
    expected_output: str
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolPlan:
    plan_id: str
    selected_tools: List[str]
    steps: List[Dict[str, Any]]


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    evidence: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ValidationReport:
    status: str
    confidence_score: float
    failure_reason: Optional[str] = None
    recommended_action: Optional[str] = None


@dataclass
class OutputPackage:
    package_id: str
    status: str
    content: str
    evidence: List[str]
    references: List[str]
    validation_report: ValidationReport
    trace_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
