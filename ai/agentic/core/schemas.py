from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid


def new_id(prefix: str) -> str:
    """
    Generate a short readable ID for requests, packages, and traces.
    Example: REQ-a1b2c3d4
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------
# Agentic Core request/response schemas
# ---------------------------------------------------------------------

@dataclass
class AgenticRequest:
    """
    Standard input request for the agentic core.
    Used by terminal runner, future API, future UI, and evaluation.
    """
    request_type: str
    query: str
    learner_id: str = "demo_user"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticResponse:
    """
    Standard response returned by the agentic core.
    """
    answer: str
    status: str
    confidence: Optional[float] = None
    trace_id: Optional[str] = None
    trace_path: Optional[str] = None
    attempts: int = 1
    selected_tools: List[str] = field(default_factory=list)
    recovery_used: bool = False
    output_package: Dict[str, Any] = field(default_factory=dict)
    scratchpad: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status,
            "confidence": self.confidence,
            "trace_id": self.trace_id,
            "trace_path": self.trace_path,
            "attempts": self.attempts,
            "selected_tools": self.selected_tools,
            "recovery_used": self.recovery_used,
            "output_package": self.output_package,
            "scratchpad": self.scratchpad,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------
# Tool interaction schemas used by the current demo
# ---------------------------------------------------------------------

@dataclass
class ToolRequest:
    """
    Structured request created by the ToolRequestAgent and sent to the
    centralized ToolInteractionManager.
    """
    request_id: str
    student_question: str
    step_goal: str
    task_type: str
    expected_output: str
    workflow_source: str = "personalized_problem_tutoring"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def user_query(self) -> str:
        """
        Backward-compatible alias.
        Old modules use request.user_query.
        New clean schema uses request.student_question.
        """
        return self.student_question

    @property
    def query(self) -> str:
        """
        Generic alias for components that use request.query.
        """
        return self.student_question

    @property
    def current_step(self) -> str:
        """
        Backward-compatible alias.
        Old modules use request.current_step.
        New clean schema uses request.step_goal.
        """
        return self.step_goal

    @property
    def context(self) -> Dict[str, Any]:
        """
        Backward-compatible alias.
        Old modules use request.context.
        New clean schema uses request.metadata.
        """
        return self.metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "student_question": self.student_question,
            "user_query": self.student_question,
            "query": self.student_question,
            "step_goal": self.step_goal,
            "current_step": self.step_goal,
            "task_type": self.task_type,
            "expected_output": self.expected_output,
            "workflow_source": self.workflow_source,
            "metadata": self.metadata,
        }


@dataclass
class ToolPlan:
    """
    Ordered plan of tools to execute.
    """
    plan_id: str
    task_type: str
    selected_tools: List[str]
    steps: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_type": self.task_type,
            "selected_tools": self.selected_tools,
            "steps": self.steps,
            "metadata": self.metadata,
        }


@dataclass(init=False)
class ToolResult:
    """
    Result returned by one executed tool.

    This class is intentionally compatible with both naming styles:
    - new clean style: status/content/evidence/references
    - old demo style: success/output
    """

    tool_name: str
    status: str
    content: str
    success: bool
    evidence: List[str]
    references: List[str]
    metadata: Dict[str, Any]

    def __init__(
        self,
        tool_name: str,
        status: Optional[str] = None,
        content: Optional[str] = None,
        success: Optional[bool] = None,
        output: Optional[str] = None,
        evidence: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ):
        self.tool_name = tool_name

        if success is None:
            success = status in {"success", "validated", "ok", "passed"}

        self.success = bool(success)

        if status is None:
            status = "success" if self.success else "failed"

        self.status = status
        self.content = content if content is not None else (output if output is not None else "")
        self.evidence = evidence or []
        self.references = references or []
        self.metadata = metadata or {}

        if extra:
            self.metadata["extra_fields"] = extra

    @property
    def output(self) -> str:
        """
        Backward-compatible alias for old modules that use result.output.
        """
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "success": self.success,
            "content": self.content,
            "output": self.content,
            "evidence": self.evidence,
            "references": self.references,
            "metadata": self.metadata,
        }


# Backward-compatible alias if any file uses ToolExecutionResult.
ToolExecutionResult = ToolResult


@dataclass
class ValidationReport:
    """
    Validation result produced by the OutputValidator.
    """
    status: str
    confidence_score: float
    failure_reason: Optional[str] = None
    recommended_action: str = "accept"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "confidence_score": self.confidence_score,
            "failure_reason": self.failure_reason,
            "recommended_action": self.recommended_action,
        }


@dataclass
class OutputPackage:
    """
    Final package returned by the centralized ToolInteractionManager.
    """
    package_id: str
    status: str
    content: str
    evidence: List[str]
    references: List[str]
    validation_report: ValidationReport
    trace_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "status": self.status,
            "content": self.content,
            "evidence": self.evidence,
            "references": self.references,
            "validation_report": self.validation_report.to_dict()
            if hasattr(self.validation_report, "to_dict")
            else self.validation_report,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }
