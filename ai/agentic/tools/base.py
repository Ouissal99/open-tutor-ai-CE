"""Base interface for external executable tools in the Tool Registry."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ai.agentic.core.schemas import ToolResult


class BaseTool(ABC):
    """Base class for external tools executed by the ToolExecutor."""

    name: str = "BaseTool"
    description: str = ""
    category: str = "general"
    risk_level: str = "low"
    requires_network: bool = False
    requires_sandbox: bool = False
    enabled: bool = True

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "risk_level": self.risk_level,
            "requires_network": self.requires_network,
            "requires_sandbox": self.requires_sandbox,
            "enabled": self.enabled,
        }

    @abstractmethod
    def run(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        raise NotImplementedError
