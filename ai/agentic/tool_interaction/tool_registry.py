"""Central registry for external tools available to the ToolExecutor."""

from typing import Dict, List

from ai.agentic.tools.base import BaseTool
from ai.agentic.tools.rag_tool import RAGTool
from ai.agentic.tools.trace_search_tool import TraceSearchTool
from ai.agentic.tools.visual_matrix_tool import VisualMatrixTool
from ai.agentic.tools.matrix_computation_tool import MatrixComputationTool
from ai.agentic.tools.calculator_tool import CalculatorTool
from ai.agentic.tools.code_sandbox_tool import CodeSandboxTool


class ToolRegistry:
    """Extensible registry for external executable tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(RAGTool())
        self.register(TraceSearchTool())
        self.register(VisualMatrixTool())
        self.register(MatrixComputationTool())
        self.register(CalculatorTool())
        self.register(CodeSandboxTool())

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError("Tool must define a non-empty name.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            available = ", ".join(sorted(self._tools))
            raise KeyError(f"Tool '{name}' not found in registry. Available tools: {available}")

        tool = self._tools[name]

        if not tool.enabled:
            raise RuntimeError(f"Tool '{name}' is registered but disabled.")

        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe(self) -> Dict[str, dict]:
        return {
            name: tool.metadata()
            for name, tool in self._tools.items()
        }
