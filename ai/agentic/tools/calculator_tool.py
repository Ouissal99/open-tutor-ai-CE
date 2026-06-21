"""CalculatorTool: safe arithmetic support for tutoring."""

import ast
import operator
from typing import Any, Dict

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorTool(BaseTool):
    name = "CalculatorTool"
    description = "Evaluate safe arithmetic expressions for tutoring."
    category = "computation"
    risk_level = "low"
    requires_network = False
    requires_sandbox = False

    def run(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> ToolResult:
        expression = (
            step.get("expression")
            or analyzed_task.get("expression")
            or request.metadata.get("expression")
            or ""
        )

        if not expression:
            return ToolResult(
                tool_name=self.name,
                status="success",
                success=True,
                output="CalculatorTool available, but no arithmetic expression was provided.",
                evidence=["No calculation was requested."],
                references=["CalculatorTool"],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "operation": "none",
                    "registry_tool": True,
                },
            )

        try:
            result = self._safe_eval(expression)
            return ToolResult(
                tool_name=self.name,
                status="success",
                success=True,
                output=f"CalculatorTool computed: {expression} = {result}",
                evidence=[f"The arithmetic result of {expression} is {result}."],
                references=["CalculatorTool safe arithmetic evaluator"],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "expression": expression,
                    "result": result,
                    "registry_tool": True,
                },
            )

        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="failed",
                success=False,
                output=f"CalculatorTool failed to evaluate expression: {exc}",
                evidence=[],
                references=["CalculatorTool safety policy"],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "expression": expression,
                    "failure_reason": "calculation_failed",
                    "error": str(exc),
                    "registry_tool": True,
                },
            )

    def _safe_eval(self, expression: str):
        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Operator {op_type.__name__} is not allowed.")
            return _ALLOWED_OPERATORS[op_type](
                self._eval_node(node.left),
                self._eval_node(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_OPERATORS:
                raise ValueError(f"Unary operator {op_type.__name__} is not allowed.")
            return _ALLOWED_OPERATORS[op_type](self._eval_node(node.operand))

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")
