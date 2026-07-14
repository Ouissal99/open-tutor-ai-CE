"""CodeSandboxTool: restricted Python execution for educational code examples.

This is a prototype sandbox. It blocks dangerous imports/tokens and runs with
timeout. It is not a production-grade secure sandbox.

For production, replace this with Docker/firejail/microVM isolation.
"""

import multiprocessing
import queue
import sys
import traceback
from io import StringIO
from typing import Any, Dict

from ai.agentic.core.schemas import ToolResult
from ai.agentic.tools.base import BaseTool


BLOCKED_TOKENS = [
    "import os",
    "from os",
    "import subprocess",
    "from subprocess",
    "import socket",
    "from socket",
    "import shutil",
    "from shutil",
    "import pathlib",
    "from pathlib",
    "import sys",
    "from sys",
    "import importlib",
    "__import__",
    "open(",
    "eval(",
    "exec(",
    "compile(",
    "globals(",
    "locals(",
    "input(",
]


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Allow only small educational imports."""
    if name == "math":
        import math
        return math

    if name == "numpy" or name.startswith("numpy."):
        import numpy
        return numpy

    raise ImportError(f"Import '{name}' is not allowed in CodeSandboxTool.")


def _run_code(code: str, output_queue: multiprocessing.Queue) -> None:
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout = StringIO()
    stderr = StringIO()

    safe_builtins = {
        "__import__": _safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "zip": zip,
    }

    safe_globals = {
        "__builtins__": safe_builtins,
    }

    try:
        sys.stdout = stdout
        sys.stderr = stderr
        exec(code, safe_globals, safe_globals)

        output_queue.put(
            {
                "success": True,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "error": None,
            }
        )
    except Exception:
        output_queue.put(
            {
                "success": False,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "error": traceback.format_exc(),
            }
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class CodeSandboxTool(BaseTool):
    name = "CodeSandboxTool"
    description = "Safely execute small Python snippets for tutoring examples."
    category = "execution"
    risk_level = "high"
    requires_network = False
    requires_sandbox = True

    def run(
        self,
        request: Any,
        step: Dict[str, Any],
        collected_context: Dict[str, Any],
        analyzed_task: Dict[str, Any],
        attempt: int = 1,
    ) -> ToolResult:
        code = (
            step.get("code")
            or analyzed_task.get("code")
            or (getattr(request, "metadata", {}) or {}).get("code")
            or ""
        )

        if not code:
            return ToolResult(
                tool_name=self.name,
                status="failed",
                success=False,
                output="No code was provided to CodeSandboxTool.",
                evidence=[],
                references=[],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "failure_reason": "missing_code",
                    "registry_tool": True,
                    "requires_code": True,
                },
            )

        lowered = code.lower()
        blocked = [token for token in BLOCKED_TOKENS if token in lowered]

        if blocked:
            return ToolResult(
                tool_name=self.name,
                status="failed",
                success=False,
                output=(
                    "Code execution blocked for safety.\n\n"
                    "Candidate code:\n"
                    "```python\n"
                    f"{code}\n"
                    "```\n\n"
                    f"Blocked token(s): {blocked}"
                ),
                evidence=["The sandbox blocked potentially unsafe code."],
                references=["CodeSandboxTool safety policy"],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "failure_reason": "blocked_code",
                    "blocked_tokens": blocked,
                    "code": code,
                    "code_source": step.get("code_source"),
                    "code_draft": step.get("code_draft"),
                    "registry_tool": True,
                },
            )

        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_run_code, args=(code, output_queue))
        process.start()
        process.join(timeout=3)

        if process.is_alive():
            process.terminate()
            process.join()
            return ToolResult(
                tool_name=self.name,
                status="failed",
                success=False,
                output=(
                    "Code execution timed out.\n\n"
                    "Candidate code:\n"
                    "```python\n"
                    f"{code}\n"
                    "```"
                ),
                evidence=["The sandbox stopped execution after timeout."],
                references=["CodeSandboxTool timeout policy"],
                metadata={
                    "attempt": attempt,
                    "source_component": self.name,
                    "failure_reason": "timeout",
                    "code": code,
                    "code_source": step.get("code_source"),
                    "code_draft": step.get("code_draft"),
                    "registry_tool": True,
                },
            )

        try:
            result = output_queue.get_nowait()
        except queue.Empty:
            result = {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": "No output returned from sandbox process.",
            }

        success = bool(result.get("success"))

        output = (
            "Candidate Python code executed by CodeSandboxTool:\n"
            "```python\n"
            f"{code}\n"
            "```\n\n"
            "Execution result:\n"
            f"STDOUT:\n{result.get('stdout', '')}\n"
            f"STDERR:\n{result.get('stderr', '')}\n"
        )

        if result.get("error"):
            output += f"ERROR:\n{result.get('error')}"

        return ToolResult(
            tool_name=self.name,
            status="success" if success else "failed",
            success=success,
            output=output,
            evidence=[
                "CodeSandboxTool executed the LLM-generated Python snippet.",
                "Execution success status was returned to the OutputValidator.",
            ],
            references=["CodeSandboxTool local restricted Python sandbox"],
            metadata={
                "attempt": attempt,
                "source_component": self.name,
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
                "error": result.get("error"),
                "code": code,
                "code_source": step.get("code_source"),
                "code_draft": step.get("code_draft"),
                "registry_tool": True,
                "sandbox_type": "restricted_python_multiprocessing_timeout",
            },
        )
