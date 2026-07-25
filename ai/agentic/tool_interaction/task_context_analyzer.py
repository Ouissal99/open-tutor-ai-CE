"""TaskContextAnalyzer for the centralized Tool Interaction Manager.

This component interprets the incoming tutoring/tool request before context
retrieval and tool selection.

It understands:
- workflow source
- task type
- topic / goal
- expected output
- code / visual / calculation intent

Context retrieval is handled by ContextCollector, which completes the
architecture-level "Task & Context Analysis" stage by retrieving:
- SKG grounding
- DPM learner profile
- previous tool traces
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import threading
from typing import Any, Dict, Optional, List

from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport


class TaskContextAnalyzer:
    """
    Request-understanding stage of the Tool Interaction Manager.

    The main path uses the configured LLM provider to produce structured task
    analysis. A deterministic fallback is kept only for provider failure, so the
    prototype remains runnable during local evaluation.
    """

    VALID_TASK_TYPES = {
        "conceptual_explanation",
        "visual_explanation",
        "calculation_or_verification",
        "code_execution",
        "code_help",
        "programming",
        "debugging",
        "personalized_support",
        "unknown_task",
    }

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        model: Optional[str] = None,
    ):
        self.llm_service = llm_service or LLMService(OpenAICompatibleTransport())
        self.model = model or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant"

    def analyze(self, request: Any) -> Dict[str, Any]:
        base_payload = self._base_payload(request)

        try:
            llm_analysis = self._run_async(self._analyze_with_llm(base_payload))
            analysis = self._normalize_analysis(base_payload, llm_analysis)
            analysis["analysis_strategy"] = "llm_structured_task_analysis"
            return analysis

        except Exception as exc:
            fallback = self._fallback_analysis(base_payload)
            fallback["analysis_strategy"] = "deterministic_fallback_after_llm_failure"
            fallback["analysis_error"] = str(exc)
            return fallback

    async def _analyze_with_llm(self, base_payload: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
You are the Task & Context Analyzer inside a centralized Tool Interaction Manager.

Your job is to understand the incoming tutoring request before context retrieval
and tool selection.

Return ONLY valid JSON. No markdown. No explanation.

The JSON must contain:
{
  "task_type": "...",
  "topic": "...",
  "goal": "...",
  "expected_output": "...",
  "workflow_source": "...",
  "learner_level": "...",
  "needs_visual_support": true/false,
  "needs_calculation": true/false,
  "needs_code_execution": true/false,
  "needs_debugging": true/false,
  "needs_grounding": true/false,
  "code_generation_required": true/false,
  "reasoning_summary": "...",
  "tool_request_goal": "...",
  "code_generation_instruction": "..."
}

Rules:
1. If the request asks for Python, NumPy, code, implementation, debugging, or executable example, set task_type to "code_execution".
2. If the request asks to compute or verify a numerical/matrix result, set task_type to "calculation_or_verification".
3. If the request asks to show, visualize, illustrate, or explain visually, set task_type to "visual_explanation".
4. If it asks for an explanation only, use "conceptual_explanation".
5. For code tasks, code_generation_instruction should tell a later LLM code-drafting step what kind of small safe Python snippet should be generated.
6. Do not retrieve memory or knowledge here. Only analyze the request.
""".strip()

        user_prompt = json.dumps(base_payload, ensure_ascii=False, indent=2)

        response = await self.llm_service.complete(
            LLMRequest(
                model=self.model,
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
                temperature=0.0,
                max_tokens=700,
            )
        )

        return self._parse_json(response.completion)

    def _base_payload(self, request: Any) -> Dict[str, Any]:
        metadata = getattr(request, "metadata", {}) or {}
        context = getattr(request, "context", {}) or {}

        user_query = (
            getattr(request, "student_question", None)
            or getattr(request, "user_query", None)
            or getattr(request, "query", None)
            or ""
        )

        return {
            "request_id": getattr(request, "request_id", None),
            "workflow_source": getattr(request, "workflow_source", None),
            "incoming_task_type": getattr(request, "task_type", "unknown_task"),
            "user_query": user_query,
            "student_question": user_query,
            "current_step": getattr(request, "current_step", ""),
            "expected_output": getattr(request, "expected_output", ""),
            "context": context,
            "metadata": metadata,
            "learner_level": context.get("learner_level", "beginner"),
            "force_recovery_test": bool(metadata.get("force_recovery_test")),
        }

    def _normalize_analysis(
        self,
        base_payload: Dict[str, Any],
        llm_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        global_text = str(base_payload.get("user_query", "") or "").lower()
        step_text = " ".join(
            str(base_payload.get(key, "") or "")
            for key in ["current_step", "expected_output"]
        ).lower()

        full_text = " ".join(
            [global_text, step_text]
        ).lower()

        explicit_visual_requested = (
            self._explicit_visual_requested(
                global_text
            )
        )

        no_code_phrases = (
            "without code",
            "no code",
            "avoid code",
            "do not use code",
            "don't use code",
            "code is confusing",
            "code confusing",
        )
        no_code_requested = any(
            phrase in global_text
            for phrase in no_code_phrases
        )

        task_type = str(llm_analysis.get("task_type") or base_payload.get("incoming_task_type") or "unknown_task")

        if task_type not in self.VALID_TASK_TYPES:
            task_type = self._infer_task_type(full_text, base_payload.get("incoming_task_type", "unknown_task"))

        topic = str(
            llm_analysis.get("topic")
            or self._infer_topic(full_text)
        )
        goal = str(
            llm_analysis.get("goal")
            or base_payload.get("user_query")
            or ""
        )

        retrieval_only_step = (
            self._is_retrieval_only_step(
                step_text
            )
        )

        operation_analysis = (
            self._resolve_step_operation(
                global_text=global_text,
                step_text=step_text,
                retrieval_only_step=(
                    retrieval_only_step
                ),
            )
        )

        incoming_task_type = str(base_payload.get("incoming_task_type") or "")
        incoming_is_code = (
            incoming_task_type
            in {"code_execution", "code_help", "programming", "debugging"}
            and not no_code_requested
        )

        step_has_code_intent = (
            self._has_code_intent(step_text)
            and not no_code_requested
        )
        global_has_code_intent = (
            self._has_code_intent(global_text)
            and not no_code_requested
        )

        # Important:
        # A global student question may ask for code, but an individual scratchpad
        # step may only ask for retrieval/grounding. In that case, do not force
        # CodeSandboxTool into the retrieval step.
        needs_code_execution = (
            not no_code_requested
            and not retrieval_only_step
            and (
                step_has_code_intent
                or incoming_is_code
                or (
                    bool(llm_analysis.get("needs_code_execution"))
                    and not self._is_retrieval_only_step(step_text)
                )
                or (
                    global_has_code_intent
                    and not step_text.strip()
                )
            )
        )

        needs_debugging = (
            not retrieval_only_step
            and (bool(llm_analysis.get("needs_debugging")) or self._has_debug_intent(step_text + " " + global_text))
        )

        needs_calculation = (
            not retrieval_only_step
            and (bool(llm_analysis.get("needs_calculation")) or self._has_calculation_intent(step_text))
        )

        needs_visual_support = (
            not retrieval_only_step
            and explicit_visual_requested
        )

        if no_code_requested and task_type in {
            "code_execution",
            "code_help",
            "programming",
            "debugging",
        }:
            task_type = "conceptual_explanation"
            needs_code_execution = False

        if retrieval_only_step:
            task_type = "conceptual_explanation"
            needs_code_execution = False
            needs_debugging = False
            needs_calculation = False
            needs_visual_support = False
        elif needs_code_execution:
            task_type = "code_execution"
        elif needs_calculation:
            task_type = "calculation_or_verification"
        elif needs_visual_support:
            task_type = "visual_explanation"

        return {
            "request_id": base_payload.get("request_id"),
            "workflow_source": base_payload.get("workflow_source"),
            "original_task_type": base_payload.get("incoming_task_type"),
            "task_type": task_type,
            "operation_type": operation_analysis[
                "operation_type"
            ],
            "operation_parameters": operation_analysis[
                "operation_parameters"
            ],
            "topic": topic,
            "goal": goal,
            "current_step": base_payload.get("current_step"),
            "student_question": base_payload.get("student_question"),
            "user_query": base_payload.get("user_query"),
            "expected_output": llm_analysis.get("expected_output") or base_payload.get("expected_output"),
            "learner_level": llm_analysis.get("learner_level") or base_payload.get("learner_level"),
            "needs_visual_support": needs_visual_support,
            "explicit_visual_requested": (
                explicit_visual_requested
            ),
            "needs_calculation": needs_calculation,
            "needs_code_execution": needs_code_execution,
            "needs_debugging": needs_debugging,
            "needs_grounding": bool(llm_analysis.get("needs_grounding", True)),
            "code_generation_required": bool(llm_analysis.get("code_generation_required")) or needs_code_execution,
            "tool_request_goal": llm_analysis.get("tool_request_goal") or goal,
            "code_generation_instruction": (
                llm_analysis.get("code_generation_instruction")
                or self._default_code_instruction(full_text)
            ),
            "reasoning_summary": llm_analysis.get("reasoning_summary", ""),
        }

    def _fallback_analysis(self, base_payload: Dict[str, Any]) -> Dict[str, Any]:
        global_text = str(base_payload.get("user_query", "") or "").lower()
        step_text = " ".join(
            str(base_payload.get(key, "") or "")
            for key in ["current_step", "expected_output"]
        ).lower()

        full_text = " ".join(
            [global_text, step_text]
        ).lower()

        explicit_visual_requested = (
            self._explicit_visual_requested(
                global_text
            )
        )

        retrieval_only_step = (
            self._is_retrieval_only_step(step_text)
        )

        operation_analysis = (
            self._resolve_step_operation(
                global_text=global_text,
                step_text=step_text,
                retrieval_only_step=(
                    retrieval_only_step
                ),
            )
        )

        incoming_task_type = str(base_payload.get("incoming_task_type") or "")
        incoming_is_code = incoming_task_type in {"code_execution", "code_help", "programming", "debugging"}

        needs_code_execution = (
            not retrieval_only_step
            and (
                self._has_code_intent(step_text)
                or incoming_is_code
                or (self._has_code_intent(global_text) and not step_text.strip())
            )
        )
        needs_debugging = not retrieval_only_step and self._has_debug_intent(step_text + " " + global_text)
        needs_calculation = not retrieval_only_step and self._has_calculation_intent(step_text)
        needs_visual_support = (
            not retrieval_only_step
            and explicit_visual_requested
        )

        if retrieval_only_step:
            task_type = "conceptual_explanation"
        else:
            task_type = self._infer_task_type(full_text, base_payload.get("incoming_task_type", "unknown_task"))

        return {
            "request_id": base_payload.get("request_id"),
            "workflow_source": base_payload.get("workflow_source"),
            "original_task_type": base_payload.get("incoming_task_type"),
            "task_type": task_type,
            "operation_type": operation_analysis[
                "operation_type"
            ],
            "operation_parameters": operation_analysis[
                "operation_parameters"
            ],
            "topic": self._infer_topic(full_text),
            "goal": base_payload.get("user_query", ""),
            "current_step": base_payload.get("current_step"),
            "student_question": base_payload.get("student_question"),
            "user_query": base_payload.get("user_query"),
            "expected_output": base_payload.get("expected_output"),
            "learner_level": base_payload.get("learner_level"),
            "needs_visual_support": needs_visual_support,
            "explicit_visual_requested": (
                explicit_visual_requested
            ),
            "needs_calculation": needs_calculation,
            "needs_code_execution": needs_code_execution,
            "needs_debugging": needs_debugging,
            "needs_grounding": True,
            "code_generation_required": needs_code_execution,
            "tool_request_goal": base_payload.get("user_query", ""),
            "code_generation_instruction": self._default_code_instruction(full_text),
            "reasoning_summary": "Fallback task analysis generated from request text.",
        }

    def _infer_task_type(self, text: str, incoming_task_type: str) -> str:
        if self._has_code_intent(text):
            return "code_execution"
        if self._has_calculation_intent(text):
            return "calculation_or_verification"
        if self._has_visual_intent(text):
            return "visual_explanation"
        if incoming_task_type:
            return incoming_task_type
        return "conceptual_explanation"

    def _infer_topic(self, text: str) -> str:
        if "convolution" in text or "kernel" in text or "cnn" in text:
            return "convolution"
        if "edge" in text:
            return "edge_detection"
        if "padding" in text:
            return "padding"
        if "stride" in text:
            return "stride"
        if "overfitting" in text:
            return "overfitting"
        if "gradient" in text:
            return "gradient descent"
        return "general topic"

    def _is_retrieval_only_step(self, text: str) -> bool:
        text = (text or "").lower()

        retrieval_terms = [
            "retrieve",
            "retrieval",
            "course-grounded evidence",
            "grounded evidence",
            "grounding",
            "rag",
            "search evidence",
            "collect evidence",
            "course evidence",
        ]

        code_terms = [
            "code",
            "python",
            "numpy",
            "implement",
            "script",
            "program",
            "debug",
        ]

        has_retrieval = any(term in text for term in retrieval_terms)
        has_code = any(term in text for term in code_terms)

        return has_retrieval and not has_code

    def _has_code_intent(self, text: str) -> bool:
        return any(
            word in text
            for word in [
                "code",
                "python",
                "numpy",
                "debug",
                "implement",
                "function",
                "script",
                "program",
                "snippet",
            ]
        )

    def _has_debug_intent(self, text: str) -> bool:
        return any(word in text for word in ["debug", "fix", "error", "wrong", "instead of"])

    def _has_calculation_intent(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "calculate",
                "compute",
                "output size",
                "result",
                "apply",
                "formula",
                "matrix",
            ]
        )

    def _has_visual_intent(self, text: str) -> bool:
        return any(
            phrase in text
            for phrase in [
                "visual",
                "show",
                "illustrate",
                "visualize",
                "draw",
                "moves",
                "movement",
            ]
        )

    def _default_code_instruction(self, text: str) -> str:
        if not self._has_code_intent(text):
            return ""

        return (
            "Generate a minimal, safe, executable Python snippet that answers the "
            "student request. The code must be deterministic, beginner-friendly, "
            "avoid unsafe imports and file/network operations, and print its result."
        )

    def _explicit_visual_requested(
        self,
        global_text: str,
    ) -> bool:
        text = str(global_text or "").lower()

        return any(
            phrase in text
            for phrase in (
                "visualize",
                "visualise",
                "show visually",
                "visual explanation",
                "draw a diagram",
                "draw",
                "diagram",
                "illustrate",
                "illustration",
                "text-based visualization",
                "text based visualization",
            )
        )

    def _conceptual_operation_for_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        normalized = str(text or "").lower()

        if any(
            term in normalized
            for term in (
                "convolution",
                "kernel",
                "cross-correlation",
                "padding",
                "stride",
            )
        ):
            operation_type = "convolution_concept"

        elif any(
            term in normalized
            for term in (
                "matrix",
                "matrices",
                "row-by-column",
                "row by column",
                "dot product",
            )
        ):
            operation_type = "matrix_concept"

        else:
            operation_type = (
                "conceptual_explanation"
            )

        return {
            "operation_type": operation_type,
            "operation_parameters": {
                "matrices": [],
                "shapes": [],
                "stride": 1,
                "padding": 0,
            },
        }

    def _resolve_step_operation(
        self,
        global_text: str,
        step_text: str,
        retrieval_only_step: bool,
    ) -> Dict[str, Any]:
        """
        Resolve the operation for one tutoring step.

        The global question supplies the original operation and
        parameters. The current step determines whether that operation
        should be executed now or whether the step is only conceptual
        or retrieval-oriented.
        """
        global_analysis = self._infer_operation(
            global_text
        )

        if retrieval_only_step:
            return self._conceptual_operation_for_text(
                " ".join(
                    [global_text, step_text]
                )
            )

        if not str(step_text or "").strip():
            return global_analysis

        step_analysis = self._infer_operation(
            step_text
        )

        global_operation = str(
            global_analysis.get("operation_type")
            or ""
        )

        step_lower = str(step_text or "").lower()

        explicit_execution_cues = any(
            phrase in step_lower
            for phrase in (
                "calculate",
                "compute",
                "apply",
                "verify",
                "solve",
                "determine",
                "perform",
                "multiply",
                "product",
                "first entry",
                "top-left",
                "top left",
                "dot product",
                "how many",
                "number of",
                "result",
            )
        )

        conceptual_cues = any(
            phrase in step_lower
            for phrase in (
                "explain",
                "understand",
                "define",
                "describe",
                "discuss",
                "introduce",
                "concept",
                "why",
                "difference",
                "compare",
            )
        )

        # A dimension-compatibility question may be expressed through
        # an explanatory step such as “explain the conditions”.
        if (
            global_operation
            == "matrix_compatibility"
            and any(
                phrase in step_lower
                for phrase in (
                    "condition",
                    "precondition",
                    "dimension",
                    "compatible",
                    "can be multiplied",
                    "determine",
                    "verify",
                    "check",
                    "example",
                )
            )
        ):
            return global_analysis

        if (
            conceptual_cues
            and not explicit_execution_cues
        ):
            return self._conceptual_operation_for_text(
                " ".join(
                    [global_text, step_text]
                )
            )

        deterministic_operations = {
            "scalar_arithmetic",
            "convolution_output_size",
            "matrix_compatibility",
            "matrix_multiplication",
            "matrix_entry",
            "dot_product",
            "valid_2d_convolution",
            "convolution_window_count",
            "kernel_fit",
        }

        if (
            global_operation
            in deterministic_operations
            and (
                explicit_execution_cues
                or step_analysis.get(
                    "operation_type"
                )
                == global_operation
            )
        ):
            return global_analysis

        return step_analysis

    def _infer_operation(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract the requested operation and its parameters.

        This remains part of Task & Context Analyzer. It does not
        choose or execute tools.
        """
        original_text = str(text or "")
        normalized = (
            original_text.lower()
            .replace("×", "x")
            .replace("–", "-")
            .replace("−", "-")
        )

        matrices = self._extract_matrix_literals(
            original_text
        )

        shapes = [
            [int(rows), int(cols)]
            for rows, cols in re.findall(
                r"\b(\d+)\s*x\s*(\d+)\b",
                normalized,
            )
        ]

        parameters: Dict[str, Any] = {
            "matrices": matrices,
            "shapes": shapes,
        }

        input_shape = self._extract_labeled_shape(
            normalized,
            labels=(
                "input",
                "input matrix",
                "matrix",
            ),
        )

        kernel_shape = self._extract_labeled_shape(
            normalized,
            labels=(
                "kernel",
                "filter",
                "window",
            ),
        )

        if input_shape:
            parameters["input_shape"] = input_shape

        if kernel_shape:
            parameters["kernel_shape"] = kernel_shape

        stride_match = re.search(
            r"\bstride\s*(?:=|of|is)?\s*(\d+)\b",
            normalized,
        )

        padding_match = re.search(
            r"\bpadding\s*(?:=|of|is)?\s*(\d+)\b",
            normalized,
        )

        parameters["stride"] = (
            int(stride_match.group(1))
            if stride_match
            else 1
        )

        parameters["padding"] = (
            int(padding_match.group(1))
            if padding_match
            else 0
        )

        has_convolution = any(
            term in normalized
            for term in (
                "convolution",
                "kernel",
                "cross-correlation",
                "sliding window",
            )
        )

        output_size_requested = any(
            phrase in normalized
            for phrase in (
                "output size",
                "output shape",
                "output dimension",
                "output dimensions",
            )
        )

        window_count_requested = (
            has_convolution
            and any(
                phrase in normalized
                for phrase in (
                    "how many valid",
                    "number of valid",
                    "convolution windows",
                    "windows exist",
                )
            )
        )

        kernel_fit_requested = (
            has_convolution
            and any(
                phrase in normalized
                for phrase in (
                    "can slide",
                    "can the kernel",
                    "kernel fit",
                    "fits inside",
                )
            )
        )

        matrix_compatibility_requested = (
            "matrix" in normalized
            and any(
                phrase in normalized
                for phrase in (
                    "can a",
                    "can the",
                    "can be multiplied",
                    "whether",
                    "multiplication possible",
                    "dimensions matter",
                )
            )
            and (
                "multipl" in normalized
                or "product" in normalized
            )
        )

        matrix_entry_requested = any(
            phrase in normalized
            for phrase in (
                "first entry",
                "top-left entry",
                "top left entry",
                "first element",
                "c[1,1]",
            )
        )

        matrix_multiplication_requested = any(
            phrase in normalized
            for phrase in (
                "multiply the matrices",
                "multiply matrices",
                "matrix multiplication",
                "matrix product",
                "product of the matrices",
            )
        )

        dot_product_requested = any(
            phrase in normalized
            for phrase in (
                "dot product",
                "row-by-column",
                "row by column",
                "multiplying row",
            )
        )

        explicit_convolution = (
            has_convolution
            and len(matrices) >= 2
            and any(
                phrase in normalized
                for phrase in (
                    "compute",
                    "calculate",
                    "valid convolution",
                    "top-left convolution",
                    "convolution step",
                )
            )
        )

        if has_convolution and output_size_requested:
            operation_type = "convolution_output_size"

        elif window_count_requested:
            operation_type = "convolution_window_count"

            if len(shapes) >= 2:
                parameters["kernel_shape"] = shapes[0]
                parameters["input_shape"] = shapes[1]

        elif kernel_fit_requested:
            operation_type = "kernel_fit"

            if len(shapes) >= 2:
                parameters["kernel_shape"] = shapes[0]
                parameters["input_shape"] = shapes[1]

        elif explicit_convolution:
            operation_type = "valid_2d_convolution"

            parameters["input_matrix"] = matrices[0]
            parameters["kernel"] = matrices[1]

        elif (
            matrix_compatibility_requested
            and len(matrices) < 2
            and len(shapes) >= 2
        ):
            operation_type = "matrix_compatibility"

            if len(shapes) >= 2:
                parameters["left_shape"] = shapes[0]
                parameters["right_shape"] = shapes[1]

        elif matrix_entry_requested and len(matrices) >= 2:
            operation_type = "matrix_entry"

            parameters["matrix_a"] = matrices[0]
            parameters["matrix_b"] = matrices[1]
            parameters["entry"] = [0, 0]

        elif (
            (
                matrix_multiplication_requested
                or "multiplied by" in normalized
                or "identity matrix" in normalized
            )
            and len(matrices) >= 2
        ):
            operation_type = "matrix_multiplication"

            parameters["matrix_a"] = matrices[0]
            parameters["matrix_b"] = matrices[1]

        elif dot_product_requested:
            operation_type = "dot_product"

            vectors = self._extract_vector_literals(
                original_text
            )

            if len(vectors) >= 2:
                parameters["left_vector"] = vectors[0]
                parameters["right_vector"] = vectors[1]

        else:
            scalar_expression = (
                self._extract_scalar_expression(
                    normalized
                )
            )

            if scalar_expression:
                operation_type = "scalar_arithmetic"
                parameters["expression"] = (
                    scalar_expression
                )

            elif has_convolution:
                operation_type = "convolution_concept"

            elif "matrix" in normalized:
                operation_type = "matrix_concept"

            else:
                operation_type = "conceptual_explanation"

        return {
            "operation_type": operation_type,
            "operation_parameters": parameters,
        }

    def _extract_labeled_shape(
        self,
        text: str,
        labels,
    ) -> Optional[List[int]]:
        for label in labels:
            pattern = (
                rf"\b{re.escape(label)}\b"
                r"(?:\s+(?:size|shape|dimensions?))?"
                r"\s*(?:=|is|of|:)?\s*"
                r"(\d+)\s*x\s*(\d+)"
            )

            match = re.search(pattern, text)

            if match:
                return [
                    int(match.group(1)),
                    int(match.group(2)),
                ]

        return None

    def _extract_matrix_literals(
        self,
        text: str,
    ) -> List[List[List[float]]]:
        results = []
        index = 0
        source = str(text or "")

        while index < len(source) - 1:
            start = source.find("[[", index)

            if start == -1:
                break

            depth = 0
            end = None

            for position in range(start, len(source)):
                char = source[position]

                if char == "[":
                    depth += 1

                elif char == "]":
                    depth -= 1

                    if depth == 0:
                        end = position + 1
                        break

            if end is None:
                break

            literal = source[start:end]

            try:
                value = ast.literal_eval(literal)
            except Exception:
                index = start + 2
                continue

            if self._is_numeric_matrix(value):
                results.append(value)

            index = end

        return results

    def _extract_vector_literals(
        self,
        text: str,
    ) -> List[List[float]]:
        matrices = self._extract_matrix_literals(text)
        masked = str(text or "")

        for matrix in matrices:
            masked = masked.replace(
                str(matrix),
                "",
            )

        vectors = []

        for literal in re.findall(
            r"(?<!\[)\[[^\[\]]+\](?!\])",
            masked,
        ):
            try:
                value = ast.literal_eval(literal)
            except Exception:
                continue

            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(item, (int, float))
                    for item in value
                )
            ):
                vectors.append(value)

        return vectors

    def _is_numeric_matrix(
        self,
        value: Any,
    ) -> bool:
        if not isinstance(value, list) or not value:
            return False

        if not all(
            isinstance(row, list) and row
            for row in value
        ):
            return False

        width = len(value[0])

        if any(len(row) != width for row in value):
            return False

        return all(
            isinstance(item, (int, float))
            for row in value
            for item in row
        )

    def _extract_scalar_expression(
        self,
        text: str,
    ) -> Optional[str]:
        normalized = str(text or "").lower()

        matrix_context = any(
            term in normalized
            for term in (
                "matrix",
                "matrices",
                "kernel",
                "convolution",
                "input shape",
                "output shape",
                "dimensions",
            )
        )

        squared_match = re.search(
            r"\b(-?\d+(?:\.\d+)?)\s+squared\s*"
            r"(plus|\+|minus|-)\s*"
            r"(-?\d+(?:\.\d+)?)\s+squared\b",
            normalized,
        )

        if squared_match:
            left, operator_word, right = (
                squared_match.groups()
            )

            operator_symbol = (
                "+"
                if operator_word in {"plus", "+"}
                else "-"
            )

            return (
                f"{left} ** 2 "
                f"{operator_symbol} "
                f"{right} ** 2"
            )

        binary_match = re.search(
            r"\b(-?\d+(?:\.\d+)?)\s*"
            r"(times|multiplied\s+by|\*|plus|\+|"
            r"minus|-|divided\s+by|/)\s*"
            r"(-?\d+(?:\.\d+)?)\b",
            normalized,
        )

        if binary_match:
            left, raw_operator, right = (
                binary_match.groups()
            )

            operator_map = {
                "times": "*",
                "multiplied by": "*",
                "*": "*",
                "plus": "+",
                "+": "+",
                "minus": "-",
                "-": "-",
                "divided by": "/",
                "/": "/",
            }

            operator_key = re.sub(
                r"\s+",
                " ",
                raw_operator.strip(),
            )

            return (
                f"{left} "
                f"{operator_map[operator_key]} "
                f"{right}"
            )

        # The character x is allowed as scalar multiplication only
        # when the surrounding request is not about matrix/kernel
        # dimensions.
        if not matrix_context:
            x_match = re.search(
                r"\b(-?\d+(?:\.\d+)?)\s*x\s*"
                r"(-?\d+(?:\.\d+)?)\b",
                normalized,
            )

            if x_match:
                return (
                    f"{x_match.group(1)} * "
                    f"{x_match.group(2)}"
                )

        return None

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        data = json.loads(text)

        if not isinstance(data, dict):
            raise ValueError("LLM task analysis did not return a JSON object.")

        return data

    def _run_async(self, coroutine):
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                result_box = {}
                error_box = {}

                def runner():
                    try:
                        result_box["result"] = asyncio.run(coroutine)
                    except Exception as exc:
                        error_box["error"] = exc

                thread = threading.Thread(target=runner)
                thread.start()
                thread.join()

                if "error" in error_box:
                    raise error_box["error"]

                return result_box.get("result")
        except RuntimeError:
            pass

        return asyncio.run(coroutine)
