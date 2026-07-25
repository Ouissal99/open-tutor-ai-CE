"""Grounding guard for final agentic tutoring answers.

This module protects the final LLM answer from adding unsupported calculations.
The LLM may still write naturally, but verified tool outputs remain the source of truth.
"""

import re
from typing import Any, Dict, List


class GroundedAnswerGuard:
    """Validate final tutoring answers against the validated OutputPackage."""

    def validate_or_fallback(
        self,
        answer: str,
        student_question: str,
        topic: str,
        package_dict: Dict[str, Any],
        dpm_context: Dict[str, Any],
    ) -> str:
        if not answer.strip():
            return self._build_fallback(
                topic=topic,
                package_dict=package_dict,
                dpm_context=dpm_context,
                reason="empty_llm_answer",
            )

        reason = self._find_grounding_failure(
            answer=answer,
            topic=topic,
            package_dict=package_dict,
        )

        if reason:
            return self._build_fallback(
                topic=topic,
                package_dict=package_dict,
                dpm_context=dpm_context,
                reason=reason,
            )

        return answer

    def _find_grounding_failure(
        self,
        answer: str,
        topic: str,
        package_dict: Dict[str, Any],
    ) -> str | None:
        if topic != "convolution":
            return None

        trusted_text = self._trusted_text(package_dict)
        answer_lower = answer.lower()

        if "middle patch" in answer_lower:
            return "unsupported_middle_patch"
        if "center patch" in answer_lower:
            return "unsupported_center_patch"

        unsupported_dimension_claims = (
            "same dimensions as the input",
            "same dimensions as input",
            "same size as the input",
            "same size as input",
            "output matrix will have the same dimensions",
            "output will have the same dimensions",
            "output retains the input dimensions",
            "output preserves the input dimensions",
        )

        for claim in unsupported_dimension_claims:
            if (
                claim in answer_lower
                and self._normalize_formula_text(claim)
                not in trusted_text
            ):
                return (
                    "unsupported_convolution_dimension_claim"
                )

        trusted_digits = re.sub(
            r"\D",
            "",
            trusted_text,
        )

        for raw_line in answer.splitlines():
            candidate_line = raw_line.strip().strip(
                "-*`| "
            )

            # Matrix rows and numeric arrays must already exist in the
            # validated package. This catches invented rows such as
            # "14 18 24" while allowing verified rows.
            if re.fullmatch(
                r"[\[\]\d,.;\s+\-]+",
                candidate_line,
            ):
                numeric_tokens = re.findall(
                    r"-?\d+(?:\.\d+)?",
                    candidate_line,
                )

                if len(numeric_tokens) >= 2:
                    numeric_signature = re.sub(
                        r"\D",
                        "",
                        "".join(numeric_tokens),
                    )

                    if (
                        numeric_signature
                        and numeric_signature
                        not in trusted_digits
                    ):
                        return (
                            "unsupported_numeric_matrix_row"
                        )

        for raw_line in answer.splitlines():
            line = raw_line.strip().strip("-*` ")

            if not line:
                continue

            has_formula = (
                "=" in line
                and ("×" in line or "*" in line or " x " in line.lower())
            )

            if not has_formula:
                continue

            normalized_line = self._normalize_formula_text(line)
            if normalized_line and normalized_line not in trusted_text:
                return "unsupported_formula_line"

        return None

    def _trusted_text(self, package_dict: Dict[str, Any]) -> str:
        content = package_dict.get("content", "") or ""
        evidence = package_dict.get("evidence", []) or []
        combined = content + "\n" + "\n".join(str(item) for item in evidence)
        return self._normalize_formula_text(combined)

    def _normalize_formula_text(self, text: str) -> str:
        text = text.lower()
        text = text.replace("×", "x")
        text = text.replace("−", "-")
        text = text.replace("`", "")
        text = text.replace("*", "")
        text = re.sub(r"\s+", "", text)
        return text

    def _build_fallback(
        self,
        topic: str,
        package_dict: Dict[str, Any],
        dpm_context: Dict[str, Any],
        reason: str,
    ) -> str:
        if topic == "convolution":
            return self._build_convolution_fallback(
                package_dict=package_dict,
                dpm_context=dpm_context,
                reason=reason,
            )
        return self._build_general_fallback(package_dict=package_dict, reason=reason)

    def _build_convolution_fallback(
        self,
        package_dict: Dict[str, Any],
        dpm_context: Dict[str, Any],
        reason: str,
    ) -> str:
        content = package_dict.get("content", "") or ""
        evidence = package_dict.get("evidence", []) or []
        references = package_dict.get("references", []) or []
        learner_level = dpm_context.get("learner_level") or "beginner"
        matrix_section = self._extract_matrix_section(content)
        evidence_lines = self._format_list(evidence, max_items=5)
        reference_lines = self._format_list(references, max_items=5)

        return (
            "**Convolution explained simply**\n\n"
            f"For a {learner_level} learner, think of convolution as a small matrix "
            "called a **kernel** sliding over an input matrix. At each position, the "
            "kernel is applied to a local patch of the input to produce one output value.\n\n"
            "**Verified example from the tool output**\n\n"
            "```text\n"
            f"{matrix_section}\n"
            "```\n\n"
            "The important point is that the tutor must use the verified "
            "values shown above exactly as returned by the validated tools.\n\n"
            "**Evidence used**\n\n"
            f"{evidence_lines}\n\n"
            "**References**\n\n"
            f"{reference_lines}\n\n"
            f"_Grounded fallback used because the first LLM draft contained unsupported "
            f"calculation details: {reason}._"
        )

    def _build_general_fallback(self, package_dict: Dict[str, Any], reason: str) -> str:
        content = package_dict.get("content", "") or ""
        evidence = package_dict.get("evidence", []) or []
        references = package_dict.get("references", []) or []
        evidence_lines = self._format_list(evidence, max_items=5)
        reference_lines = self._format_list(references, max_items=5)

        return (
            "**Grounded tutoring answer**\n\n"
            "The answer is based only on the validated tool package:\n\n"
            "```text\n"
            f"{content[:1500]}\n"
            "```\n\n"
            "**Evidence used**\n\n"
            f"{evidence_lines}\n\n"
            "**References**\n\n"
            f"{reference_lines}\n\n"
            f"_Grounded fallback used because: {reason}._"
        )

    def _extract_matrix_section(self, content: str) -> str:
        marker = "MatrixComputationTool computed"
        if marker not in content:
            return content[:1200].strip()

        start = content.find(marker)
        end = len(content)

        for next_marker in (
            "\n\n## ",
            " Course evidence retrieved",
            " Trace guidance retrieved",
            " Generated visual explanation",
        ):
            index = content.find(next_marker, start + 1)
            if index != -1:
                end = min(end, index)

        return content[start:end].strip()

    def _format_list(self, items: List[Any], max_items: int = 5) -> str:
        clean_items = []
        seen = set()
        for item in items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            clean_items.append(text)
            if len(clean_items) >= max_items:
                break
        if not clean_items:
            return "- No evidence provided."
        return "\n".join(f"- {item}" for item in clean_items)
