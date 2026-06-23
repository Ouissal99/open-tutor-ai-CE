import re
from typing import Any, Dict, List, Optional, Set

from ai.agentic.memory.trace_store import TraceStore


class TraceToolkit:
    """
    Query layer over saved tool interaction traces.

    TraceStore = low-level file loading and summarization.
    TraceToolkit = higher-level search and reuse.
    """

    def __init__(self, trace_store: Optional[TraceStore] = None):
        self.trace_store = trace_store or TraceStore()

    def get_last_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        traces = self.trace_store.list_traces()
        return [self.trace_store.summarize_trace(trace) for trace in traces[:limit]]

    def find_successful_traces(
        self,
        task_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return traces that ended successfully."""
        results = []

        for trace in self.trace_store.list_traces():
            summary = self.trace_store.summarize_trace(trace)

            if task_type and summary["task_type"] != task_type:
                continue

            if self._is_successful(trace, summary):
                results.append(summary)

            if len(results) >= limit:
                break

        return results

    def find_failed_traces(
        self,
        task_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return traces that ended as failed, invalid, rejected, or requiring retry."""
        results = []

        for trace in self.trace_store.list_traces():
            summary = self.trace_store.summarize_trace(trace)

            if task_type and summary["task_type"] != task_type:
                continue

            if self._is_failed(trace, summary):
                results.append(summary)

            if len(results) >= limit:
                break

        return results

    def find_similar_traces(
        self,
        query: str,
        task_type: Optional[str] = None,
        limit: int = 5,
        only_successful: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Lightweight similarity search over real saved traces.

        The score is based on meaningful fields only:
        - student question
        - step goal
        - task type
        - selected tools
        - content preview
        """
        query_words = self._normalize_words(query)
        scored_results = []

        for trace in self.trace_store.list_traces():
            summary = self.trace_store.summarize_trace(trace)

            if task_type and summary["task_type"] != task_type:
                continue

            if only_successful and not self._is_successful(trace, summary):
                continue

            trace_words = self._summary_words(summary)
            overlap = query_words.intersection(trace_words)

            score = float(len(overlap))
            score_breakdown = {
                "word_overlap": len(overlap),
                "task_type_boost": 0.0,
                "tool_quality_boost": 0.0,
                "confidence_boost": 0.0,
            }

            if task_type and summary["task_type"] == task_type:
                score += 3.0
                score_breakdown["task_type_boost"] = 3.0

            selected_tools = set(summary.get("selected_tools", []))
            if {"RAGTool", "TraceSearchTool"} <= selected_tools:
                score += 1.0
                score_breakdown["tool_quality_boost"] += 1.0

            if "MatrixComputationTool" in selected_tools:
                score += 0.75
                score_breakdown["tool_quality_boost"] += 0.75

            if "VisualMatrixTool" in selected_tools:
                score += 0.75
                score_breakdown["tool_quality_boost"] += 0.75

            confidence = summary.get("confidence_score")
            if isinstance(confidence, (int, float)):
                score += min(float(confidence), 1.0)
                score_breakdown["confidence_boost"] = min(float(confidence), 1.0)

            if score > 0:
                summary["similarity_score"] = round(score, 4)
                summary["matched_terms"] = sorted(overlap)
                summary["score_breakdown"] = score_breakdown
                summary["memory_source"] = "real_trace_files"
                scored_results.append(summary)

        scored_results.sort(
            key=lambda item: (
                item.get("similarity_score", 0),
                item.get("created_at") or "",
            ),
            reverse=True,
        )
        return scored_results[:limit]

    def get_trace_summary(self, trace_id_or_path: str) -> Optional[Dict[str, Any]]:
        trace = self.trace_store.load_trace(trace_id_or_path)

        if not trace:
            return None

        return self.trace_store.summarize_trace(trace)

    def _is_successful(self, trace: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        failure_reason = summary.get("failure_reason")
        if failure_reason is not None and str(failure_reason).strip().lower() not in {"", "none", "null"}:
            return False

        status = str(summary.get("status", "")).lower().strip()

        successful_statuses = {
            "success",
            "validated",
            "valid",
            "passed",
            "accepted",
            "ok",
        }

        failed_statuses = {
            "invalid",
            "failed",
            "failure",
            "rejected",
            "retry",
            "weak_grounding",
        }

        if status in failed_statuses:
            return False

        if status in successful_statuses:
            return True

        text = str(trace).lower()

        if '"status": "invalid"' in text or "'status': 'invalid'" in text:
            return False

        if '"recommended_action": "retry"' in text or "'recommended_action': 'retry'" in text:
            return False

        if '"recommended_action": "reject"' in text or "'recommended_action': 'reject'" in text:
            return False

        if '"recommended_action": "accept"' in text or "'recommended_action': 'accept'" in text:
            return True

        if '"status": "validated"' in text or '"status": "valid"' in text:
            return True

        return False

    def _is_failed(self, trace: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        failure_reason = summary.get("failure_reason")
        if failure_reason is not None and str(failure_reason).strip().lower() not in {"", "none", "null"}:
            return True

        status = str(summary.get("status", "")).lower().strip()

        failed_statuses = {
            "invalid",
            "failed",
            "failure",
            "rejected",
            "retry",
            "weak_grounding",
        }

        successful_statuses = {
            "success",
            "validated",
            "valid",
            "passed",
            "accepted",
            "ok",
        }

        if status in failed_statuses:
            return True

        if status in successful_statuses:
            return False

        text = str(trace).lower()

        failure_markers = [
            '"status": "invalid"',
            "'status': 'invalid'",
            '"status": "failed"',
            "'status': 'failed'",
            '"recommended_action": "retry"',
            "'recommended_action': 'retry'",
            '"recommended_action": "reject"',
            "'recommended_action': 'reject'",
        ]

        return any(marker in text for marker in failure_markers)

    def _summary_words(self, summary: Dict[str, Any]) -> Set[str]:
        parts = [
            summary.get("student_question", ""),
            summary.get("step_goal", ""),
            summary.get("task_type", ""),
            summary.get("content_preview", ""),
            " ".join(summary.get("selected_tools", [])),
        ]

        return self._normalize_words(" ".join(str(part) for part in parts if part))

    def _normalize_words(self, text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "to",
            "in",
            "with",
            "for",
            "is",
            "are",
            "be",
            "this",
            "that",
            "it",
            "as",
            "by",
            "on",
            "how",
            "what",
            "why",
            "use",
            "using",
            "explain",
            "simple",
            "example",
            "student",
            "learner",
        }

        return {
            token
            for token in tokens
            if token not in stopwords and len(token) >= 4
        }
