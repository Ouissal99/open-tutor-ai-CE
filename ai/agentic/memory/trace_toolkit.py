from typing import Any, Dict, List, Optional

from ai.agentic.memory.trace_store import TraceStore


class TraceToolkit:
    """
    Query layer over saved tool interaction traces.

    TraceStore = low-level file loading.
    TraceToolkit = higher-level search and reuse.

    This component will later be used by the Memory-Aware Tool Selector.
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
        """
        Return traces that ended successfully.
        Invalid traces must not be returned here.
        """
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
        """
        Return traces that ended as failed, invalid, rejected, or requiring retry.
        """
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
        Simple similarity search for the prototype.

        For now, similarity is keyword-based:
        - same task_type
        - shared words between query and trace content

        If only_successful=True, invalid traces are excluded.
        """
        query_words = self._normalize_words(query)
        scored_results = []

        for trace in self.trace_store.list_traces():
            summary = self.trace_store.summarize_trace(trace)

            if task_type and summary["task_type"] != task_type:
                continue

            if only_successful and not self._is_successful(trace, summary):
                continue

            trace_text = str(trace).lower()
            trace_words = self._normalize_words(trace_text)

            overlap = query_words.intersection(trace_words)
            score = len(overlap)

            if task_type and summary["task_type"] == task_type:
                score += 2

            if score > 0:
                summary["similarity_score"] = score
                scored_results.append(summary)

        scored_results.sort(key=lambda item: item["similarity_score"], reverse=True)
        return scored_results[:limit]

    def get_trace_summary(self, trace_id_or_path: str) -> Optional[Dict[str, Any]]:
        trace = self.trace_store.load_trace(trace_id_or_path)

        if not trace:
            return None

        return self.trace_store.summarize_trace(trace)

    def _is_successful(self, trace: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        """
        Strict success detection.

        Important:
        Do not use substring matching like "valid" in status,
        because "invalid" contains "valid".
        """
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

        if '"recommended_action": "accept"' in text or "'recommended_action': 'accept'" in text:
            return True

        if '"status": "validated"' in text or '"status": "valid"' in text:
            return True

        return False

    def _is_failed(self, trace: Dict[str, Any], summary: Dict[str, Any]) -> bool:
        """
        Strict failure detection.

        Important:
        A successful trace may contain "failure_reason": null.
        So we must not classify a trace as failed just because the key exists.
        """
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

        if status in successful_statuses:
            return False

        if status in failed_statuses:
            return True

        text = str(trace).lower()

        failure_markers = [
            '"status": "invalid"',
            "'status': 'invalid'",
            '"status": "failed"',
            "'status': 'failed'",
            "weak_grounding",
            '"recommended_action": "retry"',
            "'recommended_action': 'retry'",
            '"recommended_action": "reject"',
            "'recommended_action': 'reject'",
        ]

        return any(marker in text for marker in failure_markers)

    def _normalize_words(self, text: str) -> set:
        cleaned = (
            text.lower()
            .replace(".", " ")
            .replace(",", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace('"', " ")
            .replace("'", " ")
        )

        words = set()

        for word in cleaned.split():
            if len(word) >= 4:
                words.add(word)

        return words
