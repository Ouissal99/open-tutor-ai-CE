import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


class DynamicPersonalMemory:
    """
    Lightweight DeepTutor-style Dynamic Personal Memory.

    Structure:
    - L1: raw/recent interaction trace summaries
    - L2: tutoring notes and surface-level learning facts
    - L3: synthesized learner profile, preferences, recent memory
    """

    def __init__(self, base_dir: str = "var/agentic_memory/dpm"):
        self.base_dir = Path(base_dir)

    def load_profile(self, learner_id: str = "demo_user") -> Dict[str, Any]:
        learner_dir = self.base_dir / learner_id

        profile_path = learner_dir / "L3" / "profile.json"
        preferences_path = learner_dir / "L3" / "preferences.json"
        recent_path = learner_dir / "L3" / "recent.md"
        tutoring_path = learner_dir / "L2" / "tutoring.md"
        traces_path = learner_dir / "L1" / "traces.jsonl"

        profile, profile_loaded = self._read_json_with_status(
            profile_path,
            default={
                "learner_id": learner_id,
                "level": "unknown",
                "known_topics": [],
                "weak_topics": [],
                "learning_goals": [],
            },
        )

        preferences, preferences_loaded = self._read_json_with_status(
            preferences_path,
            default={
                "preferred_explanation_style": "unknown",
                "preferred_examples": [],
                "avoid": [],
            },
        )

        recent_memory = self._read_text(recent_path)
        tutoring_memory = self._read_text(tutoring_path)
        recent_traces = self._read_jsonl(traces_path)

        return {
            "learner_id": learner_id,
            "profile": profile,
            "preferences": preferences,
            "recent_memory": recent_memory,
            "tutoring_memory": tutoring_memory,
            "recent_traces": recent_traces,
            "profile_loaded": profile_loaded,
            "preferences_loaded": preferences_loaded,
            "recent_memory_loaded": bool(recent_memory.strip()),
            "tutoring_memory_loaded": bool(tutoring_memory.strip()),
            "recent_trace_count": len(recent_traces),
            "memory_paths": {
                "base_dir": str(learner_dir),
                "L1_traces": str(traces_path),
                "L2_tutoring": str(tutoring_path),
                "L3_profile": str(profile_path),
                "L3_preferences": str(preferences_path),
                "L3_recent": str(recent_path),
            },
        }

    def build_personalization_context(
        self,
        learner_id: str,
        topic: str,
        task_type: str,
    ) -> Dict[str, Any]:
        memory = self.load_profile(learner_id)

        profile = memory.get("profile", {})
        preferences = memory.get("preferences", {})

        weak_topics = profile.get("weak_topics", [])
        known_topics = profile.get("known_topics", [])

        normalized_topic = (topic or "").lower()
        is_weak_topic = normalized_topic in [str(item).lower() for item in weak_topics]
        is_known_topic = normalized_topic in [str(item).lower() for item in known_topics]

        return {
            "memory_source": "DynamicPersonalMemory",
            "memory_storage": "file_based_local",
            "memory_layers": ["L1_interaction_traces", "L2_tutoring_notes", "L3_profile_preferences"],
            "learner_id": learner_id,
            "learner_level": profile.get("level", "unknown"),
            "preferred_explanation_style": preferences.get(
                "preferred_explanation_style",
                "unknown",
            ),
            "preferred_examples": preferences.get("preferred_examples", []),
            "avoid": preferences.get("avoid", []),
            "known_topics": known_topics,
            "weak_topics": weak_topics,
            "learning_goals": profile.get("learning_goals", []),
            "is_weak_topic": is_weak_topic,
            "is_known_topic": is_known_topic,
            "recent_memory": memory.get("recent_memory", ""),
            "tutoring_memory": memory.get("tutoring_memory", ""),
            "recent_traces": memory.get("recent_traces", []),
            "recent_trace_count": memory.get("recent_trace_count", 0),
            "profile_loaded": memory.get("profile_loaded", False),
            "preferences_loaded": memory.get("preferences_loaded", False),
            "recent_memory_loaded": memory.get("recent_memory_loaded", False),
            "tutoring_memory_loaded": memory.get("tutoring_memory_loaded", False),
            "memory_paths": memory.get("memory_paths", {}),
            "topic": topic,
            "task_type": task_type,
        }

    def append_l1_trace_summary(
        self,
        learner_id: str,
        trace_summary: Dict[str, Any],
    ) -> None:
        traces_path = self.base_dir / learner_id / "L1" / "traces.jsonl"
        traces_path.parent.mkdir(parents=True, exist_ok=True)

        row = dict(trace_summary)
        row.setdefault("memory_layer", "L1_interaction_trace_summary")
        row.setdefault("stored_at", datetime.now(timezone.utc).isoformat())

        with traces_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _read_json_with_status(
        self,
        path: Path,
        default: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        if not path.exists():
            return default, False

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), True
        except Exception:
            return default, False

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""

        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []

        rows = []

        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        except Exception:
            return rows

        return rows
