import json
from pathlib import Path
from typing import Any, Dict, List


class DynamicPersonalMemory:
    """
    Lightweight DeepTutor-style Dynamic Personal Memory.

    This prototype follows the same role as DeepTutor DPM:
    it stores learner-specific context that helps personalize tutoring.

    Structure:
    - L1: raw/recent interaction traces
    - L2: tutoring notes and surface-level learning facts
    - L3: synthesized learner profile, preferences, recent memory
    """

    def __init__(self, base_dir: str = "data/agentic_memory/dpm"):
        self.base_dir = Path(base_dir)

    def load_profile(self, learner_id: str = "demo_user") -> Dict[str, Any]:
        learner_dir = self.base_dir / learner_id
        profile_path = learner_dir / "L3" / "profile.json"
        preferences_path = learner_dir / "L3" / "preferences.json"
        recent_path = learner_dir / "L3" / "recent.md"
        tutoring_path = learner_dir / "L2" / "tutoring.md"
        traces_path = learner_dir / "L1" / "traces.jsonl"

        profile = self._read_json(profile_path, default={
            "learner_id": learner_id,
            "level": "unknown",
            "known_topics": [],
            "weak_topics": [],
            "learning_goals": [],
        })

        preferences = self._read_json(preferences_path, default={
            "preferred_explanation_style": "unknown",
            "preferred_examples": [],
            "avoid": [],
        })

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

        is_weak_topic = topic.lower() in [item.lower() for item in weak_topics]
        is_known_topic = topic.lower() in [item.lower() for item in known_topics]

        return {
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

        with traces_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace_summary, ensure_ascii=False) + "\n")

    def _read_json(self, path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return default

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

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
