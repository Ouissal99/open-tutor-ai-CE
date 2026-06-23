"""Seed demo Dynamic Personal Memory for the agentic tutoring prototype."""

import json
from pathlib import Path


BASE_DIR = Path("var/agentic_memory/dpm/demo_user")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    write_json(
        BASE_DIR / "L3" / "profile.json",
        {
            "learner_id": "demo_user",
            "level": "beginner",
            "known_topics": [
                "basic matrices",
                "basic image processing vocabulary",
            ],
            "weak_topics": [
                "convolution",
                "kernel movement",
                "feature maps",
            ],
            "learning_goals": [
                "understand convolution intuitively",
                "connect matrix operations with visual image processing",
            ],
        },
    )

    write_json(
        BASE_DIR / "L3" / "preferences.json",
        {
            "preferred_explanation_style": "simple_step_by_step",
            "preferred_examples": [
                "small matrix examples",
                "visual analogies",
                "beginner-friendly wording",
            ],
            "avoid": [
                "heavy mathematical notation at the beginning",
                "long abstract definitions without examples",
            ],
        },
    )

    write_text(
        BASE_DIR / "L3" / "recent.md",
        """
# Recent Memory

The learner recently asked about convolution and needed help understanding how a kernel moves over an input matrix.
The next explanation should remain beginner-friendly and include a small visual or matrix example.
""",
    )

    write_text(
        BASE_DIR / "L2" / "tutoring.md",
        """
# Tutoring Memory

The learner benefits from step-by-step explanations with concrete examples.

Recent observed needs:
- Needs simple explanations before formal definitions.
- Benefits from visual or matrix-based examples.
- Convolution and kernel movement are weak concepts.
""",
    )

    traces_path = BASE_DIR / "L1" / "traces.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    if not traces_path.exists():
        traces_path.write_text(
            json.dumps(
                {
                    "trace_type": "session_summary",
                    "topic": "convolution",
                    "summary": "The learner asked for a simple explanation of convolution and needed visual support for kernel movement.",
                    "timestamp": "seed",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"Seeded demo DPM at {BASE_DIR}")


if __name__ == "__main__":
    main()
