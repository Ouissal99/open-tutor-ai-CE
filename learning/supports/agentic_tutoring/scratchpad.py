from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class ScratchpadRound:
    round_id: int
    step_goal: str
    analysis: str
    tool_request_summary: str = ""
    output_package: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"


class StepBasedScratchpad:
    """
    Passive working memory for the current tutoring process.
    """

    def __init__(self):
        self.rounds: List[ScratchpadRound] = []

    def add_round(self, step_goal: str, analysis: str) -> ScratchpadRound:
        round_item = ScratchpadRound(
            round_id=len(self.rounds) + 1,
            step_goal=step_goal,
            analysis=analysis
        )
        self.rounds.append(round_item)
        return round_item

    def update_round_with_package(self, round_id: int, request_summary: str, package) -> None:
        for round_item in self.rounds:
            if round_item.round_id == round_id:
                round_item.tool_request_summary = request_summary
                round_item.output_package = package.to_dict()
                round_item.status = package.status
                return

    def to_dict(self):
        return [asdict(round_item) for round_item in self.rounds]
