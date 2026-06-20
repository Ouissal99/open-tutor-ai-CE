from typing import Any, Dict

from ai.agentic.core.schemas import AgenticRequest
from learning.supports.agentic_tutoring.personalized_tutoring_workflow import PersonalizedTutoringWorkflow


class CapabilityRouter:
    """
    Routes an agentic request to the correct capability/workflow.

    In this first implementation phase, only one capability is fully supported:
    - personalized_tutoring

    Later, this router can be extended to:
    - question_generation
    - deep_research
    - assessment
    - visual_learning
    """

    def __init__(self):
        self.personalized_tutoring_workflow = PersonalizedTutoringWorkflow()

    def route(self, request: AgenticRequest) -> Dict[str, Any]:
        print("\n[CORE] CapabilityRouter received request")
        print(f"       request_type: {request.request_type}")

        if request.request_type == "personalized_tutoring":
            print("[CORE] Selected capability: PersonalizedTutoringWorkflow")

            return self.personalized_tutoring_workflow.run(
                student_question=request.query,
                learner_id=request.learner_id,
                metadata=request.metadata,
            )

        raise ValueError(f"Unsupported request_type: {request.request_type}")