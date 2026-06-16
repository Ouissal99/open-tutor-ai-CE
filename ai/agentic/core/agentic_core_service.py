from typing import Any, Dict

from ai.agentic.core.capability_router import CapabilityRouter
from ai.agentic.core.schemas import AgenticRequest, AgenticResponse


class AgenticCoreService:
    """
    Main backend entry point of the OpenTutorAI-Agentic prototype.

    This service receives a structured request, sends it to the CapabilityRouter,
    normalizes the workflow result, and returns a clean dictionary.

    This becomes the reusable backend logic for:
    - terminal runner
    - evaluation
    - future API
    - future UI
    """

    def __init__(self):
        self.router = CapabilityRouter()

    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        request = self._build_request(request_data)

        print("\n[CORE] AgenticCoreService received request")
        print(f"       learner_id: {request.learner_id}")
        print(f"       query: {request.query}")

        workflow_result = self.router.route(request)
        response = self._build_response(workflow_result)

        return response.to_dict()

    def _build_request(self, request_data: Dict[str, Any]) -> AgenticRequest:
        if "query" not in request_data or not request_data["query"]:
            raise ValueError("Agentic request must contain a non-empty 'query' field.")

        return AgenticRequest(
            request_type=request_data.get("request_type", "personalized_tutoring"),
            query=request_data["query"],
            learner_id=request_data.get("learner_id", "demo_user"),
            metadata=request_data.get("metadata", {}),
        )

    def _build_response(self, workflow_result: Dict[str, Any]) -> AgenticResponse:
        return AgenticResponse(
            answer=workflow_result.get("answer") or workflow_result.get("final_answer", ""),
            status=workflow_result.get("status", "unknown"),
            confidence=workflow_result.get("confidence"),
            trace_id=workflow_result.get("trace_id"),
            trace_path=workflow_result.get("trace_path"),
            attempts=workflow_result.get("attempts", 1),
            selected_tools=workflow_result.get("selected_tools", []),
            recovery_used=workflow_result.get("recovery_used", False),
            output_package=workflow_result.get("output_package", {}),
            scratchpad=workflow_result.get("scratchpad", []),
            metadata={
                "agentic_core": "AgenticCoreService",
                **workflow_result.get("metadata", {}),
            },
        )