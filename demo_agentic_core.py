import json
from ai.agentic.tutoring.personalized_tutoring_workflow import PersonalizedTutoringWorkflow


def main():
    print("=" * 80)
    print("OpenTutorAI-Agentic Core Demo")
    print("Centralized T2-Based Adaptive Tool Interaction Manager")
    print("=" * 80)

    workflow = PersonalizedTutoringWorkflow()

    result = workflow.run(
        "Explain convolution with a simple example."
    )

    print("\n[7] Final answer generated")
    print(result["final_answer"])

    print("\n[8] Scratchpad state")
    print(json.dumps(result["scratchpad"], indent=2, ensure_ascii=False))

    print("\n[9] Tool interaction trace saved")
    print(result["trace_path"])

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
