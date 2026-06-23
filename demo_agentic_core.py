from ai.llm.errors import LLMProviderUnavailableError
import json

from ai.agentic.core.agentic_core_service import AgenticCoreService


def main():
    print("=" * 80)
    print("OpenTutorAI-Agentic Core Terminal Runner")
    print("Centralized T2-Based Adaptive Tool Interaction Manager")
    print("=" * 80)

    service = AgenticCoreService()

    result = service.handle_request(
        {
            "request_type": "personalized_tutoring",
            "learner_id": "demo_user",
            "query": "Explain convolution with a simple example.",
            "metadata": {
                "source": "terminal_runner",
                "force_first_failure": True,
            },
        }
    )

    print("\n" + "=" * 80)
    print("TERMINAL RUNNER SUMMARY")
    print("=" * 80)

    print("\nFinal Answer:")
    print(result["answer"])

    print("\nExecution Metadata:")
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Trace ID: {result['trace_id']}")
    print(f"Trace Path: {result['trace_path']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Selected Tools: {result['selected_tools']}")
    print(f"Recovery Used: {result['recovery_used']}")

    print("\nOutput Package:")
    print(json.dumps(result["output_package"], indent=2))

    print("\nDemo completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except LLMProviderUnavailableError:
        print()
        print("=" * 80)
        print("LLM PROVIDER UNAVAILABLE")
        print("=" * 80)
        print("The configured Groq/OpenAI-compatible LLM provider could not be reached.")
        print("Please check your internet connection, API key, provider status, or try again later.")
        print("No ungrounded or hard-coded tutoring answer was generated.")
        print("=" * 80)
        raise SystemExit(0)
