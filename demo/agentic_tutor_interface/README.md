# OpenTutorAI Agentic Tutoring Demo

This lightweight Streamlit interface demonstrates the thesis prototype:

**T2-Based Adaptive Tool Interaction Manager for OpenTutorAI**

The interface directly calls `AgenticCoreService.handle_request()` and displays:

- workflow status and confidence;
- selected tools;
- execution attempts and recovery use;
- the learner-facing answer;
- a compact Tool Interaction Trace preview.

## Requirements

Install the backend dependencies and configure the LLM provider through environment variables.

Do not place API keys directly in source files.

## Run

From the repository root:

```bash
python -m streamlit run \
  demo/agentic_tutor_interface/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

## Generated data

Runtime traces and reports are written under `var/` and are excluded from version control.

The repository includes only a small static SKG seed required for the demonstration.
