# Evaluation Source Provenance

The official 100-case evaluation artifacts record the following Git source commit:

`104974a200cca6e4abc58016977715d43a7bd736`

The Full Manager package also records the tag:

`full-manager-official-evaluation-v1`

At repository packaging time, the working tree contained uncommitted modifications to the following files:

- `ai/agentic/memory/trace_toolkit.py`
- `ai/agentic/tools/calculator_tool.py`
- `ai/agentic/tools/matrix_computation_tool.py`
- `ai/agentic/tools/rag_tool.py`
- `evaluation/run_case.py`

No separate source snapshot of these modified files was stored with the evaluation run. Therefore, the recorded commit identifies the committed repository base, while the exact relationship between the uncommitted working-tree modifications and the generated evaluation artifacts cannot be reconstructed independently.

The published evaluation package was integrity-checked before release:

- 593 checksum entries verified
- 0 missing files
- 0 checksum mismatches
- 4 non-checksum manifest header lines
