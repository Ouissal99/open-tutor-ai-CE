import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.agentic.tool_interaction.tool_registry import ToolRegistry
from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport

TEST_CASES = Path("evaluation/test_cases/test_cases.csv")
BASE = Path("evaluation/baselines/basic_tool_use")
OUT_JSONL = BASE / "basic_tool_use_results_30_tests.jsonl"
OUT_CSV = BASE / "basic_tool_use_results_30_tests.csv"
OUT_METRICS = BASE / "basic_tool_use_metrics.txt"

def load_cases():
    with TEST_CASES.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def load_previous():
    rows = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["test_id"]] = row
    return rows

def split_tools(s):
    return sorted([x.strip() for x in str(s).split(";") if x.strip()])

def static_select_tools(case):
    q = case["question"].lower()
    category = case["category"].lower()

    tools = set()

    # Basic direct-tool rules. No manager, no memory-aware selector, no recovery.
    if any(w in q for w in ["explain", "what is", "why", "padding", "stride", "convolution", "kernel", "filter", "feature map"]):
        tools.add("RAGTool")

    if any(w in q for w in ["previous", "mistake", "learner", "beginner", "confuses", "again", "personalized"]):
        tools.add("TraceSearchTool")

    if category in {"visual", "personalized", "recovery"} or any(w in q for w in ["visual", "show", "illustrate", "moves", "kernel moves", "padding", "stride"]):
        tools.add("VisualMatrixTool")

    if category in {"calculation", "code", "recovery"} or any(w in q for w in ["compute", "calculate", "result", "output size", "apply", "matrix", "3x3", "2x2"]):
        tools.add("MatrixComputationTool")

    if any(w in q for w in ["code", "python", "numpy", "script", "program"]):
        tools.add("CodeSandboxTool")

    if not tools:
        tools.add("RAGTool")

    return sorted(tools)

def build_context():
    return {
        "dpm": {
            "learner_level": "beginner",
            "preferred_examples": ["small matrix examples", "visual analogies", "beginner-friendly wording"],
        },
        "summary": {
            "similar_successful_trace_count": 0,
            "failed_trace_count": 0,
        },
        "skg": {
            "snippets": [],
            "grounding_text": "",
        },
        "trace_memory": {},
    }

def build_step(tool_name, question):
    step = {
        "purpose": f"Basic direct use of {tool_name} for: {question}",
        "tool_name": tool_name,
    }

    # For CodeSandboxTool, provide simple safe code directly.
    if tool_name == "CodeSandboxTool":
        step["code"] = """
import numpy as np

input_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

kernel = np.array([
    [1, 0],
    [0, 1]
])

out_rows = input_matrix.shape[0] - kernel.shape[0] + 1
out_cols = input_matrix.shape[1] - kernel.shape[1] + 1
output = np.zeros((out_rows, out_cols), dtype=int)

for i in range(out_rows):
    for j in range(out_cols):
        patch = input_matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]]
        output[i, j] = np.sum(patch * kernel)

print(output)
""".strip()

    return step

def tool_result_to_dict(result):
    return {
        "tool_name": getattr(result, "tool_name", ""),
        "status": getattr(result, "status", ""),
        "success": bool(getattr(result, "success", False)),
        "output": getattr(result, "output", ""),
        "evidence": getattr(result, "evidence", []),
        "references": getattr(result, "references", []),
        "metadata": getattr(result, "metadata", {}),
    }

async def write_answer(llm, model, case, selected_tools, tool_outputs):
    evidence_text = []

    for tr in tool_outputs:
        evidence_text.append(f'## {tr["tool_name"]}\n{tr.get("output", "")}')

    evidence_joined = "\n\n".join(evidence_text)

    prompt = f"""
You are writing the final answer for a BASIC TOOL USE baseline.

Baseline condition:
- Tools may be used directly by simple rules.
- There is NO centralized Tool Interaction Manager.
- There is NO tool planning lifecycle.
- There is NO output validator.
- There is NO failure recovery.
- There is NO trace lifecycle or memory update.
- Do not claim the output was validated or recovered.

Student question:
{case["question"]}

Directly selected tools:
{", ".join(selected_tools)}

Direct tool outputs:
{evidence_joined}

Write a clear tutoring answer using only the direct tool outputs when useful.
""".strip()

    request = LLMRequest(
        model=model,
        messages=[
            Message(role="system", content="You are a basic direct-tool tutoring baseline answer writer."),
            Message(role="user", content=prompt),
        ],
        temperature=0.0,
        max_tokens=1200,
    )

    response = await llm.complete(request)
    return response.completion.strip()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=6.0)
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("BASELINE_LLM_MODEL") or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant",
    )
    args = parser.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    previous = load_previous()

    done = {tid for tid, r in previous.items() if r.get("status") == "ok" and len(r.get("final_answer", "")) > 50}
    todo = [case for case in cases if case["test_id"] not in done]

    print("Existing OK:", len(done))
    print("To run:", [c["test_id"] for c in todo])

    registry = ToolRegistry()
    llm = LLMService(OpenAICompatibleTransport())

    for i, case in enumerate(todo, start=1):
        tid = case["test_id"]
        print(f"[{i}/{len(todo)}] Basic tool baseline {tid}...")

        selected_tools = static_select_tools(case)
        expected_tools = split_tools(case["expected_tools"])
        request_obj = SimpleNamespace(
            student_question=case["question"],
            query=case["question"],
            learner_id="baseline_basic_tool_user",
        )
        analyzed_task = {
            "topic": "convolution",
            "task_type": case["category"],
            "current_step": case["question"],
        }
        context = build_context()

        tool_outputs = []
        error = ""

        try:
            for tool_name in selected_tools:
                tool = registry.get(tool_name)
                step = build_step(tool_name, case["question"])
                result = tool.run(
                    request=request_obj,
                    step=step,
                    collected_context=context,
                    analyzed_task=analyzed_task,
                    attempt=1,
                )
                tool_outputs.append(tool_result_to_dict(result))

            final_answer = await write_answer(llm, args.model, case, selected_tools, tool_outputs)

            status = "ok" if final_answer and len(final_answer) > 50 else "weak_answer"

        except Exception as exc:
            final_answer = ""
            status = "error"
            error = str(exc)

        selected_set = set(selected_tools)
        expected_set = set(expected_tools)
        tp = len(selected_set & expected_set)
        precision = tp / len(selected_set) if selected_set else 0
        recall = tp / len(expected_set) if expected_set else 0
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        row = {
            "test_id": tid,
            "category": case["category"],
            "question": case["question"],
            "baseline": "basic_tool_use_without_central_manager",
            "expected_tools": ";".join(expected_tools),
            "selected_tools": ";".join(selected_tools),
            "tool_execution_success": "Yes" if tool_outputs and all(t.get("success") for t in tool_outputs) else "No",
            "validation_used": "No",
            "recovery_used": "No",
            "trace_lifecycle_used": "No",
            "memory_update_used": "No",
            "attempts": "1",
            "tool_precision": round(precision, 4),
            "tool_recall": round(recall, 4),
            "tool_f1": round(f1, 4),
            "status": status,
            "final_answer": final_answer,
            "tool_outputs": tool_outputs,
            "error": error,
        }

        previous[tid] = row

        with OUT_JSONL.open("w", encoding="utf-8") as f:
            for original in cases:
                saved = previous.get(original["test_id"])
                if saved:
                    f.write(json.dumps(saved, ensure_ascii=False) + "\n")

        await asyncio.sleep(args.sleep)

    ordered = [previous[c["test_id"]] for c in cases if c["test_id"] in previous]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "test_id", "category", "question", "baseline",
            "expected_tools", "selected_tools",
            "tool_execution_success", "validation_used", "recovery_used",
            "trace_lifecycle_used", "memory_update_used", "attempts",
            "tool_precision", "tool_recall", "tool_f1",
            "status", "final_answer", "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    ok = [r for r in ordered if r.get("status") == "ok"]
    tool_ok = [r for r in ordered if r.get("tool_execution_success") == "Yes"]

    avg_precision = sum(float(r["tool_precision"]) for r in ordered) / len(ordered)
    avg_recall = sum(float(r["tool_recall"]) for r in ordered) / len(ordered)
    avg_f1 = sum(float(r["tool_f1"]) for r in ordered) / len(ordered)

    lines = []
    lines.append("BASIC TOOL USE WITHOUT CENTRAL MANAGER BASELINE")
    lines.append("=" * 70)
    lines.append(f"Total tests: {len(ordered)}")
    lines.append(f"Answer Generation Rate: {len(ok)/len(ordered)*100:.2f}% ({len(ok)}/{len(ordered)})")
    lines.append(f"Tool Execution Success Rate: {len(tool_ok)/len(ordered)*100:.2f}% ({len(tool_ok)}/{len(ordered)})")
    lines.append(f"Tool Selection Precision: {avg_precision*100:.2f}%")
    lines.append(f"Tool Selection Recall: {avg_recall*100:.2f}%")
    lines.append(f"Tool Selection F1-score: {avg_f1*100:.2f}%")
    lines.append("Validation Used Rate: 0.00%")
    lines.append("Recovery Used Rate: 0.00%")
    lines.append("Trace Lifecycle Used Rate: 0.00%")
    lines.append("Memory Update Used Rate: 0.00%")
    lines.append("Average Attempts: 1.00")
    lines.append("")
    lines.append("Weak/error cases:")
    for r in ordered:
        if r.get("status") != "ok":
            lines.append(f'{r["test_id"]}: status={r.get("status")} error={r.get("error","")[:200]}')

    OUT_METRICS.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_METRICS)

if __name__ == "__main__":
    asyncio.run(main())
