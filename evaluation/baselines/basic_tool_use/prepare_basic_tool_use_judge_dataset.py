import json
from pathlib import Path

src = Path("evaluation/baselines/basic_tool_use/basic_tool_use_results_30_tests.jsonl")
out = Path("evaluation/baselines/basic_tool_use/basic_tool_use_judge_dataset_30_tests.jsonl")

rows = [
    json.loads(line)
    for line in src.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

with out.open("w", encoding="utf-8") as f:
    for r in rows:
        evidence_parts = []
        for t in r.get("tool_outputs", []):
            evidence_parts.append(f'## {t.get("tool_name", "")}\n{t.get("output", "")}')

        item = {
            "test_id": r["test_id"],
            "category": r.get("category", ""),
            "question": r.get("question", ""),
            "expected_tools": r.get("expected_tools", ""),
            "selected_tools": r.get("selected_tools", ""),
            "workflow_status": r.get("status", ""),
            "validation_status": "not_applicable_no_validator",
            "recovery_used": "No",
            "attempts": "1",
            "trace_path": "",
            "report_path": "",
            "final_answer": r.get("final_answer", ""),
            "evidence_excerpt": "\n\n".join(evidence_parts) or "No evidence available.",
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Saved:", out)
print("Items:", len(rows))
