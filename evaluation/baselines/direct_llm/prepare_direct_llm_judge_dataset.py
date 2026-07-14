import json
from pathlib import Path

src = Path("evaluation/baselines/direct_llm/direct_llm_answers_30_tests.jsonl")
out = Path("evaluation/baselines/direct_llm/direct_llm_judge_dataset_30_tests.jsonl")

rows = [
    json.loads(line)
    for line in src.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

with out.open("w", encoding="utf-8") as f:
    for r in rows:
        item = {
            "test_id": r["test_id"],
            "category": r.get("category", ""),
            "question": r.get("question", ""),
            "expected_tools": "No tools in Direct LLM baseline",
            "selected_tools": "None",
            "workflow_status": "completed",
            "validation_status": "not_applicable",
            "recovery_used": "No",
            "attempts": "1",
            "trace_path": "",
            "report_path": "",
            "final_answer": r.get("final_answer", ""),
            "evidence_excerpt": "Direct LLM baseline: no external tools, no RAG evidence, no trace, no validator, no recovery.",
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Saved:", out)
print("Items:", len(rows))
