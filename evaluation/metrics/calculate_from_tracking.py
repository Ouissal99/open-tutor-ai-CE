import csv
from pathlib import Path

TRACKING_FILE = Path("evaluation/metrics/evaluation_tracking.csv")
OUT_DIR = Path("evaluation/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_tools(value):
    if not value:
        return set()
    return set(tool.strip() for tool in value.split(";") if tool.strip())

def yes(value):
    return str(value).strip().lower() in {"yes", "true", "1"}

def safe_div(a, b):
    return 0 if b == 0 else a / b

rows = []

with TRACKING_FILE.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        expected = parse_tools(row["expected_tools"])
        selected = parse_tools(row["selected_tools"])
        correct = expected.intersection(selected)

        precision = safe_div(len(correct), len(selected))
        recall = safe_div(len(correct), len(expected))
        f1 = safe_div(2 * precision * recall, precision + recall)

        row_result = {
            "test_id": row["test_id"],
            "category": row["category"],
            "workflow_completed": 1 if row["workflow_status"] == "validated" else 0,
            "validation_success": 1 if row["validation_status"] == "valid" else 0,
            "recovery_used": 1 if yes(row["recovery_used"]) else 0,
            "attempts": int(row["attempts"]),
            "tool_precision": precision,
            "tool_recall": recall,
            "tool_f1": f1,
            "trace_saved": 1 if row["trace_path"] else 0,
            "memory_updated": 1 if yes(row["memory_updated"]) else 0,
            "final_answer_generated": 1 if yes(row["final_answer_generated"]) else 0,
        }
        rows.append(row_result)

N = len(rows)
recovery_cases = [r for r in rows if r["category"] == "recovery"]

summary = {
    "Number of test cases": N,
    "Workflow Completion Rate": safe_div(sum(r["workflow_completed"] for r in rows), N) * 100,
    "Output Validation Success Rate": safe_div(sum(r["validation_success"] for r in rows), N) * 100,
    "Tool Selection Precision": safe_div(sum(r["tool_precision"] for r in rows), N) * 100,
    "Tool Selection Recall": safe_div(sum(r["tool_recall"] for r in rows), N) * 100,
    "Tool Selection F1": safe_div(sum(r["tool_f1"] for r in rows), N) * 100,
    "Recovery Success Rate": safe_div(
        sum(1 for r in recovery_cases if r["recovery_used"] == 1 and r["validation_success"] == 1),
        len(recovery_cases)
    ) * 100 if recovery_cases else "N/A",
    "Average Number of Attempts": safe_div(sum(r["attempts"] for r in rows), N),
    "pass@3": safe_div(sum(1 for r in rows if r["workflow_completed"] == 1 and r["attempts"] <= 3), N) * 100,
    "Trace Saved Rate": safe_div(sum(r["trace_saved"] for r in rows), N) * 100,
    "Memory Update Success Rate": safe_div(sum(r["memory_updated"] for r in rows), N) * 100,
    "Final Answer Generation Rate": safe_div(sum(r["final_answer_generated"] for r in rows), N) * 100,
}

per_test_path = OUT_DIR / "pilot_per_test_metrics.csv"
global_path = OUT_DIR / "pilot_global_metrics.csv"

with per_test_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

with global_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    for metric, value in summary.items():
        writer.writerow([metric, value])

print("Saved:", per_test_path)
print("Saved:", global_path)
print()
for metric, value in summary.items():
    print(f"{metric}: {value}")
