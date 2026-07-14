import csv
from pathlib import Path

path = Path("evaluation/metrics/evaluation_tracking.csv")

with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

def split_tools(value):
    return {x.strip() for x in str(value).split(";") if x.strip()}

precisions, recalls, f1s, attempts = [], [], [], []

workflow_ok = validation_ok = trace_ok = memory_ok = answer_ok = recovery_yes = pass3_ok = 0

for row in rows:
    expected = split_tools(row.get("expected_tools", ""))
    selected = split_tools(row.get("selected_tools", ""))

    status = row.get("workflow_status", "").strip()
    validation = row.get("validation_status", "").strip()
    recovery = row.get("recovery_used", "").strip().lower()
    memory = row.get("memory_updated", "").strip().lower()
    answer = row.get("final_answer_generated", "").strip().lower()
    trace_path = row.get("trace_path", "").strip()

    try:
        attempt_count = int(float(row.get("attempts", "0")))
    except Exception:
        attempt_count = 0

    attempts.append(attempt_count)

    workflow_ok += status == "validated"
    validation_ok += validation == "valid"
    trace_ok += bool(trace_path)
    memory_ok += memory == "yes"
    answer_ok += answer == "yes"
    recovery_yes += recovery == "yes"
    pass3_ok += status == "validated" and attempt_count <= 3

    precision = len(expected & selected) / len(selected) if selected else 0
    recall = len(expected & selected) / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

n = len(rows) or 1

print("CURRENT PARTIAL METRICS")
print("=" * 60)
print(f"Executed active tests: {len(rows)}")
print(f"Workflow Completion Rate: {workflow_ok / n * 100:.2f}%")
print(f"Output Validation Success Rate: {validation_ok / n * 100:.2f}%")
print(f"Tool Selection Precision: {sum(precisions) / n * 100:.2f}%")
print(f"Tool Selection Recall: {sum(recalls) / n * 100:.2f}%")
print(f"Tool Selection F1-score: {sum(f1s) / n * 100:.2f}%")
print(f"Trace Save Rate: {trace_ok / n * 100:.2f}%")
print(f"Memory Update Success Rate: {memory_ok / n * 100:.2f}%")
print(f"Final Answer Generation Rate: {answer_ok / n * 100:.2f}%")
print(f"Recovery Used Rate: {recovery_yes / n * 100:.2f}%")
print(f"Average Number of Attempts: {sum(attempts) / n:.2f}")
print(f"pass@3: {pass3_ok / n * 100:.2f}%")

print("\nPER-TEST TOOL SELECTION")
print("=" * 60)
for row, p, r, f in zip(rows, precisions, recalls, f1s):
    print(
        f"{row.get('test_id')}: "
        f"status={row.get('workflow_status')}, "
        f"validation={row.get('validation_status')}, "
        f"precision={p:.2f}, recall={r:.2f}, f1={f:.2f}"
    )
