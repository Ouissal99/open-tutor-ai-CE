import csv
import json
from pathlib import Path

tracking_path = Path("evaluation/metrics/final_evaluation_tracking_30_tests.csv")

controlled_recovery_ids = {"T26", "T27", "T28", "T29", "T30"}

required_trace_markers = [
    "selected_tools",
    "tool_results",
    "validation",
    "output",
    "trace",
]

def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

def contains_json_text(obj, text):
    if obj is None:
        return False
    return text.lower() in json.dumps(obj, ensure_ascii=False).lower()

with tracking_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

n = len(rows)

tool_execution_success = 0
trace_complete = 0
recovery_used_cases = 0
recovery_success = 0

controlled_invalid_detected = 0
controlled_recovered = 0
controlled_missing_tool_corrected = 0

per_test_rows = []

for row in rows:
    tid = row["test_id"]
    trace_path = row["trace_path"]
    log_path = Path(f"evaluation/runs/{tid}_full_manager.txt")

    trace = load_json(trace_path)
    log_text = read_text(log_path)
    combined_text = (log_text + "\n" + json.dumps(trace, ensure_ascii=False) if trace else log_text).lower()

    workflow_valid = row.get("workflow_status") == "validated" and row.get("validation_status") == "valid"

    if workflow_valid and row.get("selected_tools", "").strip():
        tool_execution_success += 1

    if trace is not None:
        present_markers = sum(1 for marker in required_trace_markers if contains_json_text(trace, marker))
        if present_markers >= 4:
            trace_complete += 1

    if row.get("recovery_used", "").strip().lower() == "yes":
        recovery_used_cases += 1
        if workflow_valid and int(float(row.get("attempts", "0"))) <= 3:
            recovery_success += 1

    is_controlled = tid in controlled_recovery_ids

    invalid_detected = (
        "status: invalid" in combined_text
        or '"status": "invalid"' in combined_text
        or "insufficient_grounding" in combined_text
        or "missing_visual_support" in combined_text
    )

    failure_recovery_triggered = "failurerecovery triggered" in combined_text or "recover_failure" in combined_text

    missing_tool_corrected = (
        "required_tools" in combined_text
        or "add_grounding_tools" in combined_text
        or "add_visual_support" in combined_text
        or "using_tools_required_by_failure_recovery" in combined_text
    )

    if is_controlled:
        if invalid_detected:
            controlled_invalid_detected += 1
        if workflow_valid and failure_recovery_triggered:
            controlled_recovered += 1
        if missing_tool_corrected:
            controlled_missing_tool_corrected += 1

    per_test_rows.append({
        "test_id": tid,
        "category": row.get("category", ""),
        "workflow_valid": "Yes" if workflow_valid else "No",
        "recovery_used": row.get("recovery_used", ""),
        "controlled_recovery_test": "Yes" if is_controlled else "No",
        "invalid_detected": "Yes" if invalid_detected else "No",
        "failure_recovery_triggered": "Yes" if failure_recovery_triggered else "No",
        "missing_tool_corrected": "Yes" if missing_tool_corrected else "No",
    })

controlled_n = len(controlled_recovery_ids)

summary = f"""PROMISED CHAPTER 5 AUDIT METRICS — CORRECTED
============================================================
Number of active tests: {n}

Tool Execution Success Rate: {tool_execution_success / n * 100:.2f}%
Trace Completeness Rate: {trace_complete / n * 100:.2f}%

All recovery-used cases: {recovery_used_cases}
Recovery Success Rate over recovery-used cases: {recovery_success / recovery_used_cases * 100 if recovery_used_cases else 0:.2f}%

Controlled forced-recovery tests: {controlled_n}
Invalid Output Detection Rate over controlled recovery tests: {controlled_invalid_detected / controlled_n * 100:.2f}%
Controlled Recovery Success Rate: {controlled_recovered / controlled_n * 100:.2f}%
Missing Tool Correction Rate over controlled recovery tests: {controlled_missing_tool_corrected / controlled_n * 100:.2f}%
"""

out_csv = Path("evaluation/metrics/promised_per_test_audit_v2.csv")
out_txt = Path("evaluation/metrics/promised_global_audit_v2.txt")

with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_test_rows[0].keys()))
    writer.writeheader()
    writer.writerows(per_test_rows)

out_txt.write_text(summary, encoding="utf-8")

print(summary)
print("Saved:", out_csv)
print("Saved:", out_txt)
