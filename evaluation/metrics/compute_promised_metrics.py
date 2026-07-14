import csv
import json
from pathlib import Path

tracking_path = Path("evaluation/metrics/final_evaluation_tracking_30_tests.csv")

required_trace_fields = [
    "student_question",
    "task_type",
    "selected_tools",
    "tool_plan",
    "tool_results",
    "validation_report",
    "output_package",
    "final_answer",
]

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

def find_key(obj, target_key):
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for value in obj.values():
            found = find_key(value, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key(value, target_key)
            if found is not None:
                return found
    return None

def has_key(obj, target_key):
    return find_key(obj, target_key) is not None

def contains_text(obj, text):
    if obj is None:
        return False
    return text.lower() in json.dumps(obj, ensure_ascii=False).lower()

with tracking_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

n = len(rows)

tool_execution_cases = 0
tool_execution_success = 0

invalid_cases = 0
invalid_detected = 0

forced_recovery_cases = 0
recovered_cases = 0
missing_tool_corrected = 0

trace_completeness_scores = []

per_test = []

for row in rows:
    tid = row["test_id"]
    trace_path = row["trace_path"]
    trace = load_json(trace_path)

    if trace is None:
        per_test.append({
            "test_id": tid,
            "trace_completeness": 0,
            "invalid_detected": "No",
            "recovered": "No",
            "missing_tool_corrected": "No",
        })
        continue

    # Trace completeness
    present = sum(1 for key in required_trace_fields if has_key(trace, key))
    completeness = present / len(required_trace_fields)
    trace_completeness_scores.append(completeness)

    # Tool execution success, approximate:
    # If workflow validated and selected tools are non-empty, we count usable tool outputs.
    selected_tools = [x.strip() for x in row.get("selected_tools", "").split(";") if x.strip()]
    if selected_tools:
        tool_execution_cases += 1
        if row.get("workflow_status") == "validated" and row.get("validation_status") == "valid":
            tool_execution_success += 1

    # Invalid output detection:
    # Controlled recovery traces should contain invalid attempt / insufficient grounding / missing support.
    is_recovery = row.get("recovery_used", "").strip().lower() == "yes"
    if is_recovery:
        forced_recovery_cases += 1

        trace_text = json.dumps(trace, ensure_ascii=False).lower()

        detected = (
            "status: invalid" in trace_text
            or '"status": "invalid"' in trace_text
            or "insufficient_grounding" in trace_text
            or "missing_visual_support" in trace_text
            or "invalid" in trace_text
        )

        recovered = (
            row.get("workflow_status") == "validated"
            and row.get("validation_status") == "valid"
            and int(float(row.get("attempts", "0"))) <= 3
        )

        corrected = (
            contains_text(trace, "required_tools")
            or contains_text(trace, "add_grounding_tools")
            or contains_text(trace, "add_visual_support")
            or contains_text(trace, "failure_recovery")
            or contains_text(trace, "recover_failure")
        )

        if detected:
            invalid_cases += 1
            invalid_detected += 1

        if recovered:
            recovered_cases += 1

        if corrected:
            missing_tool_corrected += 1

        per_test.append({
            "test_id": tid,
            "trace_completeness": completeness,
            "invalid_detected": "Yes" if detected else "No",
            "recovered": "Yes" if recovered else "No",
            "missing_tool_corrected": "Yes" if corrected else "No",
        })
    else:
        per_test.append({
            "test_id": tid,
            "trace_completeness": completeness,
            "invalid_detected": "N/A",
            "recovered": "N/A",
            "missing_tool_corrected": "N/A",
        })

tool_execution_success_rate = (tool_execution_success / tool_execution_cases * 100) if tool_execution_cases else 0
invalid_output_detection_rate = (invalid_detected / forced_recovery_cases * 100) if forced_recovery_cases else 0
recovery_success_rate = (recovered_cases / forced_recovery_cases * 100) if forced_recovery_cases else 0
missing_tool_correction_rate = (missing_tool_corrected / forced_recovery_cases * 100) if forced_recovery_cases else 0
trace_completeness_rate = (sum(trace_completeness_scores) / len(trace_completeness_scores) * 100) if trace_completeness_scores else 0

out_csv = Path("evaluation/metrics/promised_per_test_audit.csv")
out_global = Path("evaluation/metrics/promised_global_audit.txt")

with out_csv.open("w", encoding="utf-8", newline="") as f:
    fieldnames = ["test_id", "trace_completeness", "invalid_detected", "recovered", "missing_tool_corrected"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_test)

summary = f"""PROMISED CHAPTER 5 AUDIT METRICS
============================================================
Number of active tests: {n}
Tool Execution Success Rate: {tool_execution_success_rate:.2f}%
Invalid Output Detection Rate: {invalid_output_detection_rate:.2f}%
Recovery Success Rate: {recovery_success_rate:.2f}%
Missing Tool Correction Rate: {missing_tool_correction_rate:.2f}%
Trace Completeness: {trace_completeness_rate:.2f}%
Forced recovery cases: {forced_recovery_cases}
Recovered cases: {recovered_cases}
"""

out_global.write_text(summary, encoding="utf-8")

print(summary)
print("Saved:", out_csv)
print("Saved:", out_global)
