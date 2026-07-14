import csv
import json
from pathlib import Path
from collections import Counter

BASE = Path("evaluation/baselines/manager_without_recovery")
SUMMARY = BASE / "manager_without_recovery_execution_summary.csv"
OUT_CSV = BASE / "manager_without_recovery_detailed_metrics.csv"
OUT_TXT = BASE / "manager_without_recovery_metrics.txt"

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from walk(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x)

def all_text(obj):
    parts = []
    for k, v in walk(obj):
        parts.append(str(k))
        if isinstance(v, (str, int, float, bool)):
            parts.append(str(v))
    return " ".join(parts).lower()

def find_attempts(report, trace):
    vals = []
    for obj in [report, trace]:
        if obj is None:
            continue
        for k, v in walk(obj):
            if k in {"attempt", "attempt_number", "attempts"}:
                if isinstance(v, int):
                    vals.append(v)
                elif isinstance(v, str) and v.isdigit():
                    vals.append(int(v))
    return max(vals) if vals else 1

def find_recovery_used(report, trace):
    for obj in [report, trace]:
        if obj is None:
            continue
        for k, v in walk(obj):
            if k == "recovery_used" and v is True:
                return True
            if isinstance(v, str) and v in {"failure_recovery_applied", "recover_failure"}:
                return True
            if k in {"event_type", "node", "node_name"} and str(v) in {"failure_recovery_applied", "recover_failure"}:
                return True
    return False

def find_validated(report, trace, fallback_status):
    if fallback_status == "validated":
        return True

    for obj in [report, trace]:
        if obj is None:
            continue
        for k, v in walk(obj):
            if k in {"workflow_status", "validation_status", "final_status", "status"}:
                if str(v).lower() == "validated" or str(v).lower() == "valid":
                    return True
    return False

def find_invalid_detected(report, trace, fallback_status):
    if fallback_status == "failed_or_invalid":
        return True

    txt = ""
    for obj in [report, trace]:
        if obj is not None:
            txt += " " + all_text(obj)

    invalid_markers = [
        "invalid",
        "insufficient_grounding",
        "missing_required",
        "validation_failed",
        "failed validation",
        "maximum_attempts_reached",
    ]
    return any(m in txt for m in invalid_markers)

def find_trace_complete(trace_path):
    p = Path(trace_path) if trace_path else None
    if not p or not p.exists():
        return False
    obj = load_json(p)
    if obj is None:
        return False
    txt = all_text(obj)
    required_markers = [
        "selected_tools",
        "tool_plan",
        "validation",
        "output",
    ]
    return all(m in txt for m in required_markers)

def find_final_answer(report, log_path):
    if report is not None:
        for k, v in walk(report):
            if k in {"final_answer", "answer", "tutoring_answer"} and isinstance(v, str) and len(v.strip()) > 50:
                return True

    if log_path and Path(log_path).exists():
        txt = Path(log_path).read_text(encoding="utf-8", errors="ignore")
        markers = ["TutoringAnswerWriter generated", "Final answer", "final_answer"]
        return any(m in txt for m in markers)

    return False

with SUMMARY.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

out_rows = []

for r in rows:
    report = load_json(r["report_json"]) if r.get("report_json") else None
    trace = load_json(r["trace_json"]) if r.get("trace_json") else None

    attempts = find_attempts(report, trace)
    recovery_used = find_recovery_used(report, trace)
    validated = find_validated(report, trace, r.get("workflow_status", ""))
    invalid_detected = find_invalid_detected(report, trace, r.get("workflow_status", ""))
    trace_complete = find_trace_complete(r.get("trace_json", ""))
    final_answer_generated = find_final_answer(report, r.get("log_path", ""))

    out_rows.append({
        "test_id": r["test_id"],
        "category": r["category"],
        "recovery_test": r["recovery_test"],
        "process_ok": "Yes" if r["return_code"] == "0" else "No",
        "validated": "Yes" if validated else "No",
        "invalid_detected": "Yes" if invalid_detected else "No",
        "recovery_used": "Yes" if recovery_used else "No",
        "attempts": attempts,
        "trace_saved": "Yes" if Path(r["trace_json"]).exists() else "No",
        "trace_complete": "Yes" if trace_complete else "No",
        "final_answer_generated": "Yes" if final_answer_generated else "No",
        "question": r["question"],
    })

def pct(n, d):
    return 0 if d == 0 else 100 * n / d

total = len(out_rows)
process_ok = sum(r["process_ok"] == "Yes" for r in out_rows)
validated = sum(r["validated"] == "Yes" for r in out_rows)
invalid_detected = sum(r["invalid_detected"] == "Yes" for r in out_rows)
recovery_used = sum(r["recovery_used"] == "Yes" for r in out_rows)
trace_saved = sum(r["trace_saved"] == "Yes" for r in out_rows)
trace_complete = sum(r["trace_complete"] == "Yes" for r in out_rows)
final_answer = sum(r["final_answer_generated"] == "Yes" for r in out_rows)

forced = [r for r in out_rows if r["recovery_test"] == "Yes"]
forced_total = len(forced)
forced_validated = sum(r["validated"] == "Yes" for r in forced)
forced_invalid = sum(r["invalid_detected"] == "Yes" for r in forced)
forced_recovery_used = sum(r["recovery_used"] == "Yes" for r in forced)

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    fieldnames = list(out_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

lines = []
lines.append("MANAGER WITHOUT RECOVERY BASELINE — CLEAN METRICS")
lines.append("=" * 70)
lines.append(f"Total tests: {total}")
lines.append("")
lines.append(f"Process Completion Rate: {pct(process_ok, total):.2f}% ({process_ok}/{total})")
lines.append(f"Validation Success Rate: {pct(validated, total):.2f}% ({validated}/{total})")
lines.append(f"Invalid Output Detection Count: {invalid_detected}/{total}")
lines.append(f"Recovery Used Rate: {pct(recovery_used, total):.2f}% ({recovery_used}/{total})")
lines.append(f"Trace Save Rate: {pct(trace_saved, total):.2f}% ({trace_saved}/{total})")
lines.append(f"Trace Completeness Rate: {pct(trace_complete, total):.2f}% ({trace_complete}/{total})")
lines.append(f"Final Answer Generation Rate: {pct(final_answer, total):.2f}% ({final_answer}/{total})")
lines.append("")
lines.append("Controlled recovery cases:")
lines.append(f"Controlled recovery tests: {forced_total}")
lines.append(f"Validated despite recovery disabled: {forced_validated}/{forced_total}")
lines.append(f"Invalid detected / remained invalid: {forced_invalid}/{forced_total}")
lines.append(f"FailureRecovery triggered: {forced_recovery_used}/{forced_total}")
lines.append("")
lines.append("Per forced-recovery case:")
for r in forced:
    lines.append(
        f'{r["test_id"]}: validated={r["validated"]}, invalid_detected={r["invalid_detected"]}, '
        f'recovery_used={r["recovery_used"]}, attempts={r["attempts"]}, final_answer={r["final_answer_generated"]}'
    )

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print()
print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
