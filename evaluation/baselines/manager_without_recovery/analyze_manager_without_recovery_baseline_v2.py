import csv
from pathlib import Path

BASE = Path("evaluation/baselines/manager_without_recovery")
SUMMARY = BASE / "manager_without_recovery_execution_summary.csv"
OUT_CSV = BASE / "manager_without_recovery_detailed_metrics_v2.csv"
OUT_TXT = BASE / "manager_without_recovery_metrics_v2.txt"

def pct(n, d):
    return 0 if d == 0 else 100 * n / d

with SUMMARY.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

out_rows = []

for r in rows:
    status = r.get("workflow_status", "")
    validated = status == "validated"
    invalid_or_failed = status == "failed_or_invalid"

    trace_saved = bool(r.get("trace_json")) and Path(r["trace_json"]).exists()
    report_saved = bool(r.get("report_json")) and Path(r["report_json"]).exists()

    out_rows.append({
        "test_id": r["test_id"],
        "category": r["category"],
        "recovery_test": r["recovery_test"],
        "process_ok": "Yes" if r["return_code"] == "0" else "No",
        "validated": "Yes" if validated else "No",
        "invalid_or_failed": "Yes" if invalid_or_failed else "No",
        "recovery_disabled": "Yes",
        "recovery_used": "No",
        "attempts": "1",
        "trace_saved": "Yes" if trace_saved else "No",
        "report_saved": "Yes" if report_saved else "No",
        "final_answer_generated": "Yes" if validated else "No",
        "question": r["question"],
    })

total = len(out_rows)
process_ok = sum(r["process_ok"] == "Yes" for r in out_rows)
validated = sum(r["validated"] == "Yes" for r in out_rows)
invalid_or_failed = sum(r["invalid_or_failed"] == "Yes" for r in out_rows)
trace_saved = sum(r["trace_saved"] == "Yes" for r in out_rows)
report_saved = sum(r["report_saved"] == "Yes" for r in out_rows)
final_answer = sum(r["final_answer_generated"] == "Yes" for r in out_rows)

forced = [r for r in out_rows if r["recovery_test"] == "Yes"]
forced_total = len(forced)
forced_validated = sum(r["validated"] == "Yes" for r in forced)
forced_failed = sum(r["invalid_or_failed"] == "Yes" for r in forced)

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    fieldnames = list(out_rows[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

lines = []
lines.append("MANAGER WITHOUT RECOVERY BASELINE — CLEAN METRICS V2")
lines.append("=" * 70)
lines.append(f"Total tests: {total}")
lines.append("")
lines.append(f"Process Completion Rate: {pct(process_ok, total):.2f}% ({process_ok}/{total})")
lines.append(f"Validation Success Rate: {pct(validated, total):.2f}% ({validated}/{total})")
lines.append(f"Invalid or Failed Output Rate: {pct(invalid_or_failed, total):.2f}% ({invalid_or_failed}/{total})")
lines.append(f"Recovery Used Rate: 0.00% (0/{total})")
lines.append(f"Average Attempts: 1.00")
lines.append(f"Trace Save Rate: {pct(trace_saved, total):.2f}% ({trace_saved}/{total})")
lines.append(f"Report Save Rate: {pct(report_saved, total):.2f}% ({report_saved}/{total})")
lines.append(f"Final Answer Generation Rate: {pct(final_answer, total):.2f}% ({final_answer}/{total})")
lines.append("")
lines.append("Controlled recovery cases:")
lines.append(f"Controlled recovery tests: {forced_total}")
lines.append(f"Validated despite recovery disabled: {forced_validated}/{forced_total}")
lines.append(f"Invalid/failed because recovery disabled: {forced_failed}/{forced_total}")
lines.append(f"Controlled Recovery Success Rate: 0.00% (0/{forced_total})")
lines.append("")
lines.append("Per forced-recovery case:")
for r in forced:
    lines.append(
        f'{r["test_id"]}: validated={r["validated"]}, invalid_or_failed={r["invalid_or_failed"]}, '
        f'recovery_used={r["recovery_used"]}, attempts={r["attempts"]}, final_answer={r["final_answer_generated"]}'
    )

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print()
print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
