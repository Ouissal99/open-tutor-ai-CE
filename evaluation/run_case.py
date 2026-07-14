import argparse
import ast
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

RUNS_DIR = Path("evaluation/runs")
TRACES_DIR = Path("evaluation/traces")
METRICS_DIR = Path("evaluation/metrics")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRACKING_FILE = METRICS_DIR / "evaluation_tracking.csv"

HEADER = [
    "test_id",
    "category",
    "question",
    "expected_tools",
    "selected_tools",
    "workflow_status",
    "validation_status",
    "recovery_used",
    "attempts",
    "trace_path",
    "report_path",
    "memory_updated",
    "final_answer_generated",
    "notes",
]

def newest_file_after(folder, pattern, start_time):
    files = []
    for p in Path(folder).glob(pattern):
        if p.stat().st_mtime >= start_time - 2:
            files.append(p)
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def read_report(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def extract_line_value(stdout, label):
    pattern = rf"{re.escape(label)}:\s*(.*)"
    matches = re.findall(pattern, stdout)
    if not matches:
        return None
    value = matches[-1].strip()
    if value in {"None", "null", ""}:
        return None
    return value

def extract_selected_tools(stdout, report):
    value = extract_line_value(stdout, "Selected tools")
    if value:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return ";".join(str(x) for x in parsed)
        except Exception:
            return value.replace("[", "").replace("]", "").replace("'", "").replace(", ", ";")

    summary = report.get("summary", {})
    selected = summary.get("selected_tools") or report.get("selected_tools") or []
    if isinstance(selected, list):
        return ";".join(selected)
    return str(selected)

def extract_attempts(stdout):
    attempts = [int(x) for x in re.findall(r"attempt:\s*(\d+)", stdout)]
    if attempts:
        return max(attempts)
    return 1

def append_tracking(row):
    exists = TRACKING_FILE.exists()
    with TRACKING_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--expected-tools", required=True)
    parser.add_argument("--recovery-test", choices=["Yes", "No"], default="No")
    args = parser.parse_args()

    log_path = RUNS_DIR / f"{args.test_id}_full_manager.txt"
    clean_report = RUNS_DIR / f"{args.test_id}_full_manager_report.json"
    clean_trace_path = TRACES_DIR / f"{args.test_id}_trace.json"

    if clean_trace_path.exists():
        clean_trace_path.unlink()

    cmd = [
        "python",
        "scripts/run_agentic_tutoring_demo.py",
        "--question",
        args.question,
    ]

    if args.recovery_test == "Yes":
        cmd.append("--force-recovery-test")

    print("Running:", " ".join(cmd))

    start_time = time.time()

    process = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    stdout = process.stdout
    log_path.write_text(stdout, encoding="utf-8")
    print(stdout)

    latest_report = newest_file_after("var/agentic_demo_runs", "*.json", start_time)

    if latest_report:
        shutil.copyfile(latest_report, clean_report)

    report = read_report(clean_report)

    workflow_status = extract_line_value(stdout, "Workflow status") or "unknown"
    trace_source = extract_line_value(stdout, "Trace path")

    saved_trace_value = ""
    if trace_source and Path(trace_source).exists():
        shutil.copyfile(trace_source, clean_trace_path)
        saved_trace_value = str(clean_trace_path)

    selected_tools = extract_selected_tools(stdout, report)

    recovery_used = "Yes" if "FailureRecovery triggered" in stdout else "No"
    attempts = extract_attempts(stdout)

    memory_updated = "Yes" if "DPM MemoryUpdateAgent updated learner memory" in stdout else "No"
    final_answer_generated = "Yes" if "TutoringAnswerWriter generated" in stdout else "No"

    validation_status = "valid" if workflow_status == "validated" else "invalid"

    notes = "Full manager execution."
    if args.recovery_test == "Yes":
        notes = "Controlled recovery test using --force-recovery-test."
    if "grounded fallback" in stdout.lower():
        notes += " Grounded fallback was used."
    if "Provider status: unavailable" in stdout:
        notes += " Provider unavailable."

    row = {
        "test_id": args.test_id,
        "category": args.category,
        "question": args.question,
        "expected_tools": args.expected_tools,
        "selected_tools": selected_tools,
        "workflow_status": workflow_status,
        "validation_status": validation_status,
        "recovery_used": recovery_used,
        "attempts": attempts,
        "trace_path": saved_trace_value,
        "report_path": str(clean_report) if clean_report.exists() else "",
        "memory_updated": memory_updated,
        "final_answer_generated": final_answer_generated,
        "notes": notes,
    }

    append_tracking(row)

    print("\nSaved log:", log_path)
    print("Saved report:", clean_report if clean_report.exists() else "")
    print("Saved trace:", saved_trace_value)
    print("Updated tracking:", TRACKING_FILE)

if __name__ == "__main__":
    main()
