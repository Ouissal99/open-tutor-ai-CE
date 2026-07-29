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
    "trace_paths",
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
    """Aggregate selected tools across every manager round."""
    ordered_tools = []

    def add_tools(value):
        if not value:
            return

        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = [
                item.strip().strip("'").strip('"')
                for item in value.strip("[]").split(",")
                if item.strip()
            ]

        if not isinstance(parsed, list):
            return

        for tool_name in parsed:
            tool_name = str(tool_name).strip()

            if tool_name and tool_name not in ordered_tools:
                ordered_tools.append(tool_name)

    # Capture every manager selection, not only the final summary.
    for value in re.findall(
        r"selected_tools:\s*(\[[^\n]*\])",
        stdout,
    ):
        add_tools(value)

    # Also include the final demo summary when present.
    add_tools(
        extract_line_value(
            stdout,
            "Selected tools",
        )
    )

    if not ordered_tools:
        summary = report.get("summary", {})
        selected = (
            summary.get("selected_tools")
            or report.get("selected_tools")
            or []
        )

        if isinstance(selected, list):
            for tool_name in selected:
                tool_name = str(tool_name).strip()

                if tool_name and tool_name not in ordered_tools:
                    ordered_tools.append(tool_name)

    return ";".join(ordered_tools)


def extract_trace_ids(stdout):
    """Return every manager trace ID in execution order."""
    trace_ids = []

    patterns = (
        r"trace_id:\s*(TT-[A-Za-z0-9-]+)",
        r"Trace ID:\s*(TT-[A-Za-z0-9-]+)",
    )

    for pattern in patterns:
        for trace_id in re.findall(pattern, stdout):
            if trace_id not in trace_ids:
                trace_ids.append(trace_id)

    return trace_ids


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
    parser.add_argument(
        "--learner-id",
        required=True,
        help=(
            "Isolated learner identifier used for DPM "
            "and learner-specific trace retrieval."
        ),
    )
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

    for old_round_trace in TRACES_DIR.glob(
        f"{args.test_id}_round*_trace.json"
    ):
        old_round_trace.unlink()

    cmd = [
        "python",
        "scripts/run_agentic_tutoring_demo.py",
        "--question",
        args.question,
        "--learner-id",
        args.learner_id,
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

    workflow_status = (
        extract_line_value(stdout, "Workflow status")
        or "unknown"
    )
    final_trace_source = extract_line_value(
        stdout,
        "Trace path",
    )

    trace_records = []
    copied_source_paths = set()

    for round_index, trace_id in enumerate(
        extract_trace_ids(stdout),
        start=1,
    ):
        source = Path(
            f"var/agentic_traces/{trace_id}.json"
        )

        if not source.exists():
            continue

        destination = TRACES_DIR / (
            f"{args.test_id}_round"
            f"{round_index:02d}_{trace_id}_trace.json"
        )

        shutil.copyfile(source, destination)

        trace_records.append(
            {
                "trace_id": trace_id,
                "source": source,
                "saved": destination,
                "data": read_report(source),
            }
        )
        copied_source_paths.add(
            str(source.resolve())
        )

    # Fallback in case the final trace was not printed as trace_id.
    if final_trace_source:
        final_source = Path(final_trace_source)

        if (
            final_source.exists()
            and str(final_source.resolve())
            not in copied_source_paths
        ):
            round_index = len(trace_records) + 1

            destination = TRACES_DIR / (
                f"{args.test_id}_round"
                f"{round_index:02d}_final_trace.json"
            )

            shutil.copyfile(
                final_source,
                destination,
            )

            trace_records.append(
                {
                    "trace_id": final_source.stem,
                    "source": final_source,
                    "saved": destination,
                    "data": read_report(final_source),
                }
            )

    # Preserve one primary trace for backward compatibility.
    # For recovery cases, use the trace that contains recovery evidence.
    primary_record = next(
        (
            record
            for record in trace_records
            if record["data"].get("recovery_used") is True
        ),
        trace_records[-1] if trace_records else None,
    )

    saved_trace_value = ""
    trace_paths_value = ""

    if primary_record:
        shutil.copyfile(
            primary_record["source"],
            clean_trace_path,
        )
        saved_trace_value = str(clean_trace_path)

        trace_paths_value = ";".join(
            str(record["saved"])
            for record in trace_records
        )

    selected_tools = extract_selected_tools(
        stdout,
        report,
    )

    trace_attempts = []

    for record in trace_records:
        value = record["data"].get("attempts", 1)

        try:
            trace_attempts.append(int(value))
        except (TypeError, ValueError):
            pass

    recovery_from_traces = any(
        record["data"].get("recovery_used") is True
        for record in trace_records
    )

    recovery_used = (
        "Yes"
        if (
            "FailureRecovery triggered" in stdout
            or recovery_from_traces
        )
        else "No"
    )

    attempts = max(
        [
            extract_attempts(stdout),
            *trace_attempts,
        ]
    )

    memory_updated = "Yes" if "DPM MemoryUpdateAgent updated learner memory" in stdout else "No"
    final_answer_generated = "Yes" if (
    "Final tutor answer generated" in stdout
    or "TutoringAnswerWriter generated" in stdout
    ) else "No"

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
        "trace_paths": trace_paths_value,
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
