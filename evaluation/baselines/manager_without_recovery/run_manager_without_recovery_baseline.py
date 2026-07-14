import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TEST_CASES = Path("evaluation/test_cases/test_cases.csv")

BASE = Path("evaluation/baselines/manager_without_recovery")
RUNS_OUT = BASE / "runs"
TRACES_OUT = BASE / "traces"
LOGS_OUT = BASE / "logs"
SUMMARY_CSV = BASE / "manager_without_recovery_execution_summary.csv"

for d in [RUNS_OUT, TRACES_OUT, LOGS_OUT]:
    d.mkdir(parents=True, exist_ok=True)

def read_cases():
    with TEST_CASES.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def copy_if_exists(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)
    return ""

def main():
    cases = read_cases()
    rows = []

    env = os.environ.copy()
    env["NO_RECOVERY_BASELINE"] = "1"
    env["LLM_MAX_ATTEMPTS"] = env.get("LLM_MAX_ATTEMPTS", "5")
    env["LLM_RETRY_DELAY_SECONDS"] = env.get("LLM_RETRY_DELAY_SECONDS", "6")

    for i, case in enumerate(cases, start=1):
        tid = case["test_id"]
        category = case["category"]
        question = case["question"]
        expected_tools = case["expected_tools"]
        recovery_test = case.get("recovery_test", "No")

        print(f"\n[{i}/{len(cases)}] No-recovery baseline {tid}: {question}")

        cmd = [
            sys.executable,
            "evaluation/run_case.py",
            "--test-id", tid,
            "--category", category,
            "--question", question,
            "--expected-tools", expected_tools,
            "--recovery-test", recovery_test,
        ]

        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        log_path = LOGS_OUT / f"{tid}_no_recovery.log"
        log_path.write_text(proc.stdout, encoding="utf-8")

        # Copy generated artifacts immediately so later runs do not destroy this baseline.
        copied_report_json = copy_if_exists(
            Path(f"evaluation/runs/{tid}_full_manager_report.json"),
            RUNS_OUT / f"{tid}_no_recovery_report.json",
        )
        copied_report_txt = copy_if_exists(
            Path(f"evaluation/runs/{tid}_full_manager.txt"),
            RUNS_OUT / f"{tid}_no_recovery_full_manager.txt",
        )
        copied_trace = copy_if_exists(
            Path(f"evaluation/traces/{tid}_trace.json"),
            TRACES_OUT / f"{tid}_no_recovery_trace.json",
        )

        workflow_status = "unknown"
        validation_status = "unknown"
        recovery_used = "unknown"
        attempts = "unknown"
        final_answer_generated = "unknown"

        if copied_report_json:
            try:
                report = json.loads(Path(copied_report_json).read_text(encoding="utf-8"))
                workflow_status = str(report.get("workflow_status", report.get("status", "unknown")))
                validation_status = str(report.get("validation_status", report.get("final_status", "unknown")))
                recovery_used = str(report.get("recovery_used", "unknown"))
                attempts = str(report.get("attempts", "unknown"))
                final_answer_generated = "Yes" if report.get("final_answer") else "No"
            except Exception as exc:
                workflow_status = f"report_parse_error: {exc}"

        # Fallback: inspect log text
        text = proc.stdout.lower()
        if workflow_status == "unknown":
            if "workflow status: validated" in text or "workflow_status: validated" in text or "final_status: validated" in text:
                workflow_status = "validated"
            elif "invalid" in text or "failed" in text or proc.returncode != 0:
                workflow_status = "failed_or_invalid"

        rows.append({
            "test_id": tid,
            "category": category,
            "question": question,
            "expected_tools": expected_tools,
            "recovery_test": recovery_test,
            "return_code": proc.returncode,
            "workflow_status": workflow_status,
            "validation_status": validation_status,
            "recovery_used": recovery_used,
            "attempts": attempts,
            "final_answer_generated": final_answer_generated,
            "report_json": copied_report_json,
            "trace_json": copied_trace,
            "log_path": str(log_path),
        })

        print(f"    return_code={proc.returncode}")
        print(f"    workflow_status={workflow_status}")
        print(f"    attempts={attempts}")
        print(f"    recovery_used={recovery_used}")
        print(f"    report={copied_report_json or 'missing'}")
        print(f"    trace={copied_trace or 'missing'}")

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "test_id",
            "category",
            "question",
            "expected_tools",
            "recovery_test",
            "return_code",
            "workflow_status",
            "validation_status",
            "recovery_used",
            "attempts",
            "final_answer_generated",
            "report_json",
            "trace_json",
            "log_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if r["return_code"] == 0]
    err = [r for r in rows if r["return_code"] != 0]

    print("\nMANAGER WITHOUT RECOVERY BASELINE")
    print("=" * 60)
    print("Total:", len(rows))
    print("Process OK:", len(ok))
    print("Process errors:", len(err))
    print("Saved:", SUMMARY_CSV)

    if err:
        print("\nErrors:")
        for r in err:
            print(r["test_id"], "=>", r["log_path"])

if __name__ == "__main__":
    main()
