import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path


TRACKING_FIELDS = [
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


def remove_tracking_row(tracking_file: Path, test_id: str) -> None:
    """Remove a previous failed attempt before retrying the same test."""
    if not tracking_file.exists():
        return

    with tracking_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or TRACKING_FIELDS
        rows = [
            row
            for row in reader
            if row.get("test_id") != test_id
        ]

    with tracking_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_failed_case_artifacts(test_id: str) -> None:
    """Remove artifacts from an unavailable-provider attempt."""
    paths = [
        Path(f"evaluation/runs/{test_id}_full_manager.txt"),
        Path(f"evaluation/runs/{test_id}_full_manager_report.json"),
        Path(f"evaluation/traces/{test_id}_trace.json"),
    ]

    for path in paths:
        if path.exists():
            path.unlink()


def append_infrastructure_failure(
    tracking_file: Path,
    row: dict,
    note: str,
) -> None:
    """Record infrastructure failure separately from manager failure."""
    tracking_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = tracking_file.exists()

    with tracking_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKING_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "test_id": row["test_id"],
                "category": row["category"],
                "question": row["question"],
                "expected_tools": row["expected_tools"],
                "selected_tools": "",
                "workflow_status": "infrastructure_invalid",
                "validation_status": "not_evaluated",
                "recovery_used": "No",
                "attempts": "0",
                "trace_path": "",
                "trace_paths": "",
                "report_path": (
                    f"evaluation/runs/"
                    f"{row['test_id']}_full_manager_report.json"
                ),
                "memory_updated": "No",
                "final_answer_generated": "No",
                "notes": note,
            }
        )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--provider-retries", type=int, default=3)
parser.add_argument("--retry-wait", type=int, default=20)
parser.add_argument("--case-delay", type=int, default=3)
args = parser.parse_args()

if args.provider_retries < 1:
    raise ValueError("--provider-retries must be at least 1")

dataset = Path(args.dataset)
output_dir = Path(args.output_dir)
tracking_file = Path("evaluation/metrics/evaluation_tracking.csv")

output_dir.mkdir(parents=True, exist_ok=True)
tracking_file.parent.mkdir(parents=True, exist_ok=True)

if tracking_file.exists():
    tracking_file.unlink()

with dataset.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

for index, row in enumerate(rows, start=1):
    test_id = row["test_id"]

    print("=" * 90)
    print(
        f"[{index}/{len(rows)}] Full Manager "
        f"{test_id}: {row['question']}"
    )
    print("=" * 90)

    command = [
        sys.executable,
        "evaluation/run_case.py",
        "--test-id",
        test_id,
        "--learner-id",
        f"article_{test_id}",
        "--category",
        row["category"],
        "--question",
        row["question"],
        "--expected-tools",
        row["expected_tools"],
        "--recovery-test",
        row.get("recovery_test", "No") or "No",
    ]

    case_completed = False

    for provider_attempt in range(1, args.provider_retries + 1):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout, end="")

        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")

        combined_output = (
            (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )

        provider_available = (
            "Provider status: available" in combined_output
        )
        provider_unavailable = (
            "Provider status: unavailable" in combined_output
        )

        if result.returncode == 0 and provider_available:
            case_completed = True
            break

        if provider_unavailable:
            print(
                f"\n[INFRASTRUCTURE] Provider unavailable for "
                f"{test_id}. Attempt "
                f"{provider_attempt}/{args.provider_retries}."
            )
        else:
            print(
                f"\n[INFRASTRUCTURE] Case {test_id} did not produce "
                f"a valid provider status. Return code: "
                f"{result.returncode}."
            )

        remove_tracking_row(tracking_file, test_id)

        if provider_attempt < args.provider_retries:
            clean_failed_case_artifacts(test_id)

            wait_seconds = args.retry_wait * provider_attempt
            print(
                f"Retrying {test_id} after "
                f"{wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    if not case_completed:
        remove_tracking_row(tracking_file, test_id)

        append_infrastructure_failure(
            tracking_file=tracking_file,
            row=row,
            note=(
                "Provider unavailable or subprocess failed after "
                f"{args.provider_retries} infrastructure attempts. "
                "This case was not evaluated as a manager failure."
            ),
        )

        print(
            f"[INFRASTRUCTURE INVALID] {test_id} was not evaluated."
        )

    if index < len(rows) and args.case_delay > 0:
        time.sleep(args.case_delay)

destination = output_dir / "full_manager_tracking_100.csv"

if tracking_file.exists():
    shutil.copyfile(tracking_file, destination)
else:
    raise RuntimeError(
        f"Tracking file was not created: {tracking_file}"
    )

print("\nDONE.")
print("Saved:", destination)
