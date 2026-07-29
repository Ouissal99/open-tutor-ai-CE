import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


FIELDS = [
    "test_id",
    "learner_id",
    "category",
    "question",
    "expected_tools",
    "recovery_test",
    "status",
    "provider_status",
    "return_code",
    "selected_tools",
    "workflow_status",
    "validation_status",
    "recovery_used",
    "attempts",
    "final_answer_generated",
    "memory_updated",
    "report_json",
    "trace_json",
    "trace_paths",
    "log_path",
    "notes",
]


def read_csv(path: Path):
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_results(
    csv_path: Path,
    jsonl_path: Path,
    dataset_rows,
    results_by_id,
):
    ordered = [
        results_by_id[row["test_id"]]
        for row in dataset_rows
        if row["test_id"] in results_by_id
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )
        writer.writeheader()
        writer.writerows(ordered)

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in ordered:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def remove_working_artifacts(test_id: str):
    paths = [
        Path(
            f"evaluation/runs/"
            f"{test_id}_full_manager_report.json"
        ),
        Path(
            f"evaluation/runs/"
            f"{test_id}_full_manager.txt"
        ),
        Path(
            f"evaluation/traces/"
            f"{test_id}_trace.json"
        ),
    ]

    for path in paths:
        if path.exists():
            path.unlink()

    for path in Path(
        "evaluation/traces"
    ).glob(f"{test_id}_round*_trace.json"):
        path.unlink()


def latest_tracking_row(
    tracking_file: Path,
    test_id: str,
):
    matches = [
        row
        for row in read_csv(tracking_file)
        if row.get("test_id") == test_id
    ]

    return matches[-1] if matches else {}


def copy_artifacts(
    test_id: str,
    runs_dir: Path,
    traces_dir: Path,
):
    report_src = Path(
        f"evaluation/runs/"
        f"{test_id}_full_manager_report.json"
    )
    report_dst = (
        runs_dir
        / f"{test_id}_no_recovery_report.json"
    )

    report_value = ""

    if report_src.exists():
        shutil.copy2(
            report_src,
            report_dst,
        )
        report_value = str(report_dst)

    text_src = Path(
        f"evaluation/runs/"
        f"{test_id}_full_manager.txt"
    )
    text_dst = (
        runs_dir
        / f"{test_id}_no_recovery.txt"
    )

    if text_src.exists():
        shutil.copy2(
            text_src,
            text_dst,
        )

    trace_sources = []

    main_trace = Path(
        f"evaluation/traces/"
        f"{test_id}_trace.json"
    )

    if main_trace.exists():
        trace_sources.append(main_trace)

    trace_sources.extend(
        sorted(
            Path("evaluation/traces").glob(
                f"{test_id}_round*_trace.json"
            )
        )
    )

    copied_traces = []

    for source in trace_sources:
        destination_name = source.name.replace(
            test_id,
            f"{test_id}_no_recovery",
            1,
        )
        destination = (
            traces_dir
            / destination_name
        )

        shutil.copy2(
            source,
            destination,
        )
        copied_traces.append(
            str(destination)
        )

    trace_json = (
        copied_traces[0]
        if copied_traces
        else ""
    )

    return (
        report_value,
        trace_json,
        ";".join(copied_traces),
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    required=True,
)
parser.add_argument(
    "--output-dir",
    required=True,
)
parser.add_argument(
    "--provider-retries",
    type=int,
    default=3,
)
parser.add_argument(
    "--retry-wait",
    type=int,
    default=120,
)
parser.add_argument(
    "--case-delay",
    type=int,
    default=120,
)
parser.add_argument(
    "--only-ids",
    default="",
)

args = parser.parse_args()

if args.provider_retries < 1:
    raise ValueError(
        "--provider-retries must be at least 1"
    )

dataset_path = Path(args.dataset)
output_dir = Path(args.output_dir)

runs_dir = output_dir / "runs"
traces_dir = output_dir / "traces"
logs_dir = output_dir / "logs"

for directory in [
    output_dir,
    runs_dir,
    traces_dir,
    logs_dir,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

summary_csv = (
    output_dir
    / "manager_without_recovery_results_100.csv"
)
summary_jsonl = (
    output_dir
    / "manager_without_recovery_results_100.jsonl"
)

tracking_file = Path(
    "evaluation/metrics/"
    "evaluation_tracking.csv"
)

# The official summary is authoritative.
# This shared file is only temporary working output.
if tracking_file.exists():
    tracking_file.unlink()

dataset_rows = read_csv(dataset_path)

requested_ids = {
    item.strip()
    for item in args.only_ids.split(",")
    if item.strip()
}

selected_rows = [
    row
    for row in dataset_rows
    if (
        not requested_ids
        or row["test_id"] in requested_ids
    )
]

existing_rows = read_csv(summary_csv)

results_by_id = {
    row["test_id"]: row
    for row in existing_rows
}

completed_ids = {
    test_id
    for test_id, row in results_by_id.items()
    if row.get("provider_status") == "available"
}

pending_rows = [
    row
    for row in selected_rows
    if row["test_id"] not in completed_ids
]

print(
    "Existing provider-valid:",
    len(completed_ids),
)
print(
    "To run:",
    [
        row["test_id"]
        for row in pending_rows
    ],
)

environment = os.environ.copy()

# This is the actual architecture ablation.
environment["NO_RECOVERY_BASELINE"] = "1"

for index, case in enumerate(
    pending_rows,
    start=1,
):
    test_id = case["test_id"]
    base_learner_id = (
        f"article_no_recovery_{test_id}"
    )
    recovery_test = (
        case.get("recovery_test", "No")
        or "No"
    )

    print()
    print("=" * 90)
    print(
        f"[{index}/{len(pending_rows)}] "
        f"Manager Without Recovery "
        f"{test_id}"
    )
    print("=" * 90)

    final_row = None

    for provider_attempt in range(
        1,
        args.provider_retries + 1,
    ):
        # A fresh learner prevents partial DPM state from an
        # infrastructure-failed run contaminating its retry.
        learner_id = (
            base_learner_id
            if provider_attempt == 1
            else (
                f"{base_learner_id}"
                f"_infra_retry_{provider_attempt}"
            )
        )

        remove_working_artifacts(
            test_id
        )

        command = [
            sys.executable,
            "evaluation/run_case.py",
            "--test-id",
            test_id,
            "--learner-id",
            learner_id,
            "--category",
            case["category"],
            "--question",
            case["question"],
            "--expected-tools",
            case["expected_tools"],
            "--recovery-test",
            recovery_test,
        ]

        result = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        output = result.stdout or ""

        print(output, end="")

        log_path = (
            logs_dir
            / (
                f"{test_id}_attempt_"
                f"{provider_attempt}.log"
            )
        )

        log_path.write_text(
            output,
            encoding="utf-8",
        )

        provider_available = (
            "Provider status: available"
            in output
        )
        provider_unavailable = (
            "Provider status: unavailable"
            in output
            or (
                "LLM provider is currently "
                "unavailable"
            )
            in output
        )

        if provider_available:
            tracking = latest_tracking_row(
                tracking_file,
                test_id,
            )

            (
                report_json,
                trace_json,
                trace_paths,
            ) = copy_artifacts(
                test_id,
                runs_dir,
                traces_dir,
            )

            status = (
                "ok"
                if result.returncode == 0
                else "system_error"
            )

            final_row = {
                "test_id": test_id,
                "learner_id": learner_id,
                "category": case["category"],
                "question": case["question"],
                "expected_tools": (
                    case["expected_tools"]
                ),
                "recovery_test": recovery_test,
                "status": status,
                "provider_status": "available",
                "return_code": (
                    result.returncode
                ),
                "selected_tools": tracking.get(
                    "selected_tools",
                    "",
                ),
                "workflow_status": tracking.get(
                    "workflow_status",
                    "unknown",
                ),
                "validation_status": tracking.get(
                    "validation_status",
                    "unknown",
                ),
                "recovery_used": tracking.get(
                    "recovery_used",
                    "unknown",
                ),
                "attempts": tracking.get(
                    "attempts",
                    "unknown",
                ),
                "final_answer_generated": (
                    tracking.get(
                        "final_answer_generated",
                        "unknown",
                    )
                ),
                "memory_updated": tracking.get(
                    "memory_updated",
                    "unknown",
                ),
                "report_json": report_json,
                "trace_json": trace_json,
                "trace_paths": trace_paths,
                "log_path": str(log_path),
                "notes": tracking.get(
                    "notes",
                    "",
                ),
            }

            break

        print(
            f"\n[INFRASTRUCTURE] {test_id} "
            f"attempt "
            f"{provider_attempt}/"
            f"{args.provider_retries} "
            f"did not produce a valid "
            f"provider response."
        )

        if (
            provider_attempt
            < args.provider_retries
        ):
            time.sleep(
                args.retry_wait
            )

    if final_row is None:
        final_row = {
            "test_id": test_id,
            "learner_id": learner_id,
            "category": case["category"],
            "question": case["question"],
            "expected_tools": (
                case["expected_tools"]
            ),
            "recovery_test": recovery_test,
            "status": "error",
            "provider_status": (
                "unavailable"
                if provider_unavailable
                else "unknown"
            ),
            "return_code": (
                result.returncode
            ),
            "selected_tools": "",
            "workflow_status": (
                "infrastructure_invalid"
            ),
            "validation_status": (
                "not_evaluated"
            ),
            "recovery_used": "No",
            "attempts": "0",
            "final_answer_generated": "No",
            "memory_updated": "No",
            "report_json": "",
            "trace_json": "",
            "trace_paths": "",
            "log_path": str(log_path),
            "notes": (
                "Provider unavailable or "
                "provider status missing."
            ),
        }

    results_by_id[test_id] = final_row

    write_results(
        summary_csv,
        summary_jsonl,
        dataset_rows,
        results_by_id,
    )

    if index < len(pending_rows):
        time.sleep(
            args.case_delay
        )


final_rows = read_csv(summary_csv)

provider_valid = [
    row
    for row in final_rows
    if row.get("provider_status")
    == "available"
]

infrastructure_invalid = [
    row
    for row in final_rows
    if row.get("provider_status")
    != "available"
]

print()
print(
    "MANAGER WITHOUT RECOVERY "
    "OFFICIAL RUN"
)
print("=" * 70)
print(
    "Rows:",
    len(final_rows),
)
print(
    "Provider-valid:",
    len(provider_valid),
)
print(
    "Infrastructure-invalid:",
    len(infrastructure_invalid),
)
print(
    "Saved:",
    summary_csv,
)

if infrastructure_invalid:
    print(
        "Remaining IDs:",
        [
            row["test_id"]
            for row in infrastructure_invalid
        ],
    )
