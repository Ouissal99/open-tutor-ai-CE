import csv
import json
from pathlib import Path


DATASET_PATH = Path(
    "evaluation/test_cases/test_cases_100_article.csv"
)

BASIC_PATH = Path(
    "evaluation/final_results_100/"
    "basic_tool_use_official/"
    "basic_tool_use_results_100.jsonl"
)

NO_RECOVERY_PATH = Path(
    "evaluation/final_results_100/"
    "manager_without_recovery_official/"
    "manager_without_recovery_results_100.csv"
)

FULL_MANAGER_PATH = Path(
    "evaluation/final_results_100/"
    "full_manager_results.csv"
)

FULL_LOCKED_DIR = Path(
    "evaluation/final_results_100/"
    "full_manager_locked_20260725_163823"
)

OUTPUT_DIR = Path(
    "evaluation/final_results_100/"
    "unified_evaluation"
)

OLD_SUMMARY_PATH = (
    OUTPUT_DIR /
    "system_metrics_summary.csv"
)

NEW_SUMMARY_PATH = (
    OUTPUT_DIR /
    "system_metrics_summary_corrected.csv"
)


def read_csv(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_json(path):
    return json.loads(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )


def parse_tools(value):
    if value is None:
        return set()

    if isinstance(value, list):
        return {
            str(item).strip()
            for item in value
            if str(item).strip()
        }

    text = str(value).strip()

    if not text:
        return set()

    return {
        item.strip()
        for item in text.replace(
            ",",
            ";",
        ).split(";")
        if item.strip()
    }


def is_success(call):
    value = call.get("success")

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "yes",
        "1",
        "success",
        "successful",
    }


def attempt_groups(trace):
    history = trace.get(
        "tool_results_history",
        [],
    )

    groups = []

    if isinstance(history, list):
        for attempt in history:
            if not isinstance(attempt, list):
                continue

            calls = [
                call
                for call in attempt
                if isinstance(call, dict)
            ]

            if calls:
                groups.append(calls)

    if groups:
        return groups

    calls = trace.get(
        "tool_results",
        [],
    )

    if isinstance(calls, list):
        calls = [
            call
            for call in calls
            if isinstance(call, dict)
        ]

        if calls:
            return [calls]

    return []


def manager_round_paths(
    system,
    test_id,
    row,
):
    if system == "manager_without_recovery":
        paths = [
            Path(item.strip())
            for item in row.get(
                "trace_paths",
                "",
            ).split(";")
            if item.strip()
        ]

        round_paths = [
            path
            for path in paths
            if "_round" in path.name
        ]

        if round_paths:
            return sorted(round_paths)

        return [
            path
            for path in paths
            if path.exists()
        ]

    paths = sorted(
        FULL_LOCKED_DIR.rglob(
            f"{test_id}_round*_trace.json"
        )
    )

    if paths:
        return paths

    return sorted(
        FULL_LOCKED_DIR.rglob(
            f"{test_id}_trace.json"
        )
    )


def metric_row(
    system,
    display,
    metric,
    numerator,
    denominator,
    applicability,
    definition,
):
    rate = (
        numerator / denominator
        if denominator
        else 0
    )

    return {
        "system": system,
        "system_display": display,
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(rate, 6),
        "percentage": round(
            rate * 100,
            2,
        ),
        "applicability": applicability,
        "status": "Calculated",
        "definition": definition,
    }


def write_csv(
    path,
    rows,
    fieldnames,
):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


dataset = {
    row["test_id"]: row
    for row in read_csv(DATASET_PATH)
}

basic = {
    row["test_id"]: row
    for row in read_jsonl(BASIC_PATH)
}

no_recovery = {
    row["test_id"]: row
    for row in read_csv(NO_RECOVERY_PATH)
}

full_manager = {
    row["test_id"]: row
    for row in read_csv(FULL_MANAGER_PATH)
}


case_rows = []


# ---------------------------------------------------------
# Basic Tool Use
# ---------------------------------------------------------
for test_id, case in dataset.items():
    row = basic[test_id]

    expected = parse_tools(
        case["expected_tools"]
    )

    calls = row.get(
        "tool_outputs",
        [],
    )

    if not isinstance(calls, list):
        calls = []

    calls = [
        call
        for call in calls
        if isinstance(call, dict)
    ]

    selected = {
        str(
            call.get(
                "tool_name",
                "",
            )
        ).strip()
        for call in calls
        if str(
            call.get(
                "tool_name",
                "",
            )
        ).strip()
    }

    successful_tools = {
        str(
            call.get(
                "tool_name",
                "",
            )
        ).strip()
        for call in calls
        if (
            str(
                call.get(
                    "tool_name",
                    "",
                )
            ).strip()
            and is_success(call)
        )
    }

    plan_success = (
        bool(calls)
        and all(
            is_success(call)
            for call in calls
        )
    )

    expected_fulfilled = (
        bool(expected)
        and expected.issubset(
            successful_tools
        )
    )

    case_rows.append(
        {
            "system": "basic_tool_use",
            "test_id": test_id,
            "expected_tools": ";".join(
                sorted(expected)
            ),
            "initial_selected_tools": ";".join(
                sorted(selected)
            ),
            "final_selected_tools": ";".join(
                sorted(selected)
            ),
            "initial_exact_selection": (
                "Yes"
                if selected == expected
                else "No"
            ),
            "final_exact_selection": (
                "Yes"
                if selected == expected
                else "No"
            ),
            "final_plan_execution_success": (
                "Yes"
                if plan_success
                else "No"
            ),
            "all_attempted_calls_success": (
                "Yes"
                if plan_success
                else "No"
            ),
            "expected_tool_fulfillment": (
                "Yes"
                if expected_fulfilled
                else "No"
            ),
            "round_count": 1,
            "attempted_call_count": len(
                calls
            ),
        }
    )


# ---------------------------------------------------------
# Manager conditions
# ---------------------------------------------------------
for system, source in [
    (
        "manager_without_recovery",
        no_recovery,
    ),
    (
        "full_manager",
        full_manager,
    ),
]:
    for test_id, case in dataset.items():
        row = source[test_id]

        expected = parse_tools(
            case["expected_tools"]
        )

        initial_selected = set()
        final_selected = set()
        final_successful_tools = set()

        final_round_successes = []
        all_attempted_successes = []

        round_paths = manager_round_paths(
            system,
            test_id,
            row,
        )

        attempted_call_count = 0

        for path in round_paths:
            if not path.exists():
                continue

            trace = load_json(path)
            groups = attempt_groups(trace)

            if not groups:
                final_round_successes.append(
                    False
                )
                continue

            first_attempt = groups[0]
            final_attempt = groups[-1]

            for call in first_attempt:
                tool_name = str(
                    call.get(
                        "tool_name",
                        "",
                    )
                ).strip()

                if tool_name:
                    initial_selected.add(
                        tool_name
                    )

            for call in final_attempt:
                tool_name = str(
                    call.get(
                        "tool_name",
                        "",
                    )
                ).strip()

                if tool_name:
                    final_selected.add(
                        tool_name
                    )

                if (
                    tool_name
                    and is_success(call)
                ):
                    final_successful_tools.add(
                        tool_name
                    )

            final_round_successes.append(
                bool(final_attempt)
                and all(
                    is_success(call)
                    for call in final_attempt
                )
            )

            for group in groups:
                for call in group:
                    attempted_call_count += 1

                    all_attempted_successes.append(
                        is_success(call)
                    )

        final_plan_success = (
            bool(final_round_successes)
            and all(final_round_successes)
        )

        all_attempted_success = (
            bool(all_attempted_successes)
            and all(
                all_attempted_successes
            )
        )

        expected_fulfilled = (
            bool(expected)
            and expected.issubset(
                final_successful_tools
            )
        )

        case_rows.append(
            {
                "system": system,
                "test_id": test_id,
                "expected_tools": ";".join(
                    sorted(expected)
                ),
                "initial_selected_tools": (
                    ";".join(
                        sorted(
                            initial_selected
                        )
                    )
                ),
                "final_selected_tools": (
                    ";".join(
                        sorted(
                            final_selected
                        )
                    )
                ),
                "initial_exact_selection": (
                    "Yes"
                    if initial_selected
                    == expected
                    else "No"
                ),
                "final_exact_selection": (
                    "Yes"
                    if final_selected
                    == expected
                    else "No"
                ),
                "final_plan_execution_success": (
                    "Yes"
                    if final_plan_success
                    else "No"
                ),
                "all_attempted_calls_success": (
                    "Yes"
                    if all_attempted_success
                    else "No"
                ),
                "expected_tool_fulfillment": (
                    "Yes"
                    if expected_fulfilled
                    else "No"
                ),
                "round_count": len(
                    round_paths
                ),
                "attempted_call_count": (
                    attempted_call_count
                ),
            }
        )


details_path = (
    OUTPUT_DIR /
    "core_metric_case_details_corrected.csv"
)

write_csv(
    details_path,
    case_rows,
    [
        "system",
        "test_id",
        "expected_tools",
        "initial_selected_tools",
        "final_selected_tools",
        "initial_exact_selection",
        "final_exact_selection",
        "final_plan_execution_success",
        "all_attempted_calls_success",
        "expected_tool_fulfillment",
        "round_count",
        "attempted_call_count",
    ],
)


display_names = {
    "basic_tool_use": "Basic Tool Use",
    "manager_without_recovery": (
        "Manager Without Recovery"
    ),
    "full_manager": "Full Manager",
}


corrected_rows = []

for system, display in display_names.items():
    rows = [
        row
        for row in case_rows
        if row["system"] == system
    ]

    denominator = len(rows)

    initial_exact = sum(
        row["initial_exact_selection"]
        == "Yes"
        for row in rows
    )

    final_exact = sum(
        row["final_exact_selection"]
        == "Yes"
        for row in rows
    )

    execution_success = sum(
        row[
            "final_plan_execution_success"
        ]
        == "Yes"
        for row in rows
    )

    all_attempted_success = sum(
        row[
            "all_attempted_calls_success"
        ]
        == "Yes"
        for row in rows
    )

    expected_fulfilled = sum(
        row[
            "expected_tool_fulfillment"
        ]
        == "Yes"
        for row in rows
    )

    corrected_rows.extend(
        [
            metric_row(
                system,
                display,
                "Tool Selection Accuracy",
                initial_exact,
                denominator,
                "Applicable",
                (
                    "Cases where the initial "
                    "selected tool set exactly "
                    "matches the expected set. "
                    "Recovery selections are "
                    "excluded."
                ),
            ),
            metric_row(
                system,
                display,
                "Execution Success Rate",
                execution_success,
                denominator,
                "Applicable",
                (
                    "Cases where every tool call "
                    "in the final attempt of every "
                    "planned tutoring round "
                    "succeeded."
                ),
            ),
            metric_row(
                system,
                display,
                "Final Tool Selection Accuracy",
                final_exact,
                denominator,
                "Supplementary",
                (
                    "Cases where the final "
                    "post-recovery selected set "
                    "exactly matches the expected "
                    "set."
                ),
            ),
            metric_row(
                system,
                display,
                "Expected-Tool Fulfillment Rate",
                expected_fulfilled,
                denominator,
                "Supplementary",
                (
                    "Cases where every expected "
                    "tool produced at least one "
                    "successful output in the "
                    "final plan."
                ),
            ),
            metric_row(
                system,
                display,
                (
                    "All Attempted Calls "
                    "Successful Case Rate"
                ),
                all_attempted_success,
                denominator,
                "Supplementary",
                (
                    "Cases where all tool calls, "
                    "including failed calls before "
                    "recovery, succeeded."
                ),
            ),
        ]
    )


old_summary = read_csv(
    OLD_SUMMARY_PATH
)

replacement_keys = {
    (
        row["system"],
        row["metric"],
    )
    for row in corrected_rows
}

new_summary = [
    row
    for row in old_summary
    if (
        row["system"],
        row["metric"],
    )
    not in replacement_keys
]

new_summary.extend(
    corrected_rows
)

system_order = {
    "direct_llm": 0,
    "basic_tool_use": 1,
    "manager_without_recovery": 2,
    "full_manager": 3,
}

metric_order = {
    "Tool Call Success Rate": 0,
    "Tool Selection Accuracy": 1,
    "Execution Success Rate": 2,
    "Validation Pass Rate": 3,
    "Recovery Success Rate": 4,
    "Trace Completeness": 5,
    "End-to-End Success Rate": 6,
    "Tutoring Answer Accuracy": 7,
    "Final Tool Selection Accuracy": 8,
    "Expected-Tool Fulfillment Rate": 9,
    (
        "All Attempted Calls "
        "Successful Case Rate"
    ): 10,
    "Strict Complete-Trace Rate": 11,
    "Operational Answer Completion Rate": 12,
}

new_summary.sort(
    key=lambda row: (
        system_order.get(
            row["system"],
            99,
        ),
        metric_order.get(
            row["metric"],
            99,
        ),
    )
)

write_csv(
    NEW_SUMMARY_PATH,
    new_summary,
    [
        "system",
        "system_display",
        "metric",
        "numerator",
        "denominator",
        "rate",
        "percentage",
        "applicability",
        "status",
        "definition",
    ],
)


audit_lines = [
    "CORRECTED CORE METRIC AUDIT",
    "=" * 72,
    "",
]

for system, display in display_names.items():
    rows = [
        row
        for row in corrected_rows
        if row["system"] == system
    ]

    audit_lines.append(
        display.upper()
    )
    audit_lines.append("-" * 72)

    for row in rows:
        audit_lines.append(
            f"{row['metric']}: "
            f"{row['numerator']}/"
            f"{row['denominator']} "
            f"({row['percentage']}%)"
        )

    audit_lines.append("")


audit_lines.extend(
    [
        "DEFINITION NOTES",
        "-" * 72,
        (
            "Tool Selection Accuracy uses only "
            "the initial selected tools, so "
            "recovery alternatives do not "
            "contaminate selection quality."
        ),
        (
            "Execution Success Rate uses the "
            "final attempt of every tutoring "
            "round. Earlier failed attempts are "
            "retained in Tool Call Success Rate "
            "and recovery diagnostics."
        ),
        (
            "Expected-Tool Fulfillment is "
            "reported only as a supplementary "
            "diagnostic and is not the official "
            "Execution Success Rate."
        ),
    ]
)

audit = "\n".join(
    audit_lines
) + "\n"

(
    OUTPUT_DIR /
    "core_metric_correction_audit.txt"
).write_text(
    audit,
    encoding="utf-8",
)

print(audit)

print("Created:")
print(" -", details_path)
print(" -", NEW_SUMMARY_PATH)
print(
    " -",
    OUTPUT_DIR /
    "core_metric_correction_audit.txt",
)
