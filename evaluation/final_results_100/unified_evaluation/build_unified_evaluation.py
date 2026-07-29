import csv
import json
import random
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(
    "evaluation/test_cases/test_cases_100_article.csv"
)

DIRECT_PATH = Path(
    "evaluation/final_results_100/"
    "direct_llm_official/"
    "direct_llm_results_100.csv"
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


SYSTEMS = [
    "direct_llm",
    "basic_tool_use",
    "manager_without_recovery",
    "full_manager",
]

SYSTEM_DISPLAY = {
    "direct_llm": "Direct LLM",
    "basic_tool_use": "Basic Tool Use",
    "manager_without_recovery": (
        "Manager Without Recovery"
    ),
    "full_manager": "Full Manager",
}


TRACE_ELEMENTS = [
    "trace_identity",
    "question",
    "task_classification",
    "tool_plan",
    "selected_tools",
    "tool_io",
    "attempt",
    "validation",
    "recovery_status",
    "timestamps",
    "final_output_package",
]


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
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except Exception:
        return None


def write_csv(path, rows, fieldnames):
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

    text = text.replace(",", ";")

    return {
        item.strip()
        for item in text.split(";")
        if item.strip()
    }


def bool_value(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "yes",
        "1",
        "success",
        "successful",
        "valid",
        "validated",
        "passed",
        "pass",
    }


def int_value(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def is_validation_pass(value):
    return str(value).strip().lower() in {
        "valid",
        "validated",
        "passed",
        "pass",
        "success",
        "successful",
    }


def nonempty(value):
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, dict)):
        return bool(value)

    return True


def flatten_tool_attempts(trace):
    history = trace.get(
        "tool_results_history",
        [],
    )

    flattened = []

    if isinstance(history, list) and history:
        for attempt_index, attempt_calls in enumerate(
            history,
            start=1,
        ):
            if not isinstance(attempt_calls, list):
                continue

            for call in attempt_calls:
                if isinstance(call, dict):
                    copied = dict(call)
                    copied["_history_attempt"] = (
                        attempt_index
                    )
                    flattened.append(copied)

        if flattened:
            return flattened

    calls = trace.get("tool_results", [])

    if not isinstance(calls, list):
        return []

    return [
        dict(call)
        for call in calls
        if isinstance(call, dict)
    ]


def validation_attempts(trace):
    reports = trace.get(
        "validation_reports",
        [],
    )

    if isinstance(reports, list) and reports:
        return [
            report
            for report in reports
            if isinstance(report, dict)
        ]

    report = trace.get("validation_report")

    if isinstance(report, dict):
        return [report]

    package = trace.get(
        "final_package",
        trace.get("output_package", {}),
    )

    if isinstance(package, dict):
        report = package.get(
            "validation_report"
        )

        if isinstance(report, dict):
            return [report]

    return []


def trace_recovery_triggered(trace):
    return (
        bool_value(trace.get("recovery_used"))
        or int_value(trace.get("attempts"), 1) > 1
        or bool(
            trace.get("failure_recovery_steps")
        )
    )


def trace_completeness(trace):
    calls = flatten_tool_attempts(trace)
    validations = validation_attempts(trace)

    has_selected_tools = bool(calls)

    tool_selection = trace.get(
        "tool_selection"
    )

    if isinstance(tool_selection, dict):
        has_selected_tools = (
            has_selected_tools
            or bool(tool_selection)
        )

    timestamps = (
        nonempty(trace.get("created_at"))
        and (
            nonempty(trace.get("updated_at"))
            or bool(trace.get("events"))
        )
    )

    package = trace.get(
        "final_package",
        trace.get("output_package"),
    )

    checks = {
        "trace_identity": (
            nonempty(trace.get("trace_id"))
            and nonempty(trace.get("request_id"))
        ),
        "question": nonempty(
            trace.get("student_question")
        ),
        "task_classification": (
            isinstance(
                trace.get("analyzed_task"),
                dict,
            )
            and nonempty(
                trace["analyzed_task"].get(
                    "task_type"
                )
            )
        ),
        "tool_plan": (
            isinstance(
                trace.get("tool_plan"),
                dict,
            )
            and bool(
                trace["tool_plan"].get(
                    "steps"
                )
            )
        ),
        "selected_tools": has_selected_tools,
        "tool_io": (
            bool(calls)
            and all(
                nonempty(call.get("tool_name"))
                and (
                    "success" in call
                    or nonempty(call.get("status"))
                )
                and (
                    nonempty(call.get("output"))
                    or nonempty(call.get("content"))
                )
                for call in calls
            )
        ),
        "attempt": (
            "attempts" in trace
            and int_value(
                trace.get("attempts"),
                0,
            ) >= 1
        ),
        "validation": bool(validations),
        "recovery_status": (
            "recovery_used" in trace
            and "attempts" in trace
        ),
        "timestamps": timestamps,
        "final_output_package": (
            isinstance(package, dict)
            and nonempty(package.get("status"))
            and isinstance(
                package.get(
                    "validation_report"
                ),
                dict,
            )
        ),
    }

    present = sum(
        1
        for value in checks.values()
        if value
    )

    expected = len(TRACE_ELEMENTS)

    return checks, present, expected


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

    round_paths = sorted(
        FULL_LOCKED_DIR.rglob(
            f"{test_id}_round*_trace.json"
        )
    )

    if round_paths:
        return round_paths

    return sorted(
        FULL_LOCKED_DIR.rglob(
            f"{test_id}_trace.json"
        )
    )


def no_recovery_answer(row):
    path = Path(row.get("report_json", ""))

    if not path.exists():
        return ""

    report = load_json(path)

    if not isinstance(report, dict):
        return ""

    response = report.get("response", {})

    if isinstance(response, dict):
        answer = response.get("answer")

        if nonempty(answer):
            return str(answer)

    answer = report.get("final_answer")

    return str(answer) if nonempty(answer) else ""


def final_validation_status(trace):
    reports = validation_attempts(trace)

    if reports:
        return str(
            reports[-1].get("status", "")
        )

    return ""


def add_metric(
    rows,
    system,
    metric,
    numerator=None,
    denominator=None,
    applicability="Applicable",
    status="Calculated",
    definition="",
):
    if (
        numerator is None
        or denominator in {None, 0}
    ):
        rate = ""
    else:
        rate = round(
            numerator / denominator,
            6,
        )

    rows.append(
        {
            "system": system,
            "system_display": (
                SYSTEM_DISPLAY[system]
            ),
            "metric": metric,
            "numerator": (
                "" if numerator is None
                else numerator
            ),
            "denominator": (
                "" if denominator is None
                else denominator
            ),
            "rate": rate,
            "percentage": (
                ""
                if rate == ""
                else round(rate * 100, 2)
            ),
            "applicability": applicability,
            "status": status,
            "definition": definition,
        }
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

dataset_rows = read_csv(DATASET_PATH)

dataset = {
    row["test_id"]: row
    for row in dataset_rows
}

direct_rows = {
    row["test_id"]: row
    for row in read_csv(DIRECT_PATH)
}

basic_rows = {
    row["test_id"]: row
    for row in read_jsonl(BASIC_PATH)
}

no_recovery_rows = {
    row["test_id"]: row
    for row in read_csv(NO_RECOVERY_PATH)
}

full_rows = {
    row["test_id"]: row
    for row in read_csv(FULL_MANAGER_PATH)
}


per_case = []
tool_call_details = []
validation_details = []
trace_details = []
recovery_details = []
answers = {}

missing_trace_cases = {
    "manager_without_recovery": [],
    "full_manager": [],
}


# ---------------------------------------------------------
# Direct LLM
# ---------------------------------------------------------
for test_id, case in dataset.items():
    row = direct_rows[test_id]

    answer = row.get("final_answer", "")
    answers[("direct_llm", test_id)] = answer

    operational_complete = (
        row.get("status") == "ok"
        and nonempty(answer)
    )

    per_case.append(
        {
            "system": "direct_llm",
            "test_id": test_id,
            "category": case["category"],
            "expected_tools": (
                case["expected_tools"]
            ),
            "selected_tools": "",
            "exact_tool_selection": "N/A",
            "required_execution_success": "N/A",
            "all_attempted_calls_success": "N/A",
            "workflow_status": "N/A",
            "final_validation_status": "N/A",
            "recovery_triggered": "N/A",
            "recovery_successful": "N/A",
            "strict_trace_complete": "N/A",
            "trace_element_ratio": "N/A",
            "answer_generated": (
                "Yes" if nonempty(answer)
                else "No"
            ),
            "operational_completion": (
                "Yes"
                if operational_complete
                else "No"
            ),
            "judge_correct": "Pending",
            "official_end_to_end_success": (
                "Pending"
            ),
        }
    )


# ---------------------------------------------------------
# Basic Tool Use
# ---------------------------------------------------------
for test_id, case in dataset.items():
    row = basic_rows[test_id]

    expected = parse_tools(
        case["expected_tools"]
    )
    selected = parse_tools(
        row.get("selected_tools")
    )

    calls = row.get("tool_outputs", [])

    if not isinstance(calls, list):
        calls = []

    successful_expected = {
        str(call.get("tool_name", "")).strip()
        for call in calls
        if (
            isinstance(call, dict)
            and bool_value(call.get("success"))
        )
    }

    required_execution_success = (
        bool(expected)
        and expected.issubset(
            successful_expected
        )
    )

    all_calls_success = (
        bool(calls)
        and all(
            bool_value(call.get("success"))
            for call in calls
            if isinstance(call, dict)
        )
    )

    for index, call in enumerate(
        calls,
        start=1,
    ):
        if not isinstance(call, dict):
            continue

        metadata = call.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        tool_call_details.append(
            {
                "system": "basic_tool_use",
                "test_id": test_id,
                "round_trace": "",
                "trace_id": "",
                "attempt": metadata.get(
                    "attempt",
                    1,
                ),
                "call_index": index,
                "tool_name": call.get(
                    "tool_name",
                    "",
                ),
                "status": call.get(
                    "status",
                    "",
                ),
                "success": (
                    "Yes"
                    if bool_value(
                        call.get("success")
                    )
                    else "No"
                ),
                "failure_reason": (
                    metadata.get(
                        "failure_reason",
                        "",
                    )
                ),
                "output_present": (
                    "Yes"
                    if nonempty(
                        call.get("output")
                    )
                    else "No"
                ),
            }
        )

    answer = row.get("final_answer", "")
    answers[("basic_tool_use", test_id)] = answer

    operational_complete = (
        row.get("status") == "ok"
        and nonempty(answer)
    )

    per_case.append(
        {
            "system": "basic_tool_use",
            "test_id": test_id,
            "category": case["category"],
            "expected_tools": ";".join(
                sorted(expected)
            ),
            "selected_tools": ";".join(
                sorted(selected)
            ),
            "exact_tool_selection": (
                "Yes"
                if selected == expected
                else "No"
            ),
            "required_execution_success": (
                "Yes"
                if required_execution_success
                else "No"
            ),
            "all_attempted_calls_success": (
                "Yes"
                if all_calls_success
                else "No"
            ),
            "workflow_status": "N/A",
            "final_validation_status": "N/A",
            "recovery_triggered": "N/A",
            "recovery_successful": "N/A",
            "strict_trace_complete": "N/A",
            "trace_element_ratio": "N/A",
            "answer_generated": (
                "Yes"
                if nonempty(answer)
                else "No"
            ),
            "operational_completion": (
                "Yes"
                if operational_complete
                else "No"
            ),
            "judge_correct": "Pending",
            "official_end_to_end_success": (
                "Pending"
            ),
        }
    )


# ---------------------------------------------------------
# Manager systems
# ---------------------------------------------------------
for system, source_rows in [
    (
        "manager_without_recovery",
        no_recovery_rows,
    ),
    (
        "full_manager",
        full_rows,
    ),
]:
    for test_id, case in dataset.items():
        row = source_rows[test_id]
        expected = parse_tools(
            case["expected_tools"]
        )

        paths = manager_round_paths(
            system,
            test_id,
            row,
        )

        traces = []

        for path in paths:
            if not path.exists():
                continue

            trace = load_json(path)

            if isinstance(trace, dict):
                traces.append(
                    (path, trace)
                )

        if not traces:
            missing_trace_cases[
                system
            ].append(test_id)

        selected = set()
        successful_expected = set()
        case_calls = []
        case_validation_statuses = []
        recovery_rounds = []
        completeness_scores = []
        strict_trace_flags = []

        for round_index, (
            path,
            trace,
        ) in enumerate(
            traces,
            start=1,
        ):
            calls = flatten_tool_attempts(
                trace
            )
            case_calls.extend(calls)

            for call_index, call in enumerate(
                calls,
                start=1,
            ):
                tool_name = str(
                    call.get(
                        "tool_name",
                        "",
                    )
                ).strip()

                if tool_name:
                    selected.add(tool_name)

                success = bool_value(
                    call.get("success")
                )

                if success and tool_name:
                    successful_expected.add(
                        tool_name
                    )

                metadata = call.get(
                    "metadata",
                    {},
                )

                if not isinstance(
                    metadata,
                    dict,
                ):
                    metadata = {}

                attempt = metadata.get(
                    "attempt",
                    call.get(
                        "_history_attempt",
                        1,
                    ),
                )

                tool_call_details.append(
                    {
                        "system": system,
                        "test_id": test_id,
                        "round_trace": (
                            path.name
                        ),
                        "trace_id": trace.get(
                            "trace_id",
                            "",
                        ),
                        "attempt": attempt,
                        "call_index": call_index,
                        "tool_name": tool_name,
                        "status": call.get(
                            "status",
                            "",
                        ),
                        "success": (
                            "Yes"
                            if success
                            else "No"
                        ),
                        "failure_reason": (
                            metadata.get(
                                "failure_reason",
                                "",
                            )
                        ),
                        "output_present": (
                            "Yes"
                            if (
                                nonempty(
                                    call.get(
                                        "output"
                                    )
                                )
                                or nonempty(
                                    call.get(
                                        "content"
                                    )
                                )
                            )
                            else "No"
                        ),
                    }
                )

            validations = validation_attempts(
                trace
            )

            for validation_index, validation in enumerate(
                validations,
                start=1,
            ):
                status = validation.get(
                    "status",
                    "",
                )

                case_validation_statuses.append(
                    status
                )

                validation_details.append(
                    {
                        "system": system,
                        "test_id": test_id,
                        "round_trace": (
                            path.name
                        ),
                        "trace_id": trace.get(
                            "trace_id",
                            "",
                        ),
                        "validation_index": (
                            validation_index
                        ),
                        "status": status,
                        "passed": (
                            "Yes"
                            if is_validation_pass(
                                status
                            )
                            else "No"
                        ),
                        "failure_reason": (
                            validation.get(
                                "failure_reason",
                                "",
                            )
                        ),
                        "recommended_action": (
                            validation.get(
                                "recommended_action",
                                "",
                            )
                        ),
                    }
                )

            recovery_triggered = (
                trace_recovery_triggered(
                    trace
                )
            )

            final_round_validation = (
                final_validation_status(
                    trace
                )
            )

            recovery_successful = (
                recovery_triggered
                and is_validation_pass(
                    final_round_validation
                )
            )

            recovery_rounds.append(
                {
                    "triggered": (
                        recovery_triggered
                    ),
                    "successful": (
                        recovery_successful
                    ),
                }
            )

            recovery_details.append(
                {
                    "system": system,
                    "test_id": test_id,
                    "round_trace": (
                        path.name
                    ),
                    "trace_id": trace.get(
                        "trace_id",
                        "",
                    ),
                    "attempts": int_value(
                        trace.get(
                            "attempts",
                            1,
                        ),
                        1,
                    ),
                    "recovery_triggered": (
                        "Yes"
                        if recovery_triggered
                        else "No"
                    ),
                    "recovery_successful": (
                        "Yes"
                        if recovery_successful
                        else "No"
                    ),
                    "final_validation_status": (
                        final_round_validation
                    ),
                    "failure_recovery_step_count": (
                        len(
                            trace.get(
                                "failure_recovery_steps",
                                [],
                            )
                        )
                        if isinstance(
                            trace.get(
                                "failure_recovery_steps",
                                [],
                            ),
                            list,
                        )
                        else 0
                    ),
                }
            )

            checks, present, expected_elements = (
                trace_completeness(trace)
            )

            ratio = (
                present / expected_elements
                if expected_elements
                else 0
            )

            completeness_scores.append(
                ratio
            )

            strict_complete = (
                present == expected_elements
            )

            strict_trace_flags.append(
                strict_complete
            )

            detail = {
                "system": system,
                "test_id": test_id,
                "round_trace": path.name,
                "trace_id": trace.get(
                    "trace_id",
                    "",
                ),
                "elements_present": present,
                "elements_expected": (
                    expected_elements
                ),
                "completeness_ratio": round(
                    ratio,
                    6,
                ),
                "strict_complete": (
                    "Yes"
                    if strict_complete
                    else "No"
                ),
            }

            for element in TRACE_ELEMENTS:
                detail[element] = (
                    "Yes"
                    if checks[element]
                    else "No"
                )

            trace_details.append(detail)

        required_execution_success = (
            bool(expected)
            and expected.issubset(
                successful_expected
            )
        )

        all_calls_success = (
            bool(case_calls)
            and all(
                bool_value(
                    call.get("success")
                )
                for call in case_calls
            )
        )

        case_recovery_triggered = any(
            item["triggered"]
            for item in recovery_rounds
        )

        case_recovery_successful = (
            case_recovery_triggered
            and any(
                item["successful"]
                for item in recovery_rounds
            )
        )

        strict_case_trace = (
            bool(strict_trace_flags)
            and all(strict_trace_flags)
        )

        case_trace_ratio = (
            sum(completeness_scores)
            / len(completeness_scores)
            if completeness_scores
            else 0
        )

        if system == (
            "manager_without_recovery"
        ):
            answer = no_recovery_answer(
                row
            )
            workflow_status = row.get(
                "workflow_status",
                "",
            )
            provider_valid = (
                row.get("provider_status")
                == "available"
            )
        else:
            answer = row.get(
                "final_answer",
                "",
            )
            workflow_status = row.get(
                "workflow_status",
                "",
            )
            provider_valid = (
                row.get("provider_status")
                == "available"
            )

        answers[(system, test_id)] = answer

        final_case_validation = (
            case_validation_statuses[-1]
            if case_validation_statuses
            else ""
        )

        operational_complete = (
            provider_valid
            and nonempty(answer)
        )

        per_case.append(
            {
                "system": system,
                "test_id": test_id,
                "category": case["category"],
                "expected_tools": ";".join(
                    sorted(expected)
                ),
                "selected_tools": ";".join(
                    sorted(selected)
                ),
                "exact_tool_selection": (
                    "Yes"
                    if selected == expected
                    else "No"
                ),
                "required_execution_success": (
                    "Yes"
                    if required_execution_success
                    else "No"
                ),
                "all_attempted_calls_success": (
                    "Yes"
                    if all_calls_success
                    else "No"
                ),
                "workflow_status": (
                    workflow_status
                ),
                "final_validation_status": (
                    final_case_validation
                ),
                "recovery_triggered": (
                    "Yes"
                    if case_recovery_triggered
                    else "No"
                ),
                "recovery_successful": (
                    "Yes"
                    if case_recovery_successful
                    else "No"
                ),
                "strict_trace_complete": (
                    "Yes"
                    if strict_case_trace
                    else "No"
                ),
                "trace_element_ratio": round(
                    case_trace_ratio,
                    6,
                ),
                "answer_generated": (
                    "Yes"
                    if nonempty(answer)
                    else "No"
                ),
                "operational_completion": (
                    "Yes"
                    if operational_complete
                    else "No"
                ),
                "judge_correct": "Pending",
                "official_end_to_end_success": (
                    "Pending"
                ),
            }
        )


# ---------------------------------------------------------
# Main metric summary
# ---------------------------------------------------------
metric_rows = []

per_case_by_system = {
    system: [
        row
        for row in per_case
        if row["system"] == system
    ]
    for system in SYSTEMS
}

calls_by_system = {
    system: [
        row
        for row in tool_call_details
        if row["system"] == system
    ]
    for system in SYSTEMS
}

validations_by_system = {
    system: [
        row
        for row in validation_details
        if row["system"] == system
    ]
    for system in SYSTEMS
}

traces_by_system = {
    system: [
        row
        for row in trace_details
        if row["system"] == system
    ]
    for system in SYSTEMS
}


for system in SYSTEMS:
    cases = per_case_by_system[system]

    operational_count = sum(
        row["operational_completion"]
        == "Yes"
        for row in cases
    )

    add_metric(
        metric_rows,
        system,
        "Operational Answer Completion Rate",
        operational_count,
        len(cases),
        applicability="Diagnostic",
        definition=(
            "Provider-valid case with a "
            "non-empty final answer. This is "
            "not semantic correctness."
        ),
    )

    if system == "direct_llm":
        for metric in [
            "Tool Call Success Rate",
            "Tool Selection Accuracy",
            "Execution Success Rate",
            "Validation Pass Rate",
            "Recovery Success Rate",
            "Trace Completeness",
        ]:
            add_metric(
                metric_rows,
                system,
                metric,
                applicability="N/A",
                status="Not applicable",
                definition=(
                    "This condition does not "
                    "implement this mechanism."
                ),
            )

    else:
        calls = calls_by_system[system]

        successful_calls = sum(
            row["success"] == "Yes"
            for row in calls
        )

        add_metric(
            metric_rows,
            system,
            "Tool Call Success Rate",
            successful_calls,
            len(calls),
            definition=(
                "Successful individual tool "
                "calls divided by all attempted "
                "tool calls."
            ),
        )

        exact_selection = sum(
            row["exact_tool_selection"]
            == "Yes"
            for row in cases
        )

        add_metric(
            metric_rows,
            system,
            "Tool Selection Accuracy",
            exact_selection,
            len(cases),
            definition=(
                "Cases where the selected tool "
                "set exactly matches the expected "
                "tool set."
            ),
        )

        execution_success = sum(
            row[
                "required_execution_success"
            ]
            == "Yes"
            for row in cases
        )

        tool_cases = sum(
            bool(
                parse_tools(
                    row["expected_tools"]
                )
            )
            for row in cases
        )

        add_metric(
            metric_rows,
            system,
            "Execution Success Rate",
            execution_success,
            tool_cases,
            definition=(
                "Tool-requiring cases where "
                "every expected tool produced "
                "at least one successful call."
            ),
        )

        all_call_case_success = sum(
            row[
                "all_attempted_calls_success"
            ]
            == "Yes"
            for row in cases
        )

        add_metric(
            metric_rows,
            system,
            (
                "All Attempted Calls Successful "
                "Case Rate"
            ),
            all_call_case_success,
            tool_cases,
            applicability="Supplementary",
            definition=(
                "Cases where every attempted "
                "tool call succeeded, including "
                "extra selected tools."
            ),
        )

    if system in {
        "manager_without_recovery",
        "full_manager",
    }:
        validations = (
            validations_by_system[system]
        )

        passed_validations = sum(
            row["passed"] == "Yes"
            for row in validations
        )

        add_metric(
            metric_rows,
            system,
            "Validation Pass Rate",
            passed_validations,
            len(validations),
            definition=(
                "Passed attempt-level validation "
                "reports divided by all outputs "
                "subjected to validation."
            ),
        )

        traces = traces_by_system[system]

        present_elements = sum(
            int(row["elements_present"])
            for row in traces
        )

        expected_elements = sum(
            int(row["elements_expected"])
            for row in traces
        )

        add_metric(
            metric_rows,
            system,
            "Trace Completeness",
            present_elements,
            expected_elements,
            definition=(
                "Populated required lifecycle "
                "trace elements divided by all "
                "expected trace elements."
            ),
        )

        strict_complete = sum(
            row["strict_complete"] == "Yes"
            for row in traces
        )

        add_metric(
            metric_rows,
            system,
            "Strict Complete-Trace Rate",
            strict_complete,
            len(traces),
            applicability="Supplementary",
            definition=(
                "Round traces containing every "
                "required lifecycle element."
            ),
        )

    else:
        if system == "basic_tool_use":
            add_metric(
                metric_rows,
                system,
                "Validation Pass Rate",
                applicability="N/A",
                status="Not applicable",
                definition=(
                    "The Basic baseline performs "
                    "no output validation."
                ),
            )

            add_metric(
                metric_rows,
                system,
                "Trace Completeness",
                applicability="N/A",
                status="Not applicable",
                definition=(
                    "The Basic baseline has no "
                    "centralized lifecycle trace."
                ),
            )

    if system == "full_manager":
        recovery_cases = [
            row
            for row in cases
            if row["recovery_triggered"]
            == "Yes"
        ]

        recovery_successes = sum(
            row["recovery_successful"]
            == "Yes"
            for row in recovery_cases
        )

        add_metric(
            metric_rows,
            system,
            "Recovery Success Rate",
            recovery_successes,
            len(recovery_cases),
            definition=(
                "Cases with manager recovery "
                "triggered where at least one "
                "recovery-triggered round "
                "eventually passed validation."
            ),
        )

    elif system in {
        "basic_tool_use",
        "manager_without_recovery",
    }:
        add_metric(
            metric_rows,
            system,
            "Recovery Success Rate",
            applicability="N/A",
            status="Not applicable",
            definition=(
                "Recovery is absent or explicitly "
                "disabled in this condition."
            ),
        )

    add_metric(
        metric_rows,
        system,
        "End-to-End Success Rate",
        applicability="Pending",
        status="Pending GPT judge",
        definition=(
            "Operational completion combined "
            "with semantically correct final "
            "answer. Calculated after judging."
        ),
    )

    add_metric(
        metric_rows,
        system,
        "Tutoring Answer Accuracy",
        applicability="Pending",
        status="Pending GPT judge",
        definition=(
            "Semantically correct tutoring "
            "answers under the fixed judge "
            "rubric and three-run aggregation."
        ),
    )


# ---------------------------------------------------------
# Anonymized candidate dataset
# ---------------------------------------------------------
judge_candidates = []
judge_mapping = []

for test_id, case in dataset.items():
    shuffled_systems = list(SYSTEMS)

    seed = (
        20260726
        + int(test_id[1:])
    )

    random.Random(seed).shuffle(
        shuffled_systems
    )

    anonymous_labels = [
        "A",
        "B",
        "C",
        "D",
    ]

    for anonymous_label, system in zip(
        anonymous_labels,
        shuffled_systems,
    ):
        candidate_id = (
            f"{test_id}-{anonymous_label}"
        )

        answer = answers.get(
            (system, test_id),
            "",
        )

        judge_candidates.append(
            {
                "candidate_id": candidate_id,
                "test_id": test_id,
                "category": case["category"],
                "question": case["question"],
                "expected_tools": (
                    case["expected_tools"]
                ),
                "anonymous_system": (
                    anonymous_label
                ),
                "candidate_answer": answer,
                "reference_answer": "",
                "expected_facts": [],
            }
        )

        judge_mapping.append(
            {
                "candidate_id": candidate_id,
                "test_id": test_id,
                "anonymous_system": (
                    anonymous_label
                ),
                "actual_system": system,
                "actual_system_display": (
                    SYSTEM_DISPLAY[system]
                ),
            }
        )


# ---------------------------------------------------------
# Write files
# ---------------------------------------------------------
write_csv(
    OUTPUT_DIR
    / "per_case_system_metrics.csv",
    per_case,
    [
        "system",
        "test_id",
        "category",
        "expected_tools",
        "selected_tools",
        "exact_tool_selection",
        "required_execution_success",
        "all_attempted_calls_success",
        "workflow_status",
        "final_validation_status",
        "recovery_triggered",
        "recovery_successful",
        "strict_trace_complete",
        "trace_element_ratio",
        "answer_generated",
        "operational_completion",
        "judge_correct",
        "official_end_to_end_success",
    ],
)

write_csv(
    OUTPUT_DIR
    / "system_metrics_summary.csv",
    metric_rows,
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

write_csv(
    OUTPUT_DIR
    / "tool_call_details.csv",
    tool_call_details,
    [
        "system",
        "test_id",
        "round_trace",
        "trace_id",
        "attempt",
        "call_index",
        "tool_name",
        "status",
        "success",
        "failure_reason",
        "output_present",
    ],
)

write_csv(
    OUTPUT_DIR
    / "validation_details.csv",
    validation_details,
    [
        "system",
        "test_id",
        "round_trace",
        "trace_id",
        "validation_index",
        "status",
        "passed",
        "failure_reason",
        "recommended_action",
    ],
)

trace_fields = [
    "system",
    "test_id",
    "round_trace",
    "trace_id",
    "elements_present",
    "elements_expected",
    "completeness_ratio",
    "strict_complete",
] + TRACE_ELEMENTS

write_csv(
    OUTPUT_DIR
    / "trace_completeness_details.csv",
    trace_details,
    trace_fields,
)

write_csv(
    OUTPUT_DIR
    / "recovery_details.csv",
    recovery_details,
    [
        "system",
        "test_id",
        "round_trace",
        "trace_id",
        "attempts",
        "recovery_triggered",
        "recovery_successful",
        "final_validation_status",
        "failure_recovery_step_count",
    ],
)

with (
    OUTPUT_DIR
    / "judge_candidates_anonymized.jsonl"
).open(
    "w",
    encoding="utf-8",
) as handle:
    for row in judge_candidates:
        handle.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

write_csv(
    OUTPUT_DIR
    / "judge_identity_mapping_secret.csv",
    judge_mapping,
    [
        "candidate_id",
        "test_id",
        "anonymous_system",
        "actual_system",
        "actual_system_display",
    ],
)


# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------
summary_lookup = {
    (
        row["system"],
        row["metric"],
    ): row
    for row in metric_rows
}

audit_lines = [
    "UNIFIED EVALUATION AUDIT",
    "=" * 72,
    "",
    f"Dataset cases: {len(dataset)}",
    f"Per-case rows: {len(per_case)}",
    (
        "Expected per-case rows: "
        f"{len(dataset) * len(SYSTEMS)}"
    ),
    (
        "Anonymized judge candidates: "
        f"{len(judge_candidates)}"
    ),
    "",
]

for system in SYSTEMS:
    audit_lines.append(
        SYSTEM_DISPLAY[system].upper()
    )
    audit_lines.append("-" * 72)

    audit_lines.append(
        "Cases: "
        + str(
            len(
                per_case_by_system[
                    system
                ]
            )
        )
    )

    audit_lines.append(
        "Tool calls: "
        + str(
            len(
                calls_by_system[
                    system
                ]
            )
        )
    )

    if system in missing_trace_cases:
        audit_lines.append(
            "Missing trace cases: "
            + (
                ", ".join(
                    missing_trace_cases[
                        system
                    ]
                )
                or "none"
            )
        )

    for metric in [
        "Tool Call Success Rate",
        "Tool Selection Accuracy",
        "Execution Success Rate",
        "Validation Pass Rate",
        "Recovery Success Rate",
        "Trace Completeness",
        "Operational Answer Completion Rate",
    ]:
        row = summary_lookup.get(
            (system, metric)
        )

        if row is None:
            continue

        value = row["percentage"]

        audit_lines.append(
            f"{metric}: "
            + (
                "N/A or pending"
                if value == ""
                else f"{value}%"
            )
        )

    audit_lines.append("")


audit_lines.extend(
    [
        "IMPORTANT INTERPRETATION",
        "-" * 72,
        (
            "Operational completion is not "
            "semantic tutoring correctness."
        ),
        (
            "End-to-End Success Rate and "
            "Tutoring Answer Accuracy remain "
            "pending until the three judge "
            "runs are completed."
        ),
        (
            "For manager systems, individual "
            "tool calls are extracted from "
            "tool_results_history when "
            "available, so recovery attempts "
            "are not omitted."
        ),
        (
            "Manager metrics use individual "
            "round traces and do not double "
            "count the duplicated case-level "
            "main trace."
        ),
    ]
)

audit_text = "\n".join(
    audit_lines
) + "\n"

(
    OUTPUT_DIR
    / "evaluation_audit.txt"
).write_text(
    audit_text,
    encoding="utf-8",
)

print(audit_text)

print("Created:")
for path in sorted(
    OUTPUT_DIR.iterdir()
):
    if path.is_file():
        print(" -", path)
