#!/usr/bin/env python3

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "evaluation/final_results_100/unified_evaluation"
)

OUT = BASE / "posthoc_analysis_v1"

TEST_CASES_PATH = Path(
    "evaluation/test_cases/test_cases_100_article.csv"
)

CASE_METRICS_PATH = (
    BASE / "core_metric_case_details_corrected.csv"
)

SEMANTIC_PATH = (
    BASE
    / "final_aggregated_results"
    / "judge_candidate_aggregate_unblinded.csv"
)

OPERATIONAL_PATH = (
    BASE / "per_case_system_metrics.csv"
)

TOOL_CALL_PATH = (
    BASE / "tool_call_details.csv"
)

VALIDATION_PATH = (
    BASE / "validation_details.csv"
)

BOOTSTRAP_SAMPLES = 10000

SYSTEM_MAP = {
    "direct_llm": "Direct LLM",
    "basic_tool_use": "Basic Tool Use",
    "manager_without_recovery":
        "Manager Without Recovery",
    "full_manager": "Full Manager",
}

CATEGORY_DISPLAY = {
    "conceptual_grounded":
        "Conceptual / Grounded",
    "computation_verification":
        "Computation / Verification",
    "visual_step_by_step":
        "Visual Step-by-Step",
    "code_related":
        "Code-Related",
    "recovery_personalization_trace":
        "Recovery / Personalization / Trace",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "yes",
        "true",
        "1",
        "pass",
        "passed",
        "success",
        "successful",
        "valid",
        "complete",
        "completed",
        "ok",
    }


def normalize_error(value):
    if pd.isna(value):
        return "none"

    text = str(value).strip().lower()

    if text in {
        "",
        "none",
        "nan",
        "null",
        "no_error",
    }:
        return "none"

    return text


def tool_set(value):
    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }


def tool_string(values):
    return ";".join(sorted(values))


def selection_error_class(
    expected,
    selected,
):
    missing = expected - selected
    extra = selected - expected

    if not missing and not extra:
        return "exact"

    if not selected:
        return "no_tool_selected"

    if missing and extra:
        return "mixed_missing_and_extra"

    if missing:
        return "missing_expected_tools"

    return "unnecessary_extra_tools"


def wilson_interval(
    successes,
    total,
    z=1.959963984540054,
):
    if total == 0:
        return math.nan, math.nan

    proportion = successes / total
    denominator = 1 + z * z / total

    centre = (
        proportion
        + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            proportion
            * (1 - proportion)
            / total
            + z * z
            / (4 * total * total)
        )
        / denominator
    )

    return (
        max(0, centre - margin),
        min(1, centre + margin),
    )


def stable_seed(text):
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


def bootstrap_mean_ci(
    values,
    label,
):
    array = np.asarray(
        list(values),
        dtype=float,
    )

    array = array[
        ~np.isnan(array)
    ]

    if len(array) == 0:
        return math.nan, math.nan

    if len(array) == 1:
        return float(array[0]), float(array[0])

    rng = np.random.default_rng(
        stable_seed(label)
    )

    indexes = rng.integers(
        0,
        len(array),
        size=(
            BOOTSTRAP_SAMPLES,
            len(array),
        ),
    )

    sample_means = array[indexes].mean(
        axis=1
    )

    return (
        float(
            np.quantile(
                sample_means,
                0.025,
            )
        ),
        float(
            np.quantile(
                sample_means,
                0.975,
            )
        ),
    )


def cohen_kappa(first, second):
    first = list(first)
    second = list(second)

    total = len(first)

    if total == 0:
        return math.nan

    observed = sum(
        left == right
        for left, right
        in zip(first, second)
    ) / total

    first_true = sum(first) / total
    second_true = sum(second) / total

    expected = (
        first_true * second_true
        + (1 - first_true)
        * (1 - second_true)
    )

    if expected == 1:
        return 1.0

    return (
        observed - expected
    ) / (
        1 - expected
    )


def fleiss_kappa(matrix):
    if not matrix:
        return math.nan

    raters = len(matrix[0])

    item_agreements = []
    true_total = 0

    for row in matrix:
        true_count = sum(row)
        false_count = raters - true_count

        true_total += true_count

        item_agreements.append(
            (
                true_count
                * (true_count - 1)
                + false_count
                * (false_count - 1)
            )
            / (
                raters
                * (raters - 1)
            )
        )

    observed = sum(
        item_agreements
    ) / len(item_agreements)

    ratings = len(matrix) * raters
    true_probability = true_total / ratings

    expected = (
        true_probability ** 2
        + (1 - true_probability) ** 2
    )

    if expected == 1:
        return 1.0

    return (
        observed - expected
    ) / (
        1 - expected
    )


def primary_error(row):
    errors = [
        normalize_error(
            row[f"run_{run}_error_type"]
        )
        for run in [1, 2, 3]
    ]

    errors = [
        error
        for error in errors
        if error != "none"
    ]

    if not errors:
        return "none"

    counts = Counter(errors)
    maximum = max(counts.values())

    winners = sorted(
        error
        for error, count
        in counts.items()
        if count == maximum
    )

    if len(winners) == 1:
        return winners[0]

    return (
        "mixed:"
        + "|".join(winners)
    )


def find_column(
    dataframe,
    aliases,
):
    normalized = {
        str(column).strip().lower():
            column
        for column in dataframe.columns
    }

    for alias in aliases:
        key = alias.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# ==================================================
# Load and validate sources
# ==================================================

required_paths = [
    TEST_CASES_PATH,
    CASE_METRICS_PATH,
    SEMANTIC_PATH,
    OPERATIONAL_PATH,
]

for path in required_paths:
    if not path.exists():
        raise SystemExit(
            f"Missing source: {path}"
        )

tests = pd.read_csv(
    TEST_CASES_PATH
)

case_metrics = pd.read_csv(
    CASE_METRICS_PATH
)

semantic = pd.read_csv(
    SEMANTIC_PATH
)

operational = pd.read_csv(
    OPERATIONAL_PATH
)

if len(tests) != 100:
    raise SystemExit(
        "Test dataset does not contain 100 rows."
    )

category_order = (
    tests["category"]
    .drop_duplicates()
    .tolist()
)

tests["category_display"] = (
    tests["category"].map(
        CATEGORY_DISPLAY
    )
)

tests["recovery_test_bool"] = (
    tests["recovery_test"].map(
        parse_bool
    )
)

case_metrics["system_name"] = (
    case_metrics["system"].map(
        SYSTEM_MAP
    )
)

if case_metrics[
    "system_name"
].isna().any():
    raise SystemExit(
        "Unknown system in corrected case metrics."
    )

case_bool_columns = [
    "initial_exact_selection",
    "final_exact_selection",
    "final_plan_execution_success",
    "all_attempted_calls_success",
    "expected_tool_fulfillment",
]

for column in case_bool_columns:
    case_metrics[
        f"{column}_bool"
    ] = case_metrics[column].map(
        parse_bool
    )

case_metrics = case_metrics.merge(
    tests[
        [
            "test_id",
            "category",
            "category_display",
            "question",
            "recovery_test",
            "recovery_test_bool",
        ]
    ],
    on="test_id",
    how="left",
    validate="many_to_one",
)

semantic = semantic.merge(
    tests[
        [
            "test_id",
            "category",
            "category_display",
            "question",
            "expected_tools",
            "recovery_test",
            "recovery_test_bool",
        ]
    ],
    on="test_id",
    how="left",
    validate="many_to_one",
)

operational["system_name"] = (
    operational["system"].map(
        SYSTEM_MAP
    )
)

operational = operational.merge(
    tests[
        [
            "test_id",
            "question",
            "recovery_test",
            "recovery_test_bool",
            "category_display",
        ]
    ],
    on="test_id",
    how="left",
    validate="many_to_one",
)

for column in [
    "recovery_triggered",
    "recovery_successful",
    "strict_trace_complete",
    "operational_completion",
    "answer_generated",
]:
    operational[
        f"{column}_bool"
    ] = operational[column].map(
        parse_bool
    )

operational[
    "final_validation_valid_bool"
] = operational[
    "final_validation_status"
].map(parse_bool)

# ==================================================
# 1. Semantic metrics by category
# ==================================================

semantic_records = []

for (
    category,
    category_display,
    system_name,
), group in semantic.groupby(
    [
        "category",
        "category_display",
        "system_name",
    ],
    sort=False,
):
    total = len(group)

    passed = int(
        group["majority_pass"].sum()
    )

    pass_low, pass_high = (
        wilson_interval(
            passed,
            total,
        )
    )

    end_to_end = int(
        group[
            "end_to_end_success"
        ].sum()
    )

    e2e_low, e2e_high = (
        wilson_interval(
            end_to_end,
            total,
        )
    )

    score_low, score_high = (
        bootstrap_mean_ci(
            group["mean_overall_score"],
            f"{category}|{system_name}",
        )
    )

    record = {
        "category": category,
        "category_display":
            category_display,
        "system_name": system_name,
        "case_count": total,
        "majority_pass_count": passed,
        "tutoring_accuracy_pct":
            100 * passed / total,
        "tutoring_accuracy_ci95_lower_pct":
            100 * pass_low,
        "tutoring_accuracy_ci95_upper_pct":
            100 * pass_high,
        "end_to_end_success_count":
            end_to_end,
        "end_to_end_success_rate_pct":
            100 * end_to_end / total,
        "end_to_end_ci95_lower_pct":
            100 * e2e_low,
        "end_to_end_ci95_upper_pct":
            100 * e2e_high,
        "mean_overall_score":
            group[
                "mean_overall_score"
            ].mean(),
        "sd_overall_score":
            group[
                "mean_overall_score"
            ].std(ddof=1),
        "mean_score_ci95_lower":
            score_low,
        "mean_score_ci95_upper":
            score_high,
        "mean_correctness":
            group[
                "mean_correctness"
            ].mean(),
        "sd_correctness":
            group[
                "mean_correctness"
            ].std(ddof=1),
        "mean_relevance":
            group[
                "mean_relevance"
            ].mean(),
        "mean_completeness":
            group[
                "mean_completeness"
            ].mean(),
        "mean_pedagogical_quality":
            group[
                "mean_pedagogical_quality"
            ].mean(),
        "mean_grounding_consistency":
            group[
                "mean_grounding_consistency"
            ].mean(),
        "mean_judge_score_stdev":
            group[
                "stdev_overall_score"
            ].mean(),
        "unanimous_decision_rate_pct":
            100
            * group[
                "unanimous_decision"
            ].mean(),
        "critical_error_candidate_rate_pct":
            100
            * group[
                "any_critical_error"
            ].mean(),
    }

    semantic_records.append(record)

semantic_summary = pd.DataFrame(
    semantic_records
)

semantic_summary[
    "_category_order"
] = semantic_summary[
    "category"
].map(
    {
        category: index
        for index, category
        in enumerate(category_order)
    }
)

semantic_summary = (
    semantic_summary
    .sort_values(
        [
            "_category_order",
            "system_name",
        ]
    )
    .drop(
        columns=["_category_order"]
    )
)

semantic_summary.to_csv(
    OUT
    / "per_category_semantic_metrics.csv",
    index=False,
)

semantic_wide = (
    semantic_summary.pivot(
        index=[
            "category",
            "category_display",
        ],
        columns="system_name",
        values=[
            "tutoring_accuracy_pct",
            "mean_overall_score",
            "end_to_end_success_rate_pct",
        ],
    )
)

semantic_wide.columns = [
    (
        f"{metric}__"
        f"{system}"
    )
    for metric, system
    in semantic_wide.columns
]

semantic_wide.reset_index().to_csv(
    OUT
    / "per_category_semantic_comparison_wide.csv",
    index=False,
)

# ==================================================
# 2. Corrected tool metrics by category
# ==================================================

tool_records = []

for (
    category,
    category_display,
    system_name,
), group in case_metrics.groupby(
    [
        "category",
        "category_display",
        "system_name",
    ],
    sort=False,
):
    total = len(group)

    record = {
        "category": category,
        "category_display":
            category_display,
        "system_name": system_name,
        "case_count": total,
        "mean_round_count":
            group["round_count"].mean(),
        "sd_round_count":
            group["round_count"].std(ddof=1),
        "mean_attempted_call_count":
            group[
                "attempted_call_count"
            ].mean(),
        "sd_attempted_call_count":
            group[
                "attempted_call_count"
            ].std(ddof=1),
    }

    binary_metrics = {
        "initial_selection_accuracy":
            "initial_exact_selection_bool",
        "final_selection_accuracy":
            "final_exact_selection_bool",
        "final_plan_execution_success":
            "final_plan_execution_success_bool",
        "all_attempted_calls_success":
            "all_attempted_calls_success_bool",
        "expected_tool_fulfillment":
            "expected_tool_fulfillment_bool",
    }

    for metric, column in (
        binary_metrics.items()
    ):
        successes = int(
            group[column].sum()
        )

        low, high = wilson_interval(
            successes,
            total,
        )

        record[
            f"{metric}_count"
        ] = successes

        record[
            f"{metric}_pct"
        ] = 100 * successes / total

        record[
            f"{metric}_ci95_lower_pct"
        ] = 100 * low

        record[
            f"{metric}_ci95_upper_pct"
        ] = 100 * high

    tool_records.append(record)

tool_summary = pd.DataFrame(
    tool_records
)

tool_summary.to_csv(
    OUT
    / "per_category_corrected_tool_metrics.csv",
    index=False,
)

# ==================================================
# 3. Case-level validation, recovery, and trace
# ==================================================

diagnostic_rows = []

manager_operational = operational[
    operational["system_name"].isin(
        [
            "Manager Without Recovery",
            "Full Manager",
        ]
    )
].copy()

for (
    category,
    category_display,
    system_name,
), group in manager_operational.groupby(
    [
        "category",
        "category_display",
        "system_name",
    ],
    sort=False,
):
    total = len(group)
    triggered = int(
        group[
            "recovery_triggered_bool"
        ].sum()
    )

    successful = int(
        (
            group[
                "recovery_triggered_bool"
            ]
            & group[
                "recovery_successful_bool"
            ]
        ).sum()
    )

    diagnostic_rows.append(
        {
            "category": category,
            "category_display":
                category_display,
            "system_name": system_name,
            "case_count": total,
            "final_validation_valid_case_rate_pct":
                100
                * group[
                    "final_validation_valid_bool"
                ].mean(),
            "strict_trace_complete_case_rate_pct":
                100
                * group[
                    "strict_trace_complete_bool"
                ].mean(),
            "operational_completion_rate_pct":
                100
                * group[
                    "operational_completion_bool"
                ].mean(),
            "recovery_triggered_count":
                triggered,
            "recovery_trigger_rate_pct":
                100 * triggered / total,
            "recovery_success_count":
                successful,
            "recovery_success_among_triggered_pct":
                (
                    100
                    * successful
                    / triggered
                    if triggered
                    else math.nan
                ),
        }
    )

diagnostic_summary = pd.DataFrame(
    diagnostic_rows
)

diagnostic_summary.to_csv(
    OUT
    / "per_category_manager_diagnostics.csv",
    index=False,
)

# ==================================================
# 4. Judge stability by category
# ==================================================

stability_records = []

stability_groups = []

for category, group in semantic.groupby(
    "category",
    sort=False,
):
    stability_groups.append(
        (
            category,
            "All Systems",
            group,
        )
    )

for (
    category,
    system_name,
), group in semantic.groupby(
    [
        "category",
        "system_name",
    ],
    sort=False,
):
    stability_groups.append(
        (
            category,
            system_name,
            group,
        )
    )

for (
    category,
    scope_system,
    group,
) in stability_groups:
    run1 = (
        group["run_1_pass"]
        .astype(bool)
        .tolist()
    )

    run2 = (
        group["run_2_pass"]
        .astype(bool)
        .tolist()
    )

    run3 = (
        group["run_3_pass"]
        .astype(bool)
        .tolist()
    )

    matrix = list(
        zip(run1, run2, run3)
    )

    stability_records.append(
        {
            "category": category,
            "category_display":
                CATEGORY_DISPLAY.get(
                    category,
                    category,
                ),
            "scope_system":
                scope_system,
            "candidate_count":
                len(group),
            "unanimous_decision_rate_pct":
                100
                * group[
                    "unanimous_decision"
                ].mean(),
            "mean_candidate_score_stdev":
                group[
                    "stdev_overall_score"
                ].mean(),
            "fleiss_kappa":
                fleiss_kappa(matrix),
            "run1_run2_agreement_pct":
                100
                * np.mean(
                    np.asarray(run1)
                    == np.asarray(run2)
                ),
            "run1_run2_cohen_kappa":
                cohen_kappa(
                    run1,
                    run2,
                ),
            "run1_run3_agreement_pct":
                100
                * np.mean(
                    np.asarray(run1)
                    == np.asarray(run3)
                ),
            "run1_run3_cohen_kappa":
                cohen_kappa(
                    run1,
                    run3,
                ),
            "run2_run3_agreement_pct":
                100
                * np.mean(
                    np.asarray(run2)
                    == np.asarray(run3)
                ),
            "run2_run3_cohen_kappa":
                cohen_kappa(
                    run2,
                    run3,
                ),
        }
    )

pd.DataFrame(
    stability_records
).to_csv(
    OUT
    / "judge_stability_by_category.csv",
    index=False,
)

# ==================================================
# 5. Tool-selection error analysis
# ==================================================

selection = case_metrics.copy()

selection["expected_tool_set"] = (
    selection["expected_tools"].map(
        tool_set
    )
)

selection[
    "initial_selected_tool_set"
] = selection[
    "initial_selected_tools"
].map(tool_set)

selection[
    "final_selected_tool_set"
] = selection[
    "final_selected_tools"
].map(tool_set)

selection[
    "initial_selection_error_class"
] = selection.apply(
    lambda row:
        selection_error_class(
            row["expected_tool_set"],
            row[
                "initial_selected_tool_set"
            ],
        ),
    axis=1,
)

selection[
    "final_selection_error_class"
] = selection.apply(
    lambda row:
        selection_error_class(
            row["expected_tool_set"],
            row[
                "final_selected_tool_set"
            ],
        ),
    axis=1,
)

selection[
    "initial_missing_tools"
] = selection.apply(
    lambda row: tool_string(
        row["expected_tool_set"]
        - row[
            "initial_selected_tool_set"
        ]
    ),
    axis=1,
)

selection[
    "initial_extra_tools"
] = selection.apply(
    lambda row: tool_string(
        row[
            "initial_selected_tool_set"
        ]
        - row["expected_tool_set"]
    ),
    axis=1,
)

selection[
    "final_missing_tools"
] = selection.apply(
    lambda row: tool_string(
        row["expected_tool_set"]
        - row[
            "final_selected_tool_set"
        ]
    ),
    axis=1,
)

selection[
    "final_extra_tools"
] = selection.apply(
    lambda row: tool_string(
        row[
            "final_selected_tool_set"
        ]
        - row["expected_tool_set"]
    ),
    axis=1,
)

semantic_tool = semantic[
    semantic["system_name"].isin(
        selection["system_name"].unique()
    )
][
    [
        "system_name",
        "test_id",
        "majority_pass",
        "mean_overall_score",
        "any_critical_error",
    ]
]

selection = selection.merge(
    semantic_tool,
    on=[
        "system_name",
        "test_id",
    ],
    how="left",
    validate="one_to_one",
)

selection_errors = selection[
    selection[
        "initial_selection_error_class"
    ] != "exact"
].copy()

selection_error_columns = [
    "system_name",
    "test_id",
    "category",
    "category_display",
    "question",
    "expected_tools",
    "initial_selected_tools",
    "initial_selection_error_class",
    "initial_missing_tools",
    "initial_extra_tools",
    "final_selected_tools",
    "final_selection_error_class",
    "final_missing_tools",
    "final_extra_tools",
    "final_plan_execution_success",
    "expected_tool_fulfillment",
    "majority_pass",
    "mean_overall_score",
    "any_critical_error",
]

selection_errors[
    selection_error_columns
].to_csv(
    OUT
    / "tool_selection_error_cases.csv",
    index=False,
)

selection_denominators = (
    selection.groupby(
        [
            "system_name",
            "category",
        ]
    )
    .size()
    .reset_index(
        name="category_case_count"
    )
)

selection_error_summary = (
    selection_errors.groupby(
        [
            "system_name",
            "category",
            "category_display",
            "initial_selection_error_class",
        ]
    )
    .size()
    .reset_index(
        name="error_case_count"
    )
    .merge(
        selection_denominators,
        on=[
            "system_name",
            "category",
        ],
        how="left",
    )
)

selection_error_summary[
    "error_case_rate_pct"
] = (
    100
    * selection_error_summary[
        "error_case_count"
    ]
    / selection_error_summary[
        "category_case_count"
    ]
)

selection_error_summary.to_csv(
    OUT
    / "tool_selection_error_summary.csv",
    index=False,
)

# ==================================================
# 6. Semantic error analysis
# ==================================================

semantic[
    "primary_error_type"
] = semantic.apply(
    primary_error,
    axis=1,
)

semantic[
    "judge_error_occurrence_count"
] = semantic.apply(
    lambda row: sum(
        normalize_error(
            row[
                f"run_{run}_error_type"
            ]
        )
        != "none"
        for run in [1, 2, 3]
    ),
    axis=1,
)

semantic[
    "observed_error_types"
] = semantic.apply(
    lambda row: ";".join(
        sorted(
            {
                normalize_error(
                    row[
                        f"run_{run}_error_type"
                    ]
                )
                for run in [1, 2, 3]
                if normalize_error(
                    row[
                        f"run_{run}_error_type"
                    ]
                )
                != "none"
            }
        )
    ),
    axis=1,
)

semantic[
    "weak_case"
] = (
    ~semantic["majority_pass"]
    | semantic["any_critical_error"]
    | (
        semantic[
            "mean_overall_score"
        ] < 4.0
    )
)

case_metric_join = case_metrics[
    [
        "system_name",
        "test_id",
        "initial_selected_tools",
        "final_selected_tools",
        "initial_exact_selection",
        "final_exact_selection",
        "final_plan_execution_success",
        "expected_tool_fulfillment",
    ]
]

operational_join = operational[
    [
        "system_name",
        "test_id",
        "final_validation_status",
        "recovery_triggered",
        "recovery_successful",
        "strict_trace_complete",
    ]
]

semantic_errors = semantic.merge(
    case_metric_join,
    on=[
        "system_name",
        "test_id",
    ],
    how="left",
    validate="one_to_one",
).merge(
    operational_join,
    on=[
        "system_name",
        "test_id",
    ],
    how="left",
    validate="one_to_one",
)

weak_cases = semantic_errors[
    semantic_errors["weak_case"]
].copy()

weak_case_columns = [
    "system_name",
    "test_id",
    "category",
    "category_display",
    "recovery_test",
    "question",
    "expected_tools",
    "majority_pass",
    "pass_count",
    "mean_correctness",
    "mean_relevance",
    "mean_completeness",
    "mean_pedagogical_quality",
    "mean_grounding_consistency",
    "mean_overall_score",
    "stdev_overall_score",
    "any_critical_error",
    "critical_error_count",
    "primary_error_type",
    "observed_error_types",
    "judge_error_occurrence_count",
    "initial_selected_tools",
    "final_selected_tools",
    "initial_exact_selection",
    "final_exact_selection",
    "final_plan_execution_success",
    "expected_tool_fulfillment",
    "final_validation_status",
    "recovery_triggered",
    "recovery_successful",
]

weak_cases[
    weak_case_columns
].to_csv(
    OUT
    / "semantic_weak_and_error_cases.csv",
    index=False,
)

candidate_error_summary = (
    semantic[
        semantic[
            "primary_error_type"
        ] != "none"
    ]
    .groupby(
        [
            "system_name",
            "category",
            "category_display",
            "primary_error_type",
        ]
    )
    .size()
    .reset_index(
        name="candidate_count"
    )
)

candidate_error_summary.to_csv(
    OUT
    / "candidate_error_type_by_system_category.csv",
    index=False,
)

judge_error_rows = []

for _, row in semantic.iterrows():
    for run in [1, 2, 3]:
        error_type = normalize_error(
            row[
                f"run_{run}_error_type"
            ]
        )

        if error_type == "none":
            continue

        judge_error_rows.append(
            {
                "system_name":
                    row["system_name"],
                "test_id":
                    row["test_id"],
                "category":
                    row["category"],
                "category_display":
                    row[
                        "category_display"
                    ],
                "run_number": run,
                "error_type":
                    error_type,
            }
        )

judge_error_occurrences = pd.DataFrame(
    judge_error_rows
)

if len(judge_error_occurrences):
    (
        judge_error_occurrences
        .groupby(
            [
                "system_name",
                "category",
                "category_display",
                "error_type",
            ]
        )
        .size()
        .reset_index(
            name="judge_run_occurrences"
        )
        .to_csv(
            OUT
            / "judge_error_occurrences_by_system_category.csv",
            index=False,
        )
    )

# ==================================================
# 7. Full Manager recovery analysis
# ==================================================

full_operational = operational[
    operational[
        "system_name"
    ] == "Full Manager"
].copy()

full_case_metrics = case_metrics[
    case_metrics[
        "system_name"
    ] == "Full Manager"
][
    [
        "test_id",
        "final_plan_execution_success",
        "expected_tool_fulfillment",
        "round_count",
        "attempted_call_count",
    ]
]

full_semantic = semantic[
    semantic[
        "system_name"
    ] == "Full Manager"
][
    [
        "test_id",
        "majority_pass",
        "mean_overall_score",
        "primary_error_type",
        "any_critical_error",
    ]
]

full_recovery = full_operational.merge(
    full_case_metrics,
    on="test_id",
    how="left",
    validate="one_to_one",
).merge(
    full_semantic,
    on="test_id",
    how="left",
    validate="one_to_one",
)

recovery_detail_columns = [
    "test_id",
    "category",
    "category_display",
    "recovery_test",
    "question",
    "recovery_triggered",
    "recovery_successful",
    "final_validation_status",
    "strict_trace_complete",
    "final_plan_execution_success",
    "expected_tool_fulfillment",
    "round_count",
    "attempted_call_count",
    "majority_pass",
    "mean_overall_score",
    "primary_error_type",
    "any_critical_error",
]

full_recovery[
    recovery_detail_columns
].to_csv(
    OUT
    / "full_manager_recovery_case_details.csv",
    index=False,
)

recovery_summary_rows = []


def append_recovery_summary(
    label_type,
    label_value,
    group,
):
    total = len(group)

    triggered = int(
        group[
            "recovery_triggered_bool"
        ].sum()
    )

    successful = int(
        (
            group[
                "recovery_triggered_bool"
            ]
            & group[
                "recovery_successful_bool"
            ]
        ).sum()
    )

    recovery_summary_rows.append(
        {
            "slice_type": label_type,
            "slice_value": label_value,
            "case_count": total,
            "recovery_triggered_count":
                triggered,
            "recovery_trigger_rate_pct":
                100 * triggered / total,
            "recovery_success_count":
                successful,
            "recovery_success_among_triggered_pct":
                (
                    100
                    * successful
                    / triggered
                    if triggered
                    else math.nan
                ),
            "final_plan_execution_success_pct":
                100
                * group[
                    "final_plan_execution_success"
                ].map(
                    parse_bool
                ).mean(),
            "expected_tool_fulfillment_pct":
                100
                * group[
                    "expected_tool_fulfillment"
                ].map(
                    parse_bool
                ).mean(),
            "tutoring_accuracy_pct":
                100
                * group[
                    "majority_pass"
                ].mean(),
            "mean_overall_score":
                group[
                    "mean_overall_score"
                ].mean(),
        }
    )


append_recovery_summary(
    "overall",
    "All Full Manager cases",
    full_recovery,
)

for recovery_value, group in (
    full_recovery.groupby(
        "recovery_test",
        sort=False,
    )
):
    append_recovery_summary(
        "recovery_test",
        str(recovery_value),
        group,
    )

for category, group in (
    full_recovery.groupby(
        "category",
        sort=False,
    )
):
    append_recovery_summary(
        "category",
        category,
        group,
    )

pd.DataFrame(
    recovery_summary_rows
).to_csv(
    OUT
    / "full_manager_recovery_slice_summary.csv",
    index=False,
)

combined_category_errors = weak_cases[
    weak_cases["category"]
    == "recovery_personalization_trace"
]

combined_category_errors[
    weak_case_columns
].to_csv(
    OUT
    / "recovery_personalization_trace_error_cases.csv",
    index=False,
)

# ==================================================
# 8. Optional attempt-level metrics
# ==================================================

attempt_metric_rows = []
attempt_notes = []


def add_attempt_metric(
    path,
    metric_name,
    success_aliases,
):
    if not path.exists():
        attempt_notes.append(
            f"{metric_name}: source file missing."
        )
        return

    dataframe = pd.read_csv(path)

    system_column = find_column(
        dataframe,
        [
            "system",
            "system_name",
            "condition",
        ],
    )

    test_column = find_column(
        dataframe,
        [
            "test_id",
            "case_id",
        ],
    )

    success_column = find_column(
        dataframe,
        success_aliases,
    )

    if (
        system_column is None
        or test_column is None
        or success_column is None
    ):
        attempt_notes.append(
            f"{metric_name}: columns not detected; "
            f"available={list(dataframe.columns)}"
        )
        return

    subset = dataframe[
        [
            system_column,
            test_column,
            success_column,
        ]
    ].copy()

    subset.columns = [
        "system",
        "test_id",
        "success",
    ]

    subset["system_name"] = (
        subset["system"].map(
            SYSTEM_MAP
        )
    )

    subset["success_bool"] = (
        subset["success"].map(
            parse_bool
        )
    )

    subset = subset.merge(
        tests[
            [
                "test_id",
                "category",
                "category_display",
            ]
        ],
        on="test_id",
        how="left",
        validate="many_to_one",
    )

    for (
        category,
        category_display,
        system_name,
    ), group in subset.groupby(
        [
            "category",
            "category_display",
            "system_name",
        ],
        dropna=False,
        sort=False,
    ):
        if pd.isna(system_name):
            continue

        total = len(group)
        successes = int(
            group[
                "success_bool"
            ].sum()
        )

        low, high = wilson_interval(
            successes,
            total,
        )

        attempt_metric_rows.append(
            {
                "metric_name":
                    metric_name,
                "category":
                    category,
                "category_display":
                    category_display,
                "system_name":
                    system_name,
                "event_count": total,
                "success_count":
                    successes,
                "success_rate_pct":
                    100
                    * successes
                    / total,
                "ci95_lower_pct":
                    100 * low,
                "ci95_upper_pct":
                    100 * high,
                "detected_success_column":
                    success_column,
            }
        )

    attempt_notes.append(
        f"{metric_name}: included using "
        f"column '{success_column}'."
    )


add_attempt_metric(
    TOOL_CALL_PATH,
    "Tool Call Success Rate",
    [
        "tool_call_success",
        "call_success",
        "success",
        "successful",
        "status",
    ],
)

add_attempt_metric(
    VALIDATION_PATH,
    "Validation Pass Rate",
    [
        "validation_pass",
        "passed",
        "pass",
        "is_valid",
        "valid",
        "validation_status",
        "status",
    ],
)

pd.DataFrame(
    attempt_metric_rows
).to_csv(
    OUT
    / "per_category_attempt_level_metrics.csv",
    index=False,
)

# ==================================================
# 9. Text summary and audit
# ==================================================

summary_lines = [
    "POST-HOC PER-CATEGORY AND ERROR ANALYSIS",
    "=" * 78,
    "",
    "Dataset structure",
    "- Test cases: 100",
    "- Official categories: 5",
]

for category in category_order:
    count = int(
        (
            tests["category"]
            == category
        ).sum()
    )

    summary_lines.append(
        f"  - {category}: {count}"
    )

summary_lines.extend(
    [
        "",
        "Best semantic system by category",
    ]
)

for category in category_order:
    subset = semantic_summary[
        semantic_summary[
            "category"
        ] == category
    ].sort_values(
        [
            "tutoring_accuracy_pct",
            "mean_overall_score",
        ],
        ascending=False,
    )

    best = subset.iloc[0]

    summary_lines.append(
        (
            f"- {category}: "
            f"{best['system_name']} "
            f"({best['tutoring_accuracy_pct']:.2f}% accuracy, "
            f"mean score "
            f"{best['mean_overall_score']:.3f})"
        )
    )

summary_lines.extend(
    [
        "",
        "Full Manager versus Manager Without Recovery by category",
    ]
)

full_category = semantic_summary[
    semantic_summary[
        "system_name"
    ] == "Full Manager"
].set_index("category")

no_recovery_category = semantic_summary[
    semantic_summary[
        "system_name"
    ] == "Manager Without Recovery"
].set_index("category")

for category in category_order:
    difference = (
        full_category.loc[
            category,
            "tutoring_accuracy_pct",
        ]
        - no_recovery_category.loc[
            category,
            "tutoring_accuracy_pct",
        ]
    )

    summary_lines.append(
        (
            f"- {category}: "
            f"{difference:+.2f} percentage points"
        )
    )

summary_lines.extend(
    [
        "",
        "Judge stability",
        "- Three repeated GPT-4.1 judge runs were analyzed by category.",
        "- These repeated runs measure judge stability, not system execution variance.",
        "",
        "Category limitation",
        (
            "- The dataset does not provide separate official categories "
            "for personalization and recovery."
        ),
        (
            "- Both are combined with trace cases in "
            "'recovery_personalization_trace'."
        ),
        (
            "- Designated recovery cases are additionally isolated through "
            "the recovery_test field."
        ),
        "",
        "Attempt-level metric detection",
    ]
)

summary_lines.extend(
    f"- {note}"
    for note in attempt_notes
)

summary_lines.extend(
    [
        "",
        "Scientific scope",
        (
            "- This post-hoc analysis validates a controlled backend "
            "prototype."
        ),
        (
            "- It does not measure real classroom learning gains, "
            "long-term retention, student engagement, or teacher outcomes."
        ),
    ]
)

summary_path = (
    OUT / "posthoc_analysis_summary.txt"
)

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

source_paths = [
    TEST_CASES_PATH,
    CASE_METRICS_PATH,
    SEMANTIC_PATH,
    OPERATIONAL_PATH,
]

if TOOL_CALL_PATH.exists():
    source_paths.append(
        TOOL_CALL_PATH
    )

if VALIDATION_PATH.exists():
    source_paths.append(
        VALIDATION_PATH
    )

manifest_lines = [
    "POST-HOC ANALYSIS SOURCE MANIFEST",
    "=" * 78,
]

for path in source_paths:
    manifest_lines.append(
        f"{sha256(path)}  {path}"
    )

(
    OUT / "source_manifest.sha256"
).write_text(
    "\n".join(manifest_lines) + "\n",
    encoding="utf-8",
)

audit_lines = [
    "POST-HOC ANALYSIS AUDIT",
    "=" * 78,
    "",
    "- Official locked evaluation package modified: No",
    "- New experiment executed: No",
    "- System answers regenerated: No",
    "- Judge calls repeated: No",
    "- Existing three-run judge results reused: Yes",
    "- Analysis type: exploratory post-hoc category and error analysis",
    "- Test cases: 100",
    "- Systems: 4",
    "- Semantic candidates: 400",
    "- Judge decisions analyzed: 1,200",
    "- Official categories: 5",
    "- Cases per category: 20",
    "- Confidence intervals for proportions: Wilson 95%",
    "- Confidence intervals for mean score: deterministic bootstrap 95%",
    "- Bootstrap samples per group: 10,000",
    "- Tool selection/execution source: corrected per-case metrics",
    (
        "- Case-level validation diagnostic is not the same as the "
        "official attempt-level Validation Pass Rate."
    ),
    (
        "- Personalization cannot be isolated as a standalone official "
        "category from the current dataset schema."
    ),
    (
        "- Recovery-specific analysis uses recovery_test and actual "
        "Full Manager recovery evidence."
    ),
    (
        "- No per-category significance claims should be made without "
        "additional correction for multiple exploratory comparisons."
    ),
    "",
    "Generated outputs:",
]

for path in sorted(
    OUT.glob("*")
):
    audit_lines.append(
        f"- {path.name}"
    )

(
    OUT / "posthoc_analysis_audit.txt"
).write_text(
    "\n".join(audit_lines) + "\n",
    encoding="utf-8",
)

print("=" * 90)
print("POST-HOC ANALYSIS COMPLETE")
print("=" * 90)
print("Output directory:")
print(OUT)
print()
print("Semantic accuracy by category:")
print(
    semantic_summary[
        [
            "category",
            "system_name",
            "case_count",
            "tutoring_accuracy_pct",
            "mean_overall_score",
            "sd_overall_score",
            "mean_score_ci95_lower",
            "mean_score_ci95_upper",
        ]
    ].to_string(index=False)
)
print()
print("Summary:")
print(
    summary_path.read_text(
        encoding="utf-8"
    )
)
