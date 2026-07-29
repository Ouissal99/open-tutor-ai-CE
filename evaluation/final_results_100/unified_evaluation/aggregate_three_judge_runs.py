#!/usr/bin/env python3

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


BASE = Path(
    "evaluation/final_results_100/unified_evaluation"
)
RUN_DIR = BASE / "judge_runs"
OUTPUT_DIR = BASE / "final_aggregated_results"

MAPPING_PATH = (
    BASE / "judge_identity_mapping_secret.csv"
)
PER_CASE_METRICS_PATH = (
    BASE / "per_case_system_metrics.csv"
)
CORRECTED_METRICS_PATH = (
    BASE / "system_metrics_summary_corrected.csv"
)

RUN_PATHS = {
    1: RUN_DIR / "github_gpt41_run_1.jsonl",
    2: RUN_DIR / "github_gpt41_run_2.jsonl",
    3: RUN_DIR / "github_gpt41_run_3.jsonl",
}

SCORE_ALIASES = {
    "correctness": [
        "correctness",
        "correctness_score",
    ],
    "relevance": [
        "relevance",
        "relevance_score",
    ],
    "completeness": [
        "completeness",
        "completeness_score",
    ],
    "pedagogical_quality": [
        "pedagogical_quality",
        "pedagogical_quality_score",
        "pedagogy",
        "pedagogy_score",
    ],
    "grounding_consistency": [
        "grounding_consistency",
        "grounding_consistency_score",
        "grounding",
        "grounding_score",
    ],
    "overall_score": [
        "overall_score",
        "overall",
        "mean_score",
    ],
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    rows = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8",
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"Invalid JSON in {path}, "
                f"line {line_number}: {exc}"
            )

    return rows


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "1",
            "pass",
            "passed",
            "ok",
            "success",
            "successful",
            "complete",
            "completed",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "0",
            "fail",
            "failed",
            "error",
            "incomplete",
        }:
            return False

    return False


def normalize_column(name: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(name).strip().lower(),
    ).strip("_")


def system_key(value) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def normalize_test_id(value) -> str:
    text = str(value).strip().upper()

    match = re.search(r"(\d+)", text)

    if not match:
        return text

    return f"T{int(match.group(1)):03d}"


def find_column(
    dataframe: pd.DataFrame,
    exact_names: list[str],
):
    normalized = {
        normalize_column(column): column
        for column in dataframe.columns
    }

    for name in exact_names:
        normalized_name = normalize_column(name)

        if normalized_name in normalized:
            return normalized[normalized_name]

    return None


def extract_score(row: dict, metric: str):
    aliases = SCORE_ALIASES[metric]

    containers = [
        row,
        row.get("scores", {}),
        row.get("rubric_scores", {}),
        row.get("evaluation", {}),
    ]

    for container in containers:
        if not isinstance(container, dict):
            continue

        normalized = {
            normalize_column(key): value
            for key, value in container.items()
        }

        for alias in aliases:
            key = normalize_column(alias)

            if key not in normalized:
                continue

            value = normalized[key]

            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
):
    if total == 0:
        return (math.nan, math.nan)

    proportion = successes / total
    denominator = 1 + (z * z / total)

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
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def cohen_kappa(
    first: list[bool],
    second: list[bool],
):
    if len(first) != len(second):
        raise ValueError(
            "Judge lists have different lengths."
        )

    total = len(first)

    if total == 0:
        return math.nan

    observed = sum(
        left == right
        for left, right in zip(
            first,
            second,
        )
    ) / total

    first_true = (
        sum(first) / total
    )
    second_true = (
        sum(second) / total
    )

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


def fleiss_kappa(
    judgments: list[list[bool]],
):
    if not judgments:
        return math.nan

    raters = len(judgments[0])

    if raters < 2:
        return math.nan

    item_agreements = []
    total_true = 0

    for item in judgments:
        true_count = sum(item)
        false_count = raters - true_count

        total_true += true_count

        item_agreement = (
            true_count * (true_count - 1)
            + false_count * (false_count - 1)
        ) / (
            raters * (raters - 1)
        )

        item_agreements.append(
            item_agreement
        )

    observed = mean(item_agreements)

    total_ratings = (
        len(judgments) * raters
    )

    true_probability = (
        total_true / total_ratings
    )
    false_probability = (
        1 - true_probability
    )

    expected = (
        true_probability ** 2
        + false_probability ** 2
    )

    if expected == 1:
        return 1.0

    return (
        observed - expected
    ) / (
        1 - expected
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# --------------------------------------------------
# 1. Read and validate all three locked judge runs
# --------------------------------------------------

run_indexes = {}
candidate_sets = {}

for run_number, path in RUN_PATHS.items():
    rows = read_jsonl(path)

    candidate_ids = [
        row.get("candidate_id")
        for row in rows
    ]

    missing_ids = [
        index
        for index, candidate_id
        in enumerate(candidate_ids)
        if not candidate_id
    ]

    if missing_ids:
        raise SystemExit(
            f"Run {run_number} contains rows "
            f"without candidate_id."
        )

    duplicates = [
        candidate_id
        for candidate_id, count
        in Counter(candidate_ids).items()
        if count > 1
    ]

    if duplicates:
        raise SystemExit(
            f"Run {run_number} duplicates: "
            f"{duplicates[:20]}"
        )

    if len(rows) != 400:
        raise SystemExit(
            f"Run {run_number} has "
            f"{len(rows)} rows instead of 400."
        )

    run_indexes[run_number] = {
        row["candidate_id"]: row
        for row in rows
    }

    candidate_sets[run_number] = set(
        candidate_ids
    )

reference_candidates = candidate_sets[1]

for run_number in [2, 3]:
    if (
        candidate_sets[run_number]
        != reference_candidates
    ):
        missing = (
            reference_candidates
            - candidate_sets[run_number]
        )

        unexpected = (
            candidate_sets[run_number]
            - reference_candidates
        )

        raise SystemExit(
            f"Run {run_number} candidate mismatch. "
            f"Missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

# --------------------------------------------------
# 2. Aggregate three judgments per candidate
# --------------------------------------------------

candidate_records = []
error_records = []

for candidate_id in sorted(
    reference_candidates
):
    match = re.fullmatch(
        r"(T\d+)-([A-Za-z0-9]+)",
        candidate_id,
    )

    if not match:
        raise SystemExit(
            f"Unexpected candidate ID: "
            f"{candidate_id}"
        )

    test_id = normalize_test_id(
        match.group(1)
    )
    anonymous_label = (
        match.group(2).upper()
    )

    record = {
        "candidate_id": candidate_id,
        "test_id": test_id,
        "anonymous_label": anonymous_label,
    }

    passes = []
    critical_errors = []

    score_values = {
        metric: []
        for metric in SCORE_ALIASES
    }

    for run_number in [1, 2, 3]:
        row = run_indexes[
            run_number
        ][candidate_id]

        passed = parse_bool(
            row.get("pass_by_rubric")
        )

        critical_error = parse_bool(
            row.get("critical_error")
        )

        error_type = (
            row.get("error_type")
            or ""
        )

        passes.append(passed)
        critical_errors.append(
            critical_error
        )

        record[
            f"run_{run_number}_pass"
        ] = passed

        record[
            f"run_{run_number}_critical_error"
        ] = critical_error

        record[
            f"run_{run_number}_error_type"
        ] = error_type

        if error_type:
            error_records.append(
                {
                    "candidate_id": (
                        candidate_id
                    ),
                    "test_id": test_id,
                    "anonymous_label": (
                        anonymous_label
                    ),
                    "run_number": (
                        run_number
                    ),
                    "error_type": error_type,
                }
            )

        for metric in SCORE_ALIASES:
            value = extract_score(
                row,
                metric,
            )

            record[
                f"run_{run_number}_{metric}"
            ] = value

            if value is not None:
                score_values[
                    metric
                ].append(value)

    pass_count = sum(passes)

    record["pass_count"] = pass_count
    record["majority_pass"] = (
        pass_count >= 2
    )
    record["unanimous_decision"] = (
        len(set(passes)) == 1
    )
    record["critical_error_count"] = (
        sum(critical_errors)
    )
    record["any_critical_error"] = (
        any(critical_errors)
    )

    for metric, values in (
        score_values.items()
    ):
        if len(values) != 3:
            raise SystemExit(
                f"{candidate_id} has "
                f"{len(values)} values for "
                f"{metric}, expected 3."
            )

        record[
            f"mean_{metric}"
        ] = mean(values)

        record[
            f"stdev_{metric}"
        ] = (
            stdev(values)
            if len(values) > 1
            else 0.0
        )

    candidate_records.append(record)

candidate_df = pd.DataFrame(
    candidate_records
)

blind_path = (
    OUTPUT_DIR
    / "judge_candidate_aggregate_blind.csv"
)

candidate_df.to_csv(
    blind_path,
    index=False,
)

# --------------------------------------------------
# 3. Reveal system identities
# --------------------------------------------------

if not MAPPING_PATH.exists():
    raise SystemExit(
        f"Missing mapping file: "
        f"{MAPPING_PATH}"
    )

mapping_df = pd.read_csv(
    MAPPING_PATH
)

candidate_column = find_column(
    mapping_df,
    [
        "candidate_id",
        "anonymous_candidate_id",
        "blind_candidate_id",
    ],
)

label_column = find_column(
    mapping_df,
    [
        "anonymous_system",
        "anonymous_label",
        "blind_label",
        "candidate_label",
        "label",
    ],
)

system_column = find_column(
    mapping_df,
    [
        "actual_system_display",
        "actual_system",
        "system_name",
        "real_system",
        "system",
        "condition",
        "method",
        "baseline_name",
        "system_key",
    ],
)

if system_column is None:
    candidates = [
        column
        for column in mapping_df.columns
        if "system" in normalize_column(
            column
        )
        and "anonymous" not in normalize_column(
            column
        )
        and "label" not in normalize_column(
            column
        )
    ]

    if len(candidates) == 1:
        system_column = candidates[0]

if system_column is None:
    raise SystemExit(
        "Could not identify the real-system "
        "column in the secret mapping.\n"
        f"Mapping columns: "
        f"{list(mapping_df.columns)}"
    )

if candidate_column is not None:
    reveal_df = mapping_df[
        [
            candidate_column,
            system_column,
        ]
    ].copy()

    reveal_df.columns = [
        "candidate_id",
        "system_name",
    ]

    merged_df = candidate_df.merge(
        reveal_df,
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )

elif label_column is not None:
    reveal_df = mapping_df[
        [
            label_column,
            system_column,
        ]
    ].copy()

    reveal_df.columns = [
        "anonymous_label",
        "system_name",
    ]

    reveal_df[
        "anonymous_label"
    ] = (
        reveal_df["anonymous_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    reveal_df = reveal_df.drop_duplicates()

    merged_df = candidate_df.merge(
        reveal_df,
        on="anonymous_label",
        how="left",
        validate="many_to_one",
    )

else:
    raise SystemExit(
        "Could not identify candidate_id "
        "or anonymous_label in mapping.\n"
        f"Mapping columns: "
        f"{list(mapping_df.columns)}"
    )

if merged_df[
    "system_name"
].isna().any():
    missing = merged_df.loc[
        merged_df["system_name"].isna(),
        [
            "candidate_id",
            "anonymous_label",
        ],
    ]

    raise SystemExit(
        "Some candidates could not be "
        "unblinded:\n"
        f"{missing.head(20)}"
    )

merged_df[
    "system_name"
] = (
    merged_df["system_name"]
    .astype(str)
    .str.strip()
)

merged_df[
    "_system_key"
] = merged_df[
    "system_name"
].map(system_key)

# --------------------------------------------------
# 4. Derive operational completion and E2E
# --------------------------------------------------

operational_available = False
operational_note = ""

if PER_CASE_METRICS_PATH.exists():
    operational_df = pd.read_csv(
        PER_CASE_METRICS_PATH
    )

    op_test_column = find_column(
        operational_df,
        [
            "test_id",
            "case_id",
            "id",
        ],
    )

    op_system_column = find_column(
        operational_df,
        [
            "system_name",
            "system",
            "condition",
            "method",
            "baseline",
        ],
    )

    op_completion_column = find_column(
        operational_df,
        [
            "operational_answer_completion",
            "operational_completion",
            "operationally_complete",
            "workflow_completed",
            "workflow_complete",
            "answer_completion",
        ],
    )

    if op_completion_column is None:
        for column in operational_df.columns:
            normalized = normalize_column(
                column
            )

            if (
                "operational" in normalized
                and (
                    "complete" in normalized
                    or "completion" in normalized
                    or "success" in normalized
                )
            ):
                op_completion_column = (
                    column
                )
                break

    if (
        op_test_column is not None
        and op_system_column is not None
        and op_completion_column is not None
    ):
        op_join = operational_df[
            [
                op_test_column,
                op_system_column,
                op_completion_column,
            ]
        ].copy()

        op_join.columns = [
            "test_id",
            "operational_system",
            "operational_complete",
        ]

        op_join[
            "test_id"
        ] = op_join[
            "test_id"
        ].map(normalize_test_id)

        op_join[
            "_system_key"
        ] = op_join[
            "operational_system"
        ].map(system_key)

        op_join[
            "operational_complete"
        ] = op_join[
            "operational_complete"
        ].map(parse_bool)

        op_join = (
            op_join
            .groupby(
                [
                    "test_id",
                    "_system_key",
                ],
                as_index=False,
            )[
                "operational_complete"
            ]
            .max()
        )

        merged_df = merged_df.merge(
            op_join,
            on=[
                "test_id",
                "_system_key",
            ],
            how="left",
            validate="many_to_one",
        )

        missing_operational = (
            merged_df[
                "operational_complete"
            ].isna().sum()
        )

        if missing_operational == 0:
            operational_available = True

            merged_df[
                "operational_complete"
            ] = merged_df[
                "operational_complete"
            ].astype(bool)

            merged_df[
                "end_to_end_success"
            ] = (
                merged_df[
                    "operational_complete"
                ]
                & merged_df[
                    "majority_pass"
                ]
            )

            operational_note = (
                "Operational completion was "
                "joined from "
                "per_case_system_metrics.csv."
            )
        else:
            operational_note = (
                f"Operational completion could "
                f"not be matched for "
                f"{missing_operational} candidates."
            )
    else:
        operational_note = (
            "Operational completion columns "
            "could not be detected in "
            "per_case_system_metrics.csv. "
            f"Columns were: "
            f"{list(operational_df.columns)}"
        )
else:
    operational_note = (
        "per_case_system_metrics.csv "
        "was not found."
    )

unblinded_path = (
    OUTPUT_DIR
    / "judge_candidate_aggregate_unblinded.csv"
)

merged_df.drop(
    columns=["_system_key"],
    errors="ignore",
).to_csv(
    unblinded_path,
    index=False,
)

# --------------------------------------------------
# 5. Per-system semantic metrics
# --------------------------------------------------

summary_records = []

for system_name, group in (
    merged_df.groupby(
        "system_name",
        sort=True,
    )
):
    total = len(group)
    pass_count = int(
        group["majority_pass"].sum()
    )

    lower, upper = wilson_interval(
        pass_count,
        total,
    )

    summary = {
        "system_name": system_name,
        "candidate_count": total,
        "majority_pass_count": (
            pass_count
        ),
        "tutoring_answer_accuracy_pct": (
            100 * pass_count / total
        ),
        "tutoring_accuracy_ci95_lower_pct": (
            100 * lower
        ),
        "tutoring_accuracy_ci95_upper_pct": (
            100 * upper
        ),
        "unanimous_judge_decisions": int(
            group[
                "unanimous_decision"
            ].sum()
        ),
        "unanimous_judge_rate_pct": (
            100
            * group[
                "unanimous_decision"
            ].mean()
        ),
        "candidates_with_any_critical_error": int(
            group[
                "any_critical_error"
            ].sum()
        ),
        "any_critical_error_rate_pct": (
            100
            * group[
                "any_critical_error"
            ].mean()
        ),
    }

    for metric in SCORE_ALIASES:
        summary[
            f"mean_{metric}"
        ] = group[
            f"mean_{metric}"
        ].mean()

    if operational_available:
        e2e_count = int(
            group[
                "end_to_end_success"
            ].sum()
        )

        summary[
            "operational_completion_count"
        ] = int(
            group[
                "operational_complete"
            ].sum()
        )

        summary[
            "operational_completion_rate_pct"
        ] = (
            100
            * group[
                "operational_complete"
            ].mean()
        )

        summary[
            "end_to_end_success_count"
        ] = e2e_count

        summary[
            "end_to_end_success_rate_pct"
        ] = (
            100 * e2e_count / total
        )

    summary_records.append(summary)

system_summary_df = pd.DataFrame(
    summary_records
)

system_summary_path = (
    OUTPUT_DIR
    / "judge_system_summary.csv"
)

system_summary_df.to_csv(
    system_summary_path,
    index=False,
)

# --------------------------------------------------
# 6. Judge agreement
# --------------------------------------------------

agreement_records = []

pairs = [
    (1, 2),
    (1, 3),
    (2, 3),
]

for first_run, second_run in pairs:
    first = (
        merged_df[
            f"run_{first_run}_pass"
        ]
        .astype(bool)
        .tolist()
    )

    second = (
        merged_df[
            f"run_{second_run}_pass"
        ]
        .astype(bool)
        .tolist()
    )

    percent_agreement = (
        100
        * sum(
            left == right
            for left, right
            in zip(first, second)
        )
        / len(first)
    )

    agreement_records.append(
        {
            "scope": "all_systems",
            "judge_pair": (
                f"run_{first_run}_vs_"
                f"run_{second_run}"
            ),
            "candidate_count": len(first),
            "percent_agreement": (
                percent_agreement
            ),
            "cohen_kappa": cohen_kappa(
                first,
                second,
            ),
        }
    )

for system_name, group in (
    merged_df.groupby(
        "system_name",
        sort=True,
    )
):
    for first_run, second_run in pairs:
        first = (
            group[
                f"run_{first_run}_pass"
            ]
            .astype(bool)
            .tolist()
        )

        second = (
            group[
                f"run_{second_run}_pass"
            ]
            .astype(bool)
            .tolist()
        )

        agreement_records.append(
            {
                "scope": system_name,
                "judge_pair": (
                    f"run_{first_run}_vs_"
                    f"run_{second_run}"
                ),
                "candidate_count": (
                    len(first)
                ),
                "percent_agreement": (
                    100
                    * sum(
                        left == right
                        for left, right
                        in zip(first, second)
                    )
                    / len(first)
                ),
                "cohen_kappa": (
                    cohen_kappa(
                        first,
                        second,
                    )
                ),
            }
        )

judgment_matrix = (
    merged_df[
        [
            "run_1_pass",
            "run_2_pass",
            "run_3_pass",
        ]
    ]
    .astype(bool)
    .values
    .tolist()
)

overall_fleiss = fleiss_kappa(
    judgment_matrix
)

agreement_df = pd.DataFrame(
    agreement_records
)

agreement_df[
    "overall_fleiss_kappa"
] = math.nan

agreement_df.loc[
    agreement_df.index[0],
    "overall_fleiss_kappa",
] = overall_fleiss

agreement_path = (
    OUTPUT_DIR
    / "judge_inter_run_agreement.csv"
)

agreement_df.to_csv(
    agreement_path,
    index=False,
)

# --------------------------------------------------
# 7. Error-type analysis
# --------------------------------------------------

if error_records:
    error_df = pd.DataFrame(
        error_records
    )

    candidate_system_map = (
        merged_df[
            [
                "candidate_id",
                "system_name",
            ]
        ]
        .drop_duplicates(
            subset=["candidate_id"]
        )
    )

    error_df = error_df.merge(
        candidate_system_map,
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )

    if error_df["system_name"].isna().any():
        missing_count = int(
            error_df["system_name"].isna().sum()
        )

        raise SystemExit(
            f"Could not map {missing_count} "
            f"error records to systems."
        )

    error_summary_df = (
        error_df
        .groupby(
            [
                "system_name",
                "error_type",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="judge_run_occurrences"
        )
        .sort_values(
            [
                "system_name",
                "judge_run_occurrences",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )
else:
    error_summary_df = pd.DataFrame(
        columns=[
            "system_name",
            "error_type",
            "judge_run_occurrences",
        ]
    )

error_summary_path = (
    OUTPUT_DIR
    / "judge_error_type_summary.csv"
)

error_summary_df.to_csv(
    error_summary_path,
    index=False,
)

# --------------------------------------------------
# 8. Optionally merge with corrected tool metrics
# --------------------------------------------------

combined_metrics_note = (
    "Corrected metric summary was not merged."
)

if CORRECTED_METRICS_PATH.exists():
    corrected_df = pd.read_csv(
        CORRECTED_METRICS_PATH
    )

    corrected_system_column = (
        find_column(
            corrected_df,
            [
                "system_name",
                "system",
                "condition",
                "method",
                "baseline",
            ],
        )
    )

    if (
        corrected_system_column is not None
        and len(corrected_df) <= 10
    ):
        corrected_df[
            "_system_key"
        ] = corrected_df[
            corrected_system_column
        ].map(system_key)

        semantic_for_merge = (
            system_summary_df.copy()
        )

        semantic_for_merge[
            "_system_key"
        ] = semantic_for_merge[
            "system_name"
        ].map(system_key)

        semantic_columns = [
            column
            for column
            in semantic_for_merge.columns
            if column != "system_name"
        ]

        combined_df = corrected_df.merge(
            semantic_for_merge[
                semantic_columns
            ],
            on="_system_key",
            how="outer",
        )

        combined_df = combined_df.drop(
            columns=["_system_key"],
            errors="ignore",
        )

        combined_path = (
            OUTPUT_DIR
            / "final_system_metrics_combined.csv"
        )

        combined_df.to_csv(
            combined_path,
            index=False,
        )

        combined_metrics_note = (
            "Corrected programmatic metrics "
            "were merged with the semantic "
            "judge metrics."
        )
    else:
        combined_metrics_note = (
            "The corrected metric summary "
            "appears to use a long format or "
            "has no detectable system column. "
            "It was preserved unchanged and "
            "not automatically merged."
        )

# --------------------------------------------------
# 9. Write audit report
# --------------------------------------------------

total_majority_pass = int(
    merged_df[
        "majority_pass"
    ].sum()
)

unanimous_count = int(
    merged_df[
        "unanimous_decision"
    ].sum()
)

audit_lines = [
    "FINAL THREE-RUN GPT-4.1 JUDGE AUDIT",
    "=" * 72,
    "",
    "Input integrity",
    f"- Run 1 rows: {len(run_indexes[1])}",
    f"- Run 2 rows: {len(run_indexes[2])}",
    f"- Run 3 rows: {len(run_indexes[3])}",
    "- Total judgments: 1200",
    "- Candidate IDs aligned across runs: Yes",
    "- Duplicate candidate IDs: 0",
    "",
    "Aggregation protocol",
    "- Three independent GPT-4.1 judgments per candidate",
    "- Final pass requires at least 2 of 3 runs",
    "- Rubric scores are averaged across all three runs",
    "- System identities were revealed only after all runs were locked",
    "",
    "Overall semantic results",
    f"- Majority-pass candidates: {total_majority_pass}/400",
    (
        "- Pooled majority-pass rate: "
        f"{100 * total_majority_pass / 400:.2f}%"
    ),
    f"- Unanimous decisions: {unanimous_count}/400",
    (
        "- Unanimous-decision rate: "
        f"{100 * unanimous_count / 400:.2f}%"
    ),
    (
        "- Overall Fleiss' kappa: "
        f"{overall_fleiss:.6f}"
    ),
    "",
    "Operational/E2E status",
    f"- {operational_note}",
    "",
    "Combined metrics status",
    f"- {combined_metrics_note}",
    "",
    "Generated files",
    f"- {blind_path}",
    f"- {unblinded_path}",
    f"- {system_summary_path}",
    f"- {agreement_path}",
    f"- {error_summary_path}",
]

audit_path = (
    OUTPUT_DIR
    / "final_judge_aggregation_audit.txt"
)

audit_path.write_text(
    "\n".join(audit_lines) + "\n",
    encoding="utf-8",
)

print("=" * 72)
print("THREE-RUN AGGREGATION COMPLETE")
print("=" * 72)
print(f"Candidates aggregated: {len(merged_df)}")
print("Total judgments: 1200")
print(
    "Systems revealed:",
    ", ".join(
        sorted(
            merged_df[
                "system_name"
            ].unique()
        )
    ),
)
print(
    "Majority-pass candidates:",
    f"{total_majority_pass}/400",
)
print(
    "Pooled majority-pass rate:",
    f"{100 * total_majority_pass / 400:.2f}%",
)
print(
    "Unanimous decisions:",
    f"{unanimous_count}/400",
)
print(
    "Overall Fleiss' kappa:",
    f"{overall_fleiss:.6f}",
)
print()
print("Per-system results:")
print(
    system_summary_df.to_string(
        index=False,
    )
)
print()
print("Operational status:")
print(operational_note)
print()
print("Outputs written to:")
print(OUTPUT_DIR)
