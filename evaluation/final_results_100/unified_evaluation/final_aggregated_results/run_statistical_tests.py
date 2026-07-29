#!/usr/bin/env python3

import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "evaluation/final_results_100/unified_evaluation/"
    "final_aggregated_results"
)

INPUT_PATH = (
    BASE / "judge_candidate_aggregate_unblinded.csv"
)
OUTPUT_PATH = BASE / "statistical_tests.csv"

RANDOM_SEED = 20260726
BOOTSTRAP_SAMPLES = 20000

df = pd.read_csv(INPUT_PATH)

required = {
    "test_id",
    "system_name",
    "majority_pass",
    "mean_overall_score",
}

missing = required - set(df.columns)

if missing:
    raise SystemExit(
        f"Missing required columns: {sorted(missing)}"
    )

df["majority_pass"] = (
    df["majority_pass"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)

if df["majority_pass"].isna().any():
    raise SystemExit(
        "Could not parse some majority_pass values."
    )

systems = sorted(df["system_name"].unique())

expected_systems = {
    "Direct LLM",
    "Basic Tool Use",
    "Manager Without Recovery",
    "Full Manager",
}

if set(systems) != expected_systems:
    raise SystemExit(
        f"Unexpected systems: {systems}"
    )


def exact_mcnemar(first, second):
    first_only = int(
        ((first == 1) & (second == 0)).sum()
    )

    second_only = int(
        ((first == 0) & (second == 1)).sum()
    )

    discordant = first_only + second_only

    if discordant == 0:
        return (
            first_only,
            second_only,
            1.0,
        )

    smaller = min(first_only, second_only)

    probability = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2 ** discordant)

    p_value = min(1.0, 2 * probability)

    return (
        first_only,
        second_only,
        p_value,
    )


def bootstrap_difference(
    first,
    second,
    seed,
):
    rng = np.random.default_rng(seed)

    differences = first - second
    total = len(differences)

    samples = rng.choice(
        differences,
        size=(BOOTSTRAP_SAMPLES, total),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(differences.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


comparisons = [
    ("Full Manager", "Direct LLM"),
    ("Full Manager", "Basic Tool Use"),
    (
        "Full Manager",
        "Manager Without Recovery",
    ),
]

records = []

for index, (first_system, second_system) in enumerate(
    comparisons,
    start=1,
):
    first = (
        df[df["system_name"] == first_system]
        .set_index("test_id")
        .sort_index()
    )

    second = (
        df[df["system_name"] == second_system]
        .set_index("test_id")
        .sort_index()
    )

    common = first.index.intersection(second.index)

    if len(common) != 100:
        raise SystemExit(
            f"{first_system} vs {second_system} "
            f"has {len(common)} paired tests."
        )

    first_pass = (
        first.loc[common, "majority_pass"]
        .astype(int)
        .to_numpy()
    )

    second_pass = (
        second.loc[common, "majority_pass"]
        .astype(int)
        .to_numpy()
    )

    first_only, second_only, mcnemar_p = (
        exact_mcnemar(
            first_pass,
            second_pass,
        )
    )

    pass_difference, pass_low, pass_high = (
        bootstrap_difference(
            first_pass.astype(float),
            second_pass.astype(float),
            RANDOM_SEED + index,
        )
    )

    first_score = (
        first.loc[common, "mean_overall_score"]
        .astype(float)
        .to_numpy()
    )

    second_score = (
        second.loc[common, "mean_overall_score"]
        .astype(float)
        .to_numpy()
    )

    score_difference, score_low, score_high = (
        bootstrap_difference(
            first_score,
            second_score,
            RANDOM_SEED + 100 + index,
        )
    )

    records.append(
        {
            "comparison": (
                f"{first_system} vs {second_system}"
            ),
            "first_system_accuracy_pct": (
                100 * first_pass.mean()
            ),
            "second_system_accuracy_pct": (
                100 * second_pass.mean()
            ),
            "accuracy_difference_pp": (
                100 * pass_difference
            ),
            "accuracy_difference_ci95_lower_pp": (
                100 * pass_low
            ),
            "accuracy_difference_ci95_upper_pp": (
                100 * pass_high
            ),
            "first_only_successes": first_only,
            "second_only_successes": second_only,
            "mcnemar_exact_p_value": mcnemar_p,
            "first_mean_overall_score": (
                first_score.mean()
            ),
            "second_mean_overall_score": (
                second_score.mean()
            ),
            "mean_score_difference": (
                score_difference
            ),
            "mean_score_difference_ci95_lower": (
                score_low
            ),
            "mean_score_difference_ci95_upper": (
                score_high
            ),
        }
    )

results = pd.DataFrame(records)

results["mcnemar_significant_0_05"] = (
    results["mcnemar_exact_p_value"] < 0.05
)

results["accuracy_ci_excludes_zero"] = (
    (
        results[
            "accuracy_difference_ci95_lower_pp"
        ] > 0
    )
    | (
        results[
            "accuracy_difference_ci95_upper_pp"
        ] < 0
    )
)

results["score_ci_excludes_zero"] = (
    (
        results[
            "mean_score_difference_ci95_lower"
        ] > 0
    )
    | (
        results[
            "mean_score_difference_ci95_upper"
        ] < 0
    )
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("=" * 110)
print("PAIRED STATISTICAL TESTS")
print("=" * 110)
print(results.to_string(index=False))
print()
print("Written to:", OUTPUT_PATH)
