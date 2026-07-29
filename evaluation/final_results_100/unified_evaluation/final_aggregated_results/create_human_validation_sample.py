#!/usr/bin/env python3

import json
import random
from pathlib import Path

import pandas as pd


BASE = Path(
    "evaluation/final_results_100/unified_evaluation"
)

AGGREGATE_PATH = (
    BASE
    / "final_aggregated_results"
    / "judge_candidate_aggregate_unblinded.csv"
)

CANDIDATES_PATH = (
    BASE / "judge_candidates_anonymized.jsonl"
)

OUTPUT_DIR = (
    BASE
    / "final_aggregated_results"
    / "human_validation"
)

RANDOM_SEED = 20260726
SAMPLES_PER_SYSTEM = 5


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "pass",
        "passed",
    }


def first_text(record, names):
    containers = [
        record,
        record.get("candidate", {}),
        record.get("data", {}),
        record.get("item", {}),
    ]

    for container in containers:
        if not isinstance(container, dict):
            continue

        for name in names:
            value = container.get(name)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def choose_rows(group, rng):
    group = group.copy()

    group["majority_pass"] = group[
        "majority_pass"
    ].map(parse_bool)

    group["unanimous_decision"] = group[
        "unanimous_decision"
    ].map(parse_bool)

    selected = []

    pools = [
        # Two cases where the three judge runs disagreed.
        group[
            ~group["unanimous_decision"]
        ],

        # One unanimous failure.
        group[
            group["unanimous_decision"]
            & ~group["majority_pass"]
        ],

        # One unanimous pass.
        group[
            group["unanimous_decision"]
            & group["majority_pass"]
        ],
    ]

    targets = [2, 1, 1]

    for pool, target in zip(pools, targets):
        available = [
            index
            for index in pool.index
            if index not in selected
        ]

        take = min(target, len(available))

        if take:
            selected.extend(
                rng.sample(available, take)
            )

    remaining = [
        index
        for index in group.index
        if index not in selected
    ]

    needed = SAMPLES_PER_SYSTEM - len(selected)

    if needed > len(remaining):
        raise SystemExit(
            f"Not enough cases for "
            f"{group['system_name'].iloc[0]}."
        )

    selected.extend(
        rng.sample(remaining, needed)
    )

    return group.loc[selected]


if not AGGREGATE_PATH.exists():
    raise SystemExit(
        f"Missing aggregate file: {AGGREGATE_PATH}"
    )

if not CANDIDATES_PATH.exists():
    raise SystemExit(
        f"Missing candidate file: {CANDIDATES_PATH}"
    )

aggregate = pd.read_csv(AGGREGATE_PATH)

required_columns = {
    "candidate_id",
    "test_id",
    "system_name",
    "majority_pass",
    "pass_count",
    "unanimous_decision",
    "mean_overall_score",
}

missing = required_columns - set(
    aggregate.columns
)

if missing:
    raise SystemExit(
        f"Missing aggregate columns: "
        f"{sorted(missing)}"
    )

candidate_records = {}

for line in CANDIDATES_PATH.read_text(
    encoding="utf-8"
).splitlines():
    if not line.strip():
        continue

    record = json.loads(line)
    candidate_id = record.get("candidate_id")

    if candidate_id:
        candidate_records[candidate_id] = record

rng = random.Random(RANDOM_SEED)

samples = []

for system_name, group in aggregate.groupby(
    "system_name",
    sort=True,
):
    sample = choose_rows(group, rng)
    samples.append(sample)

sample_df = pd.concat(
    samples,
    ignore_index=True,
)

# Shuffle so systems are not grouped in the blind file.
sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(drop=True)

sample_df["review_id"] = [
    f"H{number:03d}"
    for number in range(
        1,
        len(sample_df) + 1,
    )
]

questions = []
answers = []

for candidate_id in sample_df["candidate_id"]:
    record = candidate_records.get(
        candidate_id,
        {},
    )

    question = first_text(
        record,
        [
            "question",
            "prompt",
            "user_question",
            "input",
            "query",
        ],
    )

    answer = first_text(
        record,
        [
            "candidate_answer",
            "answer",
            "response",
            "final_answer",
            "output",
            "system_answer",
        ],
    )

    questions.append(question)
    answers.append(answer)

sample_df["question"] = questions
sample_df["candidate_answer"] = answers

missing_questions = int(
    (sample_df["question"].str.len() == 0).sum()
)

missing_answers = int(
    (
        sample_df["candidate_answer"].str.len()
        == 0
    ).sum()
)

if missing_questions or missing_answers:
    raise SystemExit(
        "Could not extract all question/answer text. "
        f"Missing questions={missing_questions}, "
        f"missing answers={missing_answers}."
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

blind_columns = [
    "review_id",
    "question",
    "candidate_answer",
]

blind_df = sample_df[blind_columns].copy()

blind_df["human_correctness_1_to_5"] = ""
blind_df["human_relevance_1_to_5"] = ""
blind_df["human_completeness_1_to_5"] = ""
blind_df[
    "human_pedagogical_quality_1_to_5"
] = ""
blind_df[
    "human_grounding_consistency_1_to_5"
] = ""
blind_df["human_critical_error_yes_no"] = ""
blind_df["human_pass_yes_no"] = ""
blind_df["reviewer_notes"] = ""

key_columns = [
    "review_id",
    "candidate_id",
    "test_id",
    "system_name",
    "majority_pass",
    "pass_count",
    "unanimous_decision",
    "mean_correctness",
    "mean_relevance",
    "mean_completeness",
    "mean_pedagogical_quality",
    "mean_grounding_consistency",
    "mean_overall_score",
]

blind_path = (
    OUTPUT_DIR
    / "human_validation_sample_blind.csv"
)

key_path = (
    OUTPUT_DIR
    / "human_validation_key_secret.csv"
)

blind_df.to_csv(
    blind_path,
    index=False,
)

sample_df[key_columns].to_csv(
    key_path,
    index=False,
)

print("=" * 72)
print("HUMAN VALIDATION SAMPLE CREATED")
print("=" * 72)
print("Sample size:", len(blind_df))
print()
print("Cases per system:")
print(
    sample_df[
        "system_name"
    ].value_counts().to_string()
)
print()
print(
    "Judge-disagreement cases:",
    int(
        (
            ~sample_df[
                "unanimous_decision"
            ].map(parse_bool)
        ).sum()
    ),
)
print(
    "Majority passes:",
    int(
        sample_df[
            "majority_pass"
        ].map(parse_bool).sum()
    ),
)
print(
    "Majority failures:",
    int(
        (
            ~sample_df[
                "majority_pass"
            ].map(parse_bool)
        ).sum()
    ),
)
print()
print("Blind review file:", blind_path)
print("Secret key file:", key_path)
