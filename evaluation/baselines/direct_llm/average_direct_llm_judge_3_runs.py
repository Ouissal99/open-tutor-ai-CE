import csv
from pathlib import Path
from collections import defaultdict

runs = [
    Path("evaluation/baselines/direct_llm/judge_repeats/run1/direct_llm_judge_scores_30_tests.csv"),
    Path("evaluation/baselines/direct_llm/judge_repeats/run2/direct_llm_judge_scores_30_tests.csv"),
    Path("evaluation/baselines/direct_llm/judge_repeats/run3/direct_llm_judge_scores_30_tests.csv"),
]

fields = [
    "correctness",
    "source_faithfulness",
    "grounding_quality",
    "personalization",
    "applicability",
    "vividness",
    "logical_depth",
    "clarity",
    "overall_score",
]

per_test = defaultdict(lambda: defaultdict(list))
meta = {}

for path in runs:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 30:
        raise ValueError(f"{path} has {len(rows)} rows, expected 30")

    bad = [r["test_id"] for r in rows if r.get("judge_status") != "ok"]
    if bad:
        raise ValueError(f"{path} has judge errors: {bad}")

    for row in rows:
        tid = row["test_id"]
        meta[tid] = {
            "test_id": tid,
            "category": row.get("category", ""),
            "question": row.get("question", ""),
        }
        for field in fields:
            per_test[tid][field].append(float(row[field]))

out_csv = Path("evaluation/baselines/direct_llm/direct_llm_judge_3run_average_scores.csv")
out_summary = Path("evaluation/baselines/direct_llm/direct_llm_judge_3run_average_summary.txt")

rows_out = []

for tid in sorted(per_test.keys()):
    row = dict(meta[tid])
    for field in fields:
        vals = per_test[tid][field]
        row[field] = round(sum(vals) / len(vals), 3)
    rows_out.append(row)

with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["test_id", "category", "question", *fields])
    writer.writeheader()
    writer.writerows(rows_out)

lines = []
lines.append("DIRECT LLM BASELINE — 3-RUN LLM-AS-A-JUDGE AVERAGE")
lines.append("=" * 70)
lines.append("Judge runs: 3")
lines.append("Judged tests per run: 30")
lines.append("Total judge decisions: 90")
lines.append("")

for field in fields:
    vals = [float(r[field]) for r in rows_out]
    lines.append(f"{field}: {sum(vals)/len(vals):.2f}/5")

lines.append("")
lines.append("Lowest average overall scores:")
for r in sorted(rows_out, key=lambda x: float(x["overall_score"]))[:8]:
    lines.append(f'{r["test_id"]}: overall={float(r["overall_score"]):.2f}/5 | {r["question"]}')

out_summary.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print()
print("Saved:", out_csv)
print("Saved:", out_summary)
