import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport
from evaluation.metrics.run_llm_judge_30_tests import (
    RUBRIC_FIELDS,
    judge_one,
    compute_summary,
)

DATASET_PATH = Path("evaluation/baselines/direct_llm/direct_llm_judge_dataset_30_tests.jsonl")
OUT_CSV = Path("evaluation/baselines/direct_llm/direct_llm_judge_scores_30_tests.csv")
OUT_JSONL = Path("evaluation/baselines/direct_llm/direct_llm_judge_raw_30_tests.jsonl")
OUT_SUMMARY = Path("evaluation/baselines/direct_llm/direct_llm_judge_summary_30_tests.txt")

def load_items():
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def load_previous():
    rows = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["test_id"]] = row
    return rows

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=10.0)
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("JUDGE_LLM_MODEL") or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant",
    )
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing judge dataset: {DATASET_PATH}. Run prepare_direct_llm_judge_dataset.py first.")

    items = load_items()
    previous = load_previous()

    ok_ids = {tid for tid, row in previous.items() if row.get("judge_status") == "ok"}
    todo = [item for item in items if item["test_id"] not in ok_ids]

    print("Existing OK judgments:", len(ok_ids))
    print("To judge:", [x["test_id"] for x in todo])

    llm = LLMService(OpenAICompatibleTransport())

    for i, item in enumerate(todo, start=1):
        tid = item["test_id"]
        print(f"[{i}/{len(todo)}] Judging Direct LLM {tid}...")

        try:
            row = await judge_one(llm, args.model, item)
        except Exception as exc:
            row = {
                "test_id": tid,
                "category": item.get("category", ""),
                "question": item.get("question", ""),
                "model": args.model,
                "judge_status": "error",
                **{field: 0 for field in RUBRIC_FIELDS},
                "overall_score": 0,
                "major_issue": "judge_error",
                "rationale": str(exc),
                "raw_judge_output": "",
            }

        previous[tid] = row
        await asyncio.sleep(args.sleep)

        with OUT_JSONL.open("w", encoding="utf-8") as f:
            for original in items:
                saved = previous.get(original["test_id"])
                if saved:
                    f.write(json.dumps(saved, ensure_ascii=False) + "\n")

    ordered = [previous[item["test_id"]] for item in items]

    fieldnames = [
        "test_id",
        "category",
        "question",
        "model",
        "judge_status",
        *RUBRIC_FIELDS,
        "overall_score",
        "major_issue",
        "rationale",
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = compute_summary(ordered).replace(
        "LLM-AS-A-JUDGE QUALITY EVALUATION — 30 TESTS",
        "DIRECT LLM BASELINE — LLM-AS-A-JUDGE QUALITY EVALUATION"
    )
    OUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_SUMMARY)

if __name__ == "__main__":
    asyncio.run(main())
