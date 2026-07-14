import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport

from evaluation.metrics.run_llm_judge_30_tests import (
    DATASET_PATH,
    OUT_CSV,
    OUT_JSONL,
    OUT_SUMMARY,
    RUBRIC_FIELDS,
    judge_one,
    compute_summary,
)

def load_dataset():
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def load_previous_rows():
    rows = {}
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["test_id"]] = row
    return rows

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("JUDGE_LLM_MODEL") or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant",
    )
    parser.add_argument("--sleep", type=float, default=6.0)
    args = parser.parse_args()

    items = load_dataset()
    previous = load_previous_rows()

    ok_ids = {tid for tid, row in previous.items() if row.get("judge_status") == "ok"}
    retry_items = [item for item in items if item["test_id"] not in ok_ids]

    print("Existing OK judgments:", len(ok_ids))
    print("To retry:", [x["test_id"] for x in retry_items])

    llm = LLMService(OpenAICompatibleTransport())

    for i, item in enumerate(retry_items, start=1):
        tid = item["test_id"]
        print(f"[retry {i}/{len(retry_items)}] Judging {tid}...")

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

    ordered_rows = [previous[item["test_id"]] for item in items]

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in ordered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
        for row in ordered_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = compute_summary(ordered_rows)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_SUMMARY)

if __name__ == "__main__":
    asyncio.run(main())
