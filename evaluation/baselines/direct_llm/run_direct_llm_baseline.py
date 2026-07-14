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

from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport

DATASET_PATH = Path("evaluation/metrics/judge_dataset_30_tests.jsonl")
OUT_JSONL = Path("evaluation/baselines/direct_llm/direct_llm_answers_30_tests.jsonl")
OUT_CSV = Path("evaluation/baselines/direct_llm/direct_llm_answers_30_tests.csv")

def load_dataset():
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

def build_prompt(item):
    return f"""
You are a tutoring assistant.

Answer the student question directly.

Important baseline condition:
- Do NOT use external tools.
- Do NOT mention RAG, traces, validators, tool outputs, memory, or recovery.
- Do NOT claim that you calculated using a tool.
- Give the best direct LLM-only tutoring answer you can.
- Be clear, educational, and concise.

Student question:
{item["question"]}
""".strip()

async def generate_one(llm, model, item):
    request = LLMRequest(
        model=model,
        messages=[
            Message(role="system", content="You are a direct LLM tutor baseline. Answer without external tools."),
            Message(role="user", content=build_prompt(item)),
        ],
        temperature=0.0,
        max_tokens=1200,
    )

    response = await llm.complete(request)

    return {
        "test_id": item["test_id"],
        "category": item.get("category", ""),
        "question": item.get("question", ""),
        "baseline": "direct_llm_tutor",
        "model": model,
        "status": "ok",
        "final_answer": response.completion.strip(),
        "error": "",
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("BASELINE_LLM_MODEL") or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant",
    )
    parser.add_argument("--sleep", type=float, default=6.0)
    args = parser.parse_args()

    items = load_dataset()
    previous = load_previous()

    done = {tid for tid, row in previous.items() if row.get("status") == "ok" and len(row.get("final_answer", "")) > 50}
    todo = [item for item in items if item["test_id"] not in done]

    print("Existing OK answers:", len(done))
    print("To generate:", [x["test_id"] for x in todo])

    llm = LLMService(OpenAICompatibleTransport())

    for i, item in enumerate(todo, start=1):
        tid = item["test_id"]
        print(f"[{i}/{len(todo)}] Direct LLM baseline for {tid}...")

        try:
            row = await generate_one(llm, args.model, item)
        except Exception as exc:
            row = {
                "test_id": tid,
                "category": item.get("category", ""),
                "question": item.get("question", ""),
                "baseline": "direct_llm_tutor",
                "model": args.model,
                "status": "error",
                "final_answer": "",
                "error": str(exc),
            }

        previous[tid] = row
        await asyncio.sleep(args.sleep)

        with OUT_JSONL.open("w", encoding="utf-8") as f:
            for original in items:
                saved = previous.get(original["test_id"])
                if saved:
                    f.write(json.dumps(saved, ensure_ascii=False) + "\n")

    ordered_rows = [previous[item["test_id"]] for item in items if item["test_id"] in previous]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["test_id", "category", "question", "baseline", "model", "status", "final_answer", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered_rows)

    ok = [r for r in ordered_rows if r.get("status") == "ok"]
    err = [r for r in ordered_rows if r.get("status") != "ok"]

    print()
    print("DIRECT LLM BASELINE")
    print("=" * 50)
    print("Total rows:", len(ordered_rows))
    print("OK:", len(ok))
    print("Errors:", len(err))
    for r in err:
        print(r["test_id"], "=>", r.get("error", "")[:250])

    print()
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    asyncio.run(main())
