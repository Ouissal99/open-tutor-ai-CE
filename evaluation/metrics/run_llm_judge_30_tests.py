import argparse
import asyncio
import csv
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from ai.llm.schemas import LLMRequest, Message
from ai.llm.service import LLMService
from ai.llm.transports.openai_compatible import OpenAICompatibleTransport

DATASET_PATH = Path("evaluation/metrics/judge_dataset_30_tests.jsonl")
OUT_CSV = Path("evaluation/metrics/llm_judge_scores_30_tests.csv")
OUT_JSONL = Path("evaluation/metrics/llm_judge_raw_30_tests.jsonl")
OUT_SUMMARY = Path("evaluation/metrics/llm_judge_summary_30_tests.txt")

RUBRIC_FIELDS = [
    "correctness",
    "source_faithfulness",
    "grounding_quality",
    "personalization",
    "applicability",
    "vividness",
    "logical_depth",
    "clarity",
]

def load_items():
    items = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])

    raise ValueError("Could not parse judge JSON")

def build_prompt(item):
    return f"""
You are an evaluation judge for a Master thesis prototype about an agentic tutoring system.

Evaluate the FINAL ANSWER only with respect to the STUDENT QUESTION and the available EVIDENCE EXCERPT.

Use a strict 1-5 scale:
1 = very poor
2 = weak
3 = acceptable but with important issues
4 = good
5 = excellent

Important judging rules:
- Correctness: penalize mathematical or conceptual mistakes.
- Source faithfulness: penalize claims not supported by the evidence/tools.
- Grounding quality: reward explicit use of validated tool outputs and references.
- Personalization: reward adaptation to learner level or stated learner difficulty.
- Applicability: reward directly answering the exact question.
- Vividness: reward helpful examples, analogies, or visual explanations.
- Logical depth: reward step-by-step reasoning and explanation structure.
- Clarity: reward easy-to-read, coherent wording.
- If the answer is generic and does not answer the exact question, lower correctness and applicability.
- If the answer says a calculation was not required when the question asks for calculation, lower correctness/applicability.
- If the answer contains duplicated references, minor formatting issues, or verbose evidence dumps, lower clarity slightly.
- Do not be overly generous. Use 5 only when the answer is clearly excellent.

Return ONLY valid JSON with this exact schema:
{{
  "correctness": 1,
  "source_faithfulness": 1,
  "grounding_quality": 1,
  "personalization": 1,
  "applicability": 1,
  "vividness": 1,
  "logical_depth": 1,
  "clarity": 1,
  "overall_score": 1,
  "major_issue": "short text",
  "rationale": "short explanation"
}}

STUDENT QUESTION:
{item.get("question", "")}

CATEGORY:
{item.get("category", "")}

EXPECTED TOOLS:
{item.get("expected_tools", "")}

SELECTED TOOLS:
{item.get("selected_tools", "")}

EVIDENCE EXCERPT:
{item.get("evidence_excerpt", "")[:3500]}

FINAL ANSWER:
{item.get("final_answer", "")[:6000]}
""".strip()

async def judge_one(llm, model, item):
    request = LLMRequest(
        model=model,
        messages=[
            Message(role="system", content="You are a strict educational answer-quality evaluator. Return only JSON."),
            Message(role="user", content=build_prompt(item)),
        ],
        temperature=0.0,
        max_tokens=1200,
    )

    response = await llm.complete(request)
    raw = response.completion
    parsed = extract_json(raw)

    result = {
        "test_id": item["test_id"],
        "category": item.get("category", ""),
        "question": item.get("question", ""),
        "model": model,
        "judge_status": "ok",
    }

    def to_score(value):
        try:
            if isinstance(value, str):
                m = re.search(r"[-+]?\\d*\\.?\\d+", value)
                value = m.group(0) if m else 0
            value = float(value)
            if value < 1:
                value = 1
            if value > 5:
                value = 5
            return round(value, 2)
        except Exception:
            return 0

    for field in RUBRIC_FIELDS:
        result[field] = to_score(parsed.get(field, 0))

    result["overall_score"] = to_score(parsed.get("overall_score", 0))
    result["major_issue"] = str(parsed.get("major_issue", "")).replace("\n", " ").strip()
    result["rationale"] = str(parsed.get("rationale", "")).replace("\n", " ").strip()
    result["raw_judge_output"] = raw

    return result

def compute_summary(rows):
    ok_rows = [r for r in rows if r.get("judge_status") == "ok"]

    lines = []
    lines.append("LLM-AS-A-JUDGE QUALITY EVALUATION — 30 TESTS")
    lines.append("=" * 60)
    lines.append(f"Judged tests: {len(ok_rows)} / {len(rows)}")
    lines.append("")

    if not ok_rows:
        return "\n".join(lines)

    for field in RUBRIC_FIELDS + ["overall_score"]:
        vals = [float(r[field]) for r in ok_rows]
        avg = sum(vals) / len(vals)
        lines.append(f"{field}: {avg:.2f}/5")

    lines.append("")
    lines.append("Lowest overall scores:")
    for r in sorted(ok_rows, key=lambda x: float(x["overall_score"]))[:8]:
        lines.append(
            f'{r["test_id"]}: overall={float(r["overall_score"]):.2f}/5 | issue={r.get("major_issue","")}'
        )

    return "\n".join(lines)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--model", type=str, default=os.getenv("JUDGE_LLM_MODEL") or os.getenv("AGENTIC_LLM_MODEL") or "llama-3.1-8b-instant")
    args = parser.parse_args()

    items = load_items()

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        items = [x for x in items if x["test_id"] in wanted]

    if args.limit:
        items = items[:args.limit]

    llm = LLMService(OpenAICompatibleTransport())

    rows = []
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as raw_out:
        for i, item in enumerate(items, start=1):
            tid = item["test_id"]
            print(f"[{i}/{len(items)}] Judging {tid}...")

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

            rows.append(row)
            raw_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_out.flush()

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
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = compute_summary(rows)
    OUT_SUMMARY.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print()
    print("Saved:", OUT_CSV)
    print("Saved:", OUT_JSONL)
    print("Saved:", OUT_SUMMARY)

if __name__ == "__main__":
    asyncio.run(main())
