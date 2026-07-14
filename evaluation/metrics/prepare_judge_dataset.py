import csv
import json
import re
from pathlib import Path

tracking_path = Path("evaluation/metrics/final_evaluation_tracking_30_tests.csv")
out_path = Path("evaluation/metrics/judge_dataset_30_tests.jsonl")

def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def extract_final_answer(log_text):
    marker = "[8] Final tutor answer generated"
    if marker not in log_text:
        return ""

    after = log_text.split(marker, 1)[1]

    stop_markers = [
        "\n[9] Scratchpad state",
        "\n[10] Tool interaction trace saved",
        "\n================================================================================",
    ]

    end = len(after)
    for stop in stop_markers:
        idx = after.find(stop)
        if idx != -1:
            end = min(end, idx)

    return after[:end].strip()

def extract_evidence(log_text):
    evidence_parts = []

    patterns = [
        r"Evidence Used[:\n](.*?)(?:\n\*\*References|\n\[9\] Scratchpad state|\n\[10\]|$)",
        r"Evidence used[:\n](.*?)(?:\n\*\*References|\n\[9\] Scratchpad state|\n\[10\]|$)",
        r"### Validated output(.*?)(?:\n\[9\] Scratchpad state|\n\[10\]|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, log_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            evidence_parts.append(m.group(1).strip())

    return "\n\n".join(evidence_parts).strip()

with tracking_path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

count = 0
missing_answer = []

with out_path.open("w", encoding="utf-8") as out:
    for row in rows:
        tid = row["test_id"]
        log_path = Path(f"evaluation/runs/{tid}_full_manager.txt")
        log_text = read_text(log_path)

        answer = extract_final_answer(log_text)
        evidence = extract_evidence(log_text)

        if not answer:
            missing_answer.append(tid)

        item = {
            "test_id": tid,
            "category": row.get("category", ""),
            "question": row.get("question", ""),
            "expected_tools": row.get("expected_tools", ""),
            "selected_tools": row.get("selected_tools", ""),
            "workflow_status": row.get("workflow_status", ""),
            "validation_status": row.get("validation_status", ""),
            "recovery_used": row.get("recovery_used", ""),
            "attempts": row.get("attempts", ""),
            "trace_path": row.get("trace_path", ""),
            "report_path": row.get("report_path", ""),
            "final_answer": answer,
            "evidence_excerpt": evidence[:4000],
        }

        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1

print("Prepared judge dataset:", out_path)
print("Items:", count)
print("Missing final answers:", missing_answer)
