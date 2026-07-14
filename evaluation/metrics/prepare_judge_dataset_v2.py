import csv, json, re
from pathlib import Path

tracking = Path("evaluation/metrics/final_evaluation_tracking_30_tests.csv")
out_path = Path("evaluation/metrics/judge_dataset_30_tests.jsonl")

def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

def answer_from_log(text):
    markers = ["[8] Final tutor answer generated", "Final tutor answer generated"]
    stops = ["\n[9] Scratchpad state", "\n[10] Tool interaction trace saved", "\n===="]
    for m in markers:
        if m in text:
            part = text.split(m, 1)[1]
            end = len(part)
            for s in stops:
                i = part.find(s)
                if i != -1:
                    end = min(end, i)
            ans = part[:end].strip()
            if len(ans) > 80:
                return ans
    return ""

def candidates_from_json(obj):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, str):
                txt = v.strip()
                if len(txt) > 80:
                    score = 0
                    if "final" in kl: score += 4
                    if "answer" in kl: score += 4
                    if "response" in kl: score += 3
                    if "output" in kl: score += 2
                    if any(w in txt.lower() for w in ["convolution", "kernel", "padding", "stride"]):
                        score += 1
                    found.append((score, len(txt), txt))
            found.extend(candidates_from_json(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(candidates_from_json(v))
    return found

def answer_from_report(obj):
    cands = candidates_from_json(obj)
    if not cands:
        return ""
    cands.sort(reverse=True)
    return cands[0][2].strip()

def evidence_from_log(text):
    m = re.search(
        r"(Evidence Used|Evidence used|### Validated output)(.*?)(\n\[9\] Scratchpad state|\n\[10\]|$)",
        text,
        flags=re.I | re.S,
    )
    return m.group(0).strip()[:4000] if m else ""

with tracking.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

missing = []

with out_path.open("w", encoding="utf-8") as out:
    for row in rows:
        tid = row["test_id"]
        log_text = read_text(f"evaluation/runs/{tid}_full_manager.txt")
        report = load_json(row["report_path"])

        ans = answer_from_log(log_text)
        if not ans:
            ans = answer_from_report(report)

        if not ans:
            missing.append(tid)

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
            "final_answer": ans,
            "evidence_excerpt": evidence_from_log(log_text),
        }
        out.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Prepared:", out_path)
print("Items:", len(rows))
print("Missing final answers:", missing)
