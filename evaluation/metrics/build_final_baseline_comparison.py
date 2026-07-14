from pathlib import Path
import csv
import re

OUT_CSV = Path("evaluation/metrics/final_baseline_comparison.csv")
OUT_TXT = Path("evaluation/metrics/final_baseline_comparison.txt")

def read_text(path):
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""

def extract_percent(text, label):
    m = re.search(re.escape(label) + r":\s*([0-9.]+)%", text)
    return m.group(1) + "%" if m else "N/A"

def extract_score(text, label="overall_score"):
    m = re.search(re.escape(label) + r":\s*([0-9.]+)/5", text)
    return m.group(1) + "/5" if m else "N/A"

full_metrics = read_text("evaluation/final_results_30_tests/final_metrics_30_tests.txt")
full_audit = read_text("evaluation/final_results_30_tests/promised_global_audit_v2.txt")
full_judge = read_text("evaluation/final_results_30_tests/llm_judge_3run_average_summary.txt")

direct_judge = read_text("evaluation/final_results_30_tests/baselines/direct_llm/direct_llm_judge_3run_average_summary.txt")

basic_metrics = read_text("evaluation/final_results_30_tests/baselines/basic_tool_use/basic_tool_use_metrics.txt")
basic_judge = read_text("evaluation/final_results_30_tests/baselines/basic_tool_use/basic_tool_use_judge_3run_average_summary.txt")

no_recovery = read_text("evaluation/final_results_30_tests/baselines/manager_without_recovery/manager_without_recovery_metrics_v2.txt")

rows = [
    {
        "condition": "Direct LLM Tutor",
        "workflow_or_process_completion": "100.00%",
        "tool_execution_success": "N/A",
        "tool_selection_precision": "N/A",
        "tool_selection_recall": "N/A",
        "tool_selection_f1": "N/A",
        "validation_success": "N/A",
        "controlled_recovery_success": "N/A",
        "trace_save_rate": "0.00%",
        "memory_update_success": "0.00%",
        "final_answer_generation": "100.00%",
        "llm_judge_3run_overall": extract_score(direct_judge),
        "interpretation": "Fluent direct answers, but no external tools, validation, recovery, trace lifecycle, or memory update.",
    },
    {
        "condition": "Basic Tool Use Without Central Manager",
        "workflow_or_process_completion": "100.00%",
        "tool_execution_success": extract_percent(basic_metrics, "Tool Execution Success Rate"),
        "tool_selection_precision": extract_percent(basic_metrics, "Tool Selection Precision"),
        "tool_selection_recall": extract_percent(basic_metrics, "Tool Selection Recall"),
        "tool_selection_f1": extract_percent(basic_metrics, "Tool Selection F1-score"),
        "validation_success": "0.00% / not used",
        "controlled_recovery_success": "0.00% / not used",
        "trace_save_rate": "0.00% / not used",
        "memory_update_success": "0.00% / not used",
        "final_answer_generation": extract_percent(basic_metrics, "Answer Generation Rate"),
        "llm_judge_3run_overall": extract_score(basic_judge),
        "interpretation": "Tools execute successfully, but static selection misses expected tools and there is no validation, recovery, trace lifecycle, or memory update.",
    },
    {
        "condition": "Manager Without Recovery",
        "workflow_or_process_completion": extract_percent(no_recovery, "Process Completion Rate"),
        "tool_execution_success": "N/A",
        "tool_selection_precision": "N/A",
        "tool_selection_recall": "N/A",
        "tool_selection_f1": "N/A",
        "validation_success": extract_percent(no_recovery, "Validation Success Rate"),
        "controlled_recovery_success": "0.00%",
        "trace_save_rate": extract_percent(no_recovery, "Trace Save Rate"),
        "memory_update_success": "N/A",
        "final_answer_generation": extract_percent(no_recovery, "Final Answer Generation Rate"),
        "llm_judge_3run_overall": "N/A for invalid/failed cases",
        "interpretation": "Validation exists, but forced recovery cases fail when retry/recovery is disabled.",
    },
    {
        "condition": "Full Tool Interaction Manager",
        "workflow_or_process_completion": extract_percent(full_metrics, "Workflow Completion Rate"),
        "tool_execution_success": extract_percent(full_audit, "Tool Execution Success Rate"),
        "tool_selection_precision": extract_percent(full_metrics, "Tool Selection Precision"),
        "tool_selection_recall": extract_percent(full_metrics, "Tool Selection Recall"),
        "tool_selection_f1": extract_percent(full_metrics, "Tool Selection F1-score"),
        "validation_success": extract_percent(full_metrics, "Output Validation Success Rate"),
        "controlled_recovery_success": extract_percent(full_audit, "Controlled Recovery Success Rate"),
        "trace_save_rate": extract_percent(full_metrics, "Trace Save Rate"),
        "memory_update_success": extract_percent(full_metrics, "Memory Update Success Rate"),
        "final_answer_generation": extract_percent(full_metrics, "Final Answer Generation Rate"),
        "llm_judge_3run_overall": extract_score(full_judge),
        "interpretation": "Complete lifecycle: centralized selection, planning, execution, validation, recovery, trace saving, memory update, and final answer generation.",
    },
]

fieldnames = list(rows[0].keys())

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = []
lines.append("FINAL BASELINE COMPARISON")
lines.append("=" * 80)
lines.append("")

for r in rows:
    lines.append(r["condition"])
    lines.append("-" * len(r["condition"]))
    for k in fieldnames:
        if k != "condition":
            lines.append(f"{k}: {r[k]}")
    lines.append("")

lines.append("Main conclusion:")
lines.append("- Direct LLM and Basic Tool Use can produce fluent answers, and their judge scores may be high.")
lines.append("- However, they do not provide the full controlled tool-interaction lifecycle promised by the proposed approach.")
lines.append("- Basic Tool Use has lower recall than the Full Manager, showing that simple static tool selection misses expected tools.")
lines.append("- Manager Without Recovery confirms that validation alone is insufficient: controlled recovery cases fail when recovery is disabled.")
lines.append("- The Full Tool Interaction Manager is strongest on reliability, recoverability, traceability, memory update, and lifecycle control.")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print()
print("Saved:", OUT_CSV)
print("Saved:", OUT_TXT)
