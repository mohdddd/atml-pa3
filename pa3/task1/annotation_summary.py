"""Summarise the Task 1 qualitative annotation sheet per condition (and per format).

    python -m pa3.task1.annotation_summary
reads  results/task1/summary/annotation_sheet.csv
writes results/task1/summary/annotation_summary.{csv,md}
"""
from __future__ import annotations

import csv
from collections import defaultdict

from pa3.common.paths import results_dir

ORDER = ["Baseline", "Full FT (mid, u625)", "Full FT", "Full FT + L2-SP", "LoRA r=4", "LoRA r=16", "OPLoRA r=16"]


def main() -> None:
    d = results_dir("task1", "summary")
    rows = list(csv.DictReader(open(d / "annotation_sheet.csv")))
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = []
    for c in ORDER:
        rs = by.get(c, [])
        if not rs:
            continue
        fmt = [r for r in rs if r["format"] != "plain"]
        out.append({
            "condition": c,
            "responses": len(rs),
            "format_yes": sum(r["format_compliance"] == "yes" for r in fmt),
            "format_partial": sum(r["format_compliance"] == "partial" for r in fmt),
            "format_no": sum(r["format_compliance"] == "no" for r in fmt),
            "json_valid": sum(r["format_compliance"] == "yes" for r in fmt if r["format"] == "json"),
            "template_exact": sum(r["format_compliance"] == "yes" for r in fmt if r["format"] == "template"),
            "mean_coherence": round(sum(int(r["coherence_relevance"]) for r in rs) / len(rs), 2),
            "repetition_minor": sum(r["repetition"] == "minor" for r in rs),
            "repetition_severe": sum(r["repetition"] == "severe" for r in rs),
            "math_framing": sum(r["unnecessary_math_framing"] == "yes" for r in rs),
            "inappropriate": sum(r["inappropriate_content"] != "none" for r in rs),
            "truncated": sum(r["truncated"] == "True" for r in rs),
        })
    with open(d / "annotation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    hdr = ["condition", "JSON valid (of 2)", "template exact (of 2)", "format yes/partial/no (of 4)",
           "mean coherence (1-5)", "repetition minor/severe", "math framing (of 6)", "inappropriate", "truncated"]
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
    for o in out:
        md += (f"| {o['condition']} | {o['json_valid']} | {o['template_exact']} | "
               f"{o['format_yes']}/{o['format_partial']}/{o['format_no']} | {o['mean_coherence']:.2f} | "
               f"{o['repetition_minor']}/{o['repetition_severe']} | {o['math_framing']} | {o['inappropriate']} | "
               f"{o['truncated']} |\n")
    md += ("\n6 responses per condition (2 questions × plain/JSON/template); format columns count the 4 JSON + "
           "template responses. Rubric: annotation_rubric.md.\n")
    (d / "annotation_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
