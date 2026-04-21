from __future__ import annotations

import json
from pathlib import Path


def write_markdown_report(summary_path: Path, output_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    lines = ["# Benchmark Summary", ""]
    planners = summary.get("planners", {})
    lines.append("## Planner Metrics")
    lines.append("")
    lines.append("| planner | runs | success_rate | median_time_ms | median_path_len |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, stats in planners.items():
        lines.append(
            f"| {name} | {stats.get('runs', 0)} | {stats.get('success_rate', 0):.3f} | "
            f"{stats.get('median_time_ms', 0):.2f} | {stats.get('median_path_len', 0):.4f} |"
        )

    lines.append("")
    lines.append("## Failure Counts")
    lines.append("")
    failures = summary.get("failure_counts", {})
    if failures:
        for k, v in failures.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
