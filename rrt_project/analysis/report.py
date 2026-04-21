from __future__ import annotations

import json
from pathlib import Path

from rrt_project.analysis.summary_viz import render_summary_charts


def _rel(target: Path, base: Path) -> str:
    return str(target.relative_to(base)).replace('\\', '/')


def write_markdown_report(summary_path: Path, output_path: Path) -> None:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    reports_dir = output_path.parent
    project_root = reports_dir.parent.parent
    figures_dir = reports_dir / "figures"
    figures = render_summary_charts(
        summary_path=summary_path,
        records_path=project_root / "results" / "records.csv",
        out_dir=figures_dir,
    )

    lines = ["# Benchmark Summary", ""]

    lines.append("## Matrix")
    lines.append("")
    matrix = summary.get("matrix", {})
    lines.append(f"- scenes: {matrix.get('scene_count', 0)}")
    lines.append(f"- planners: {', '.join(matrix.get('planners', []))}")
    lines.append(f"- ablations: {', '.join(matrix.get('ablations', []))}")
    lines.append(f"- seeds: {len(matrix.get('seeds', []))}")

    lines.append("")
    lines.append("## Comparison Figures")
    lines.append("")
    lines.append(f"![success_rate_bar]({_rel(Path(figures['success_rate_bar']), reports_dir)})")
    lines.append("")
    lines.append(f"![median_time_bar]({_rel(Path(figures['median_time_bar']), reports_dir)})")
    lines.append("")
    lines.append(f"![planned_vs_validated_success_bar]({_rel(Path(figures['planned_vs_validated_success_bar']), reports_dir)})")
    lines.append("")
    lines.append(f"![complex_runtime_boxplot]({_rel(Path(figures['complex_runtime_boxplot']), reports_dir)})")
    lines.append("")
    lines.append(f"![complex_pathlen_boxplot]({_rel(Path(figures['complex_pathlen_boxplot']), reports_dir)})")

    lines.append("")
    lines.append("## Planner/Ablation Metrics")
    lines.append("")
    lines.append(
        "| planner | ablation | runs | planned_success_rate | validated_success_rate | "
        "hard_validation_pass_rate | median_time_ms | median_path_len |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    planners = summary.get("planners", {})
    for planner, ab_map in planners.items():
        for ablation, stats in ab_map.items():
            lines.append(
                f"| {planner} | {ablation} | {stats.get('runs', 0)} | "
                f"{stats.get('planned_success_rate', stats.get('success_rate', 0)):.3f} | "
                f"{stats.get('success_rate', 0):.3f} | "
                f"{stats.get('hard_validation_pass_rate', stats.get('success_rate', 0)):.3f} | "
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

    lines.append("")
    lines.append("## Tuning History")
    lines.append("")
    history = summary.get("tuning_history", [])
    if history:
        for item in history:
            lines.append(f"- round {item.get('round')}: {item.get('changes')}")
    else:
        lines.append("- none")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
