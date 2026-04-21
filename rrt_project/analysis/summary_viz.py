from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_records(records_path: Path) -> List[dict]:
    rows: List[dict] = []
    with records_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            row["success"] = str(row.get("success", "")).lower() in {"true", "1", "yes"}
            row["planned_success"] = str(row.get("planned_success", "")).lower() in {"true", "1", "yes"}
            row["hard_validation_passed"] = str(row.get("hard_validation_passed", "")).lower() in {"true", "1", "yes"}
            row["time_ms"] = float(row.get("time_ms", 0.0) or 0.0)
            row["path_len"] = float(row.get("path_len", 0.0) or 0.0)
            row["difficulty"] = row.get("difficulty", "")
            rows.append(row)
    return rows


def _abbr(label: str) -> str:
    # planner::ablation -> short label
    p, a = label.split("::", 1)
    p_map = {
        "rrt": "RRT",
        "rrt_connect": "RRTC",
        "rrt_star": "RRT*",
        "cci_bridge_rrt": "CBRRT",
    }
    a_map = {
        "baseline": "B",
        "cci_only": "C",
        "bridge_only": "BR",
        "cci_bridge": "CB",
    }
    return f"{p_map.get(p,p)}-{a_map.get(a,a)}"


def render_summary_charts(summary_path: Path, records_path: Path, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    groups: Dict[str, dict] = summary.get("planner_ablation_groups", {})

    ordered = sorted(groups.items(), key=lambda kv: kv[0])
    labels = [k for k, _ in ordered]
    xlabels = [_abbr(k) for k in labels]

    # 1) Success rate bar chart
    succ = [v.get("success_rate", 0.0) for _, v in ordered]
    plt.figure(figsize=(14, 4.8), dpi=140)
    bars = plt.bar(range(len(labels)), succ, color="#3b82f6")
    for i, b in enumerate(bars):
        plt.text(i, b.get_height() + 0.01, f"{succ[i]:.2f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, 1.08)
    plt.xticks(range(len(labels)), xlabels, rotation=45, ha="right")
    plt.ylabel("Success Rate")
    plt.title("Success Rate by Planner/Ablation")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p_success = out_dir / "success_rate_bar.png"
    plt.savefig(p_success)
    plt.close()

    # 2) Median time bar chart
    med_time = [v.get("median_time_ms", 0.0) for _, v in ordered]
    plt.figure(figsize=(14, 4.8), dpi=140)
    bars = plt.bar(range(len(labels)), med_time, color="#10b981")
    for i, b in enumerate(bars):
        plt.text(i, b.get_height() + max(5.0, 0.01 * max(med_time)), f"{med_time[i]:.0f}", ha="center", va="bottom", fontsize=7)
    plt.xticks(range(len(labels)), xlabels, rotation=45, ha="right")
    plt.ylabel("Median Planning Time (ms)")
    plt.title("Median Planning Time by Planner/Ablation")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p_time_bar = out_dir / "median_time_bar.png"
    plt.savefig(p_time_bar)
    plt.close()

    # 2.5) Planned success vs hard-validated success
    planned = [v.get("planned_success_rate", v.get("success_rate", 0.0)) for _, v in ordered]
    validated = [v.get("success_rate", 0.0) for _, v in ordered]
    xpos = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(14, 4.8), dpi=140)
    plt.bar(xpos - width / 2.0, planned, width=width, color="#60a5fa", label="planned_success_rate")
    plt.bar(xpos + width / 2.0, validated, width=width, color="#2563eb", label="hard_validated_success_rate")
    plt.ylim(0, 1.08)
    plt.xticks(xpos, xlabels, rotation=45, ha="right")
    plt.ylabel("Rate")
    plt.title("Planned vs Hard-Validated Success Rate")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    p_validation_cmp = out_dir / "planned_vs_validated_success_bar.png"
    plt.savefig(p_validation_cmp)
    plt.close()

    rows = _load_records(records_path)

    # 3) Complex-scene runtime boxplot
    data_time: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["difficulty"] == "complex":
            data_time[f"{r['planner']}::{r.get('ablation_mode','baseline')}"] .append(r["time_ms"])
    ordered_keys = [k for k in labels if k in data_time]
    plt.figure(figsize=(14, 5.5), dpi=140)
    plt.boxplot([data_time[k] for k in ordered_keys], labels=[_abbr(k) for k in ordered_keys], showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Planning Time (ms)")
    plt.title("Runtime Distribution on Complex Scenes")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p_time_box = out_dir / "complex_runtime_boxplot.png"
    plt.savefig(p_time_box)
    plt.close()

    # 4) Complex-scene path-length boxplot (successful only)
    data_len: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["difficulty"] == "complex" and r["success"]:
            data_len[f"{r['planner']}::{r.get('ablation_mode','baseline')}"] .append(r["path_len"])
    ordered_keys_len = [k for k in labels if k in data_len and len(data_len[k]) > 0]
    plt.figure(figsize=(14, 5.5), dpi=140)
    plt.boxplot([data_len[k] for k in ordered_keys_len], labels=[_abbr(k) for k in ordered_keys_len], showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Path Length")
    plt.title("Path-Length Distribution on Complex Scenes (Successful Runs)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    p_len_box = out_dir / "complex_pathlen_boxplot.png"
    plt.savefig(p_len_box)
    plt.close()

    return {
        "success_rate_bar": str(p_success),
        "median_time_bar": str(p_time_bar),
        "planned_vs_validated_success_bar": str(p_validation_cmp),
        "complex_runtime_boxplot": str(p_time_box),
        "complex_pathlen_boxplot": str(p_len_box),
    }
