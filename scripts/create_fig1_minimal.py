#!/usr/bin/env python3
"""
Create Figure 1 from canonical paper metrics.

Source of truth: outputs/paper_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "outputs" / "paper_metrics.json"
DEFAULT_OUT = ROOT / "paper" / "figures" / "fig1_main_results.pdf"


FIG_METHODS = [
    ("UKGE", "UKGE"),
    ("Energy", "Energy"),
    ("MC\nDropout", "MCDropout"),
    ("Deep\nEnsemble", "DeepEnsemble"),
    ("SNGP", "SNGP"),
    ("", None),
    ("Semantic\n($U_{sem}$)", "GPOnly"),
    ("Structural\n($U_{str}$)", "CoverageOnly"),
    ("", None),
    ("CAGP\n(ours)", "CAGP"),
]


def _load_metrics(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_figure_series(metrics_payload: dict, dataset: str = "fb15k237") -> Dict[str, List[float]]:
    """
    Build deterministic plotting arrays from the canonical metrics payload.
    """
    temporal = metrics_payload["paper_summary"]["temporal_ood"][dataset]

    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []
    statuses: List[str] = []

    for label, method_key in FIG_METHODS:
        labels.append(label)
        if method_key is None:
            values.append(np.nan)
            errors.append(0.0)
            statuses.append("spacer")
            continue

        method_blob = temporal[method_key]
        status = method_blob["status"]
        val = method_blob["overall_auroc"]
        err = method_blob["overall_auroc_std"]

        if status != "ok" or val is None:
            values.append(np.nan)
            errors.append(0.0)
            statuses.append(status)
            continue

        values.append(float(val))
        errors.append(float(err or 0.0))
        statuses.append(status)

    return {
        "labels": labels,
        "values": values,
        "errors": errors,
        "statuses": statuses,
    }


def _configure_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10


def create_figure(series: Dict[str, List[float]], out_path: Path) -> None:
    _configure_style()
    methods = series["labels"]
    auroc = series["values"]
    yerr = series["errors"]

    colors = [
        "#CCCCCC",
        "#CCCCCC",
        "#CCCCCC",
        "#CCCCCC",
        "#CCCCCC",
        "white",
        "#87CEEB",
        "#90EE90",
        "white",
        "#2C5F7F",
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(methods))
    bars = ax.bar(
        x,
        auroc,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.7,
        yerr=yerr,
        capsize=3,
        error_kw={"linewidth": 1.2, "color": "#444444"},
    )

    bars[-1].set_linewidth(2.5)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.5)

    for i, (method, val, err) in enumerate(zip(methods, auroc, yerr)):
        if not method or np.isnan(val):
            continue
        fontweight = "bold" if i == len(methods) - 1 else "normal"
        fontsize = 11 if i == len(methods) - 1 else 9
        if val > 0.90:
            y_pos = val - 0.03
            va = "top"
            color = "white"
        else:
            y_pos = val + err + 0.02
            va = "bottom"
            color = "black"
        ax.text(
            i,
            y_pos,
            f"{val:.2f}",
            ha="center",
            va=va,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
        )

    ax.set_ylabel("AUROC", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Temporal OOD Detection on FB15k-237", fontsize=13, fontweight="bold", pad=15)

    from matplotlib.patches import Patch

    legend = [
        Patch(facecolor="#CCCCCC", label="Probabilistic baselines"),
        Patch(facecolor="#87CEEB", label="Semantic (entity variance)"),
        Patch(facecolor="#90EE90", label="Structural (coverage)"),
        Patch(facecolor="#2C5F7F", linewidth=2.5, label="CAGP (ours)"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.95)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    png_path = out_path.with_suffix(".png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Figure 1 from canonical metrics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="Path to canonical paper metrics JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output PDF path.",
    )
    args = parser.parse_args()

    payload = _load_metrics(args.metrics)
    series = build_figure_series(payload, dataset="fb15k237")
    create_figure(series, args.out)


if __name__ == "__main__":
    main()
