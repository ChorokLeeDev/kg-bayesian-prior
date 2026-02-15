#!/usr/bin/env python3
"""
Validate provenance and consistency constraints for paper_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = ROOT / "outputs" / "paper_metrics.json"

TEMPORAL_METRICS = [
    "emerging_auroc",
    "novel_ctx_auroc",
    "overall_auroc",
    "overall_aupr",
]


def _load_json(path: Path, retries: int = 5, backoff_seconds: float = 0.05) -> dict:
    """
    Read JSON with short retries to tolerate concurrent non-atomic writers.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            with path.open() as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt == retries - 1:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to load JSON from {path}")


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values))


def _check_metrics_consistency(payload: dict, tolerance: float = 1e-9) -> List[str]:
    issues: List[str] = []
    for dataset, ds_blob in payload.get("datasets", {}).items():
        for method, method_blob in ds_blob.get("methods", {}).items():
            metrics = method_blob.get("metrics", {})
            for metric in TEMPORAL_METRICS:
                m = metrics.get(metric)
                if not m:
                    issues.append(f"Missing metric block: {dataset}/{method}/{metric}")
                    continue

                # Check 1: mean(seed_values) == value
                if m.get("status") == "ok":
                    seed_values_map = m.get("seed_values", {})
                    seed_values = [float(v) for v in seed_values_map.values()]
                    if not seed_values:
                        issues.append(
                            f"{dataset}/{method}/{metric}: status=ok but no seed values"
                        )
                    else:
                        recomputed = _mean(seed_values)
                        stored = float(m["value"])
                        if not math.isclose(
                            recomputed, stored, rel_tol=0.0, abs_tol=tolerance
                        ):
                            issues.append(
                                f"{dataset}/{method}/{metric}: stored value {stored} "
                                f"!= recomputed mean {recomputed}"
                            )

                    # Check 2: n_seeds requirement
                    n_seeds = int(m.get("n_seeds", 0))
                    allow_lt3 = bool(m.get("allow_lt3_seeds", False))
                    if n_seeds < 3 and not allow_lt3:
                        issues.append(
                            f"{dataset}/{method}/{metric}: n_seeds={n_seeds} < 3 without explicit allow_lt3_seeds"
                        )

                # Check 3: mixed-source override requires provenance tag
                source_files = m.get("source_files", [])
                provenance_tag = m.get("provenance_tag")
                if len(source_files) > 1 and not provenance_tag:
                    issues.append(
                        f"{dataset}/{method}/{metric}: mixed source_files without provenance_tag"
                    )

    return issues


def _parse_tex_number(token: str) -> float:
    token = token.strip()
    if token.startswith("."):
        token = "0" + token
    return float(token)


def _extract_temporal_table_values(tex: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Parse temporal table rows in paper/sections/experiments_uai.tex.
    Returns dataset->method->{emerging_auroc, overall_auroc}
    """
    # Remove tiny std annotations to make number parsing stable.
    tex = re.sub(r"\\tiny\{[^}]*\}", "", tex)
    # Collapse multiline rows like method + continuation line.
    tex = tex.replace("\n           &", " &")

    method_map = {
        "UKGE": "UKGE",
        "Energy": "Energy",
        "MC Drop.": "MCDropout",
        "Deep Ens.": "DeepEnsemble",
        "SNGP": "SNGP",
        "$U_{\\text{sem}}$": "GPOnly",
        "$U_{\\text{str}}$": "CoverageOnly",
        "CAGP": "CAGP",
        "RelCondVar": "RelCondVar",
    }
    datasets = ["wn18rr", "fb15k237", "yago", "icews14"]

    out: Dict[str, Dict[str, Dict[str, float]]] = {d: {} for d in datasets}
    for raw_label, method in method_map.items():
        row_re = re.compile(rf"^{re.escape(raw_label)}\s*&(.+?)\\\\", re.MULTILINE)
        m = row_re.search(tex)
        if not m:
            continue
        row = m.group(1)
        cells = [c.strip() for c in row.split("&")]
        values = []
        for cell in cells:
            if cell == "---":
                values.append(None)
                continue
            num_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", cell)
            values.append(_parse_tex_number(num_match.group(1)) if num_match else None)
        if len(values) != 8:
            continue
        for i, dataset in enumerate(datasets):
            em = values[2 * i]
            overall = values[2 * i + 1]
            if em is not None:
                out[dataset].setdefault(method, {})["emerging_auroc"] = em
            if overall is not None:
                out[dataset].setdefault(method, {})["overall_auroc"] = overall
    return out


def _extract_aupr_table_values(main_tex: str) -> Dict[str, Dict[str, float]]:
    """
    Parse AUPR appendix table in paper/main.tex.
    Returns dataset->method->overall_aupr.
    """
    method_map = {
        "UKGE": "UKGE",
        "Energy": "Energy",
        "MC Dropout": "MCDropout",
        "Deep Ensemble": "DeepEnsemble",
        "SNGP": "SNGP",
        "$U_{\\text{sem}}$": "GPOnly",
        "$U_{\\text{str}}$": "CoverageOnly",
        "CAGP": "CAGP",
    }
    datasets = ["wn18rr", "fb15k237", "icews14"]
    out: Dict[str, Dict[str, float]] = {d: {} for d in datasets}

    for raw_label, method in method_map.items():
        row_re = re.compile(rf"^{re.escape(raw_label)}\s*&(.+?)\\\\", re.MULTILINE)
        m = row_re.search(main_tex)
        if not m:
            continue
        row = m.group(1)
        cells = [c.strip() for c in row.split("&")]
        if len(cells) != 3:
            continue
        for dataset, cell in zip(datasets, cells):
            if cell == "---":
                continue
            num_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", cell)
            if not num_match:
                continue
            out[dataset][method] = _parse_tex_number(num_match.group(1))
    return out


def _check_paper_cross_reference(metrics: dict, root: Path) -> List[str]:
    """
    Ensure table values in paper are present in paper_metrics summary.
    """
    issues: List[str] = []

    exp_path = root / "paper" / "sections" / "experiments_uai.tex"
    main_path = root / "paper" / "main.tex"
    exp_tex = exp_path.read_text()
    main_tex = main_path.read_text()

    temporal_tex_vals = _extract_temporal_table_values(exp_tex)
    temporal_metrics = metrics["paper_summary"]["temporal_ood"]
    for dataset, ds_vals in temporal_tex_vals.items():
        for method, mm in ds_vals.items():
            for key in ["emerging_auroc", "overall_auroc"]:
                tex_val = mm.get(key)
                if tex_val is None:
                    continue
                metric_val = temporal_metrics.get(dataset, {}).get(method, {}).get(key)
                if metric_val is None:
                    issues.append(
                        f"paper temporal table has {dataset}/{method}/{key}={tex_val} "
                        f"but metrics map has no value"
                    )
                    continue
                if not math.isclose(float(metric_val), float(tex_val), abs_tol=0.01):
                    issues.append(
                        f"paper temporal table mismatch {dataset}/{method}/{key}: "
                        f"paper={tex_val}, metrics={metric_val}"
                    )

    aupr_tex_vals = _extract_aupr_table_values(main_tex)
    aupr_metrics = metrics["paper_summary"]["aupr"]
    for dataset, ds_vals in aupr_tex_vals.items():
        for method, tex_val in ds_vals.items():
            metric_val = aupr_metrics.get(dataset, {}).get(method, {}).get("overall_aupr")
            if metric_val is None:
                issues.append(
                    f"paper AUPR table has {dataset}/{method}={tex_val} "
                    f"but metrics map has no value"
                )
                continue
            if not math.isclose(float(metric_val), float(tex_val), abs_tol=0.01):
                issues.append(
                    f"paper AUPR table mismatch {dataset}/{method}: "
                    f"paper={tex_val}, metrics={metric_val}"
                )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paper metrics provenance.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help="Path to paper_metrics.json",
    )
    parser.add_argument(
        "--check-paper",
        action="store_true",
        help="Cross-check values in paper tables against metrics summary.",
    )
    args = parser.parse_args()

    payload = _load_json(args.metrics)
    issues = _check_metrics_consistency(payload)
    if args.check_paper:
        issues.extend(_check_paper_cross_reference(payload, ROOT))

    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print(f"Validation passed for {args.metrics}")


if __name__ == "__main__":
    main()
