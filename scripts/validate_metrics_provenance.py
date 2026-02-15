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
    # Restrict parsing to the temporal table to avoid collisions with similarly
    # named rows in other tables.
    table_match = re.search(
        r"\\label\{tab:temporal_ood\}(.*?)\\end\{table\}", tex, flags=re.DOTALL
    )
    if table_match:
        tex = table_match.group(1)

    # Remove tiny std annotations to make number parsing stable.
    tex = re.sub(r"\\tiny\{[^}]*\}", "", tex)
    # Collapse multiline rows like method + continuation line.
    tex = re.sub(r"\n\s+&", " &", tex)

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
        parsed = False
        for m in row_re.finditer(tex):
            row = m.group(1)
            cells = [c.strip() for c in row.split("&")]
            values = []
            for cell in cells:
                if cell == "---":
                    values.append(None)
                    continue
                num_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", cell)
                values.append(
                    _parse_tex_number(num_match.group(1)) if num_match else None
                )
            # Temporal table must have 4 datasets x 2 metrics = 8 values.
            if len(values) != 8:
                continue
            for i, dataset in enumerate(datasets):
                em = values[2 * i]
                overall = values[2 * i + 1]
                if em is not None:
                    out[dataset].setdefault(method, {})["emerging_auroc"] = em
                if overall is not None:
                    out[dataset].setdefault(method, {})["overall_auroc"] = overall
            parsed = True
            break
        if not parsed:
            continue
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
        parsed = False
        for m in row_re.finditer(main_tex):
            row = m.group(1)
            cells = [c.strip() for c in row.split("&")]
            # AUPR table has 3 dataset columns.
            if len(cells) != 3:
                continue
            for dataset, cell in zip(datasets, cells):
                if cell == "---":
                    continue
                num_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", cell)
                if not num_match:
                    continue
                out[dataset][method] = _parse_tex_number(num_match.group(1))
            parsed = True
            break
        if not parsed:
            continue
    return out


def _extract_standard_table_fb_values(exp_tex: str) -> Dict[str, float]:
    """
    Parse Table `tab:standard` in experiments_uai.tex and return FB15k-237 AUROC
    per method.
    """
    table_match = re.search(
        r"\\label\{tab:standard\}(.*?)\\end\{table\}", exp_tex, flags=re.DOTALL
    )
    if not table_match:
        return {}
    table_tex = table_match.group(1)
    table_tex = re.sub(r"\{\\scriptsize[^}]*\}", "", table_tex)
    table_tex = table_tex.replace("\n", "\n")

    method_map = {
        "UKGE": "UKGE",
        "Energy": "Energy",
        "$U_{\\text{sem}}$": "GPOnly",
        "$U_{\\text{str}}$": "CoverageOnly",
        "CAGP": "CAGP",
        "RelCondVar": "RelCondVar",
    }
    out: Dict[str, float] = {}
    for raw_label, method in method_map.items():
        row_re = re.compile(
            rf"^{re.escape(raw_label)}\s*&\s*([^&]+?)\s*&\s*([^\\\\]+?)\\\\",
            flags=re.MULTILINE,
        )
        m = row_re.search(table_tex)
        if not m:
            continue
        fb_cell = m.group(2).strip()
        num_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", fb_cell)
        if not num_match:
            continue
        out[method] = _parse_tex_number(num_match.group(1))
    return out


def _extract_method_comparison_values(exp_tex: str) -> Dict[str, Dict[str, float]]:
    """
    Parse Table `tab:method_comparison` in experiments_uai.tex and return
    per-method standard-vs-temporal AUROC entries on FB15k-237.
    """
    table_match = re.search(
        r"\\label\{tab:method_comparison\}(.*?)\\end\{table\}",
        exp_tex,
        flags=re.DOTALL,
    )
    if not table_match:
        return {}
    table_tex = table_match.group(1)
    table_tex = re.sub(r"\{\\scriptsize[^}]*\}", "", table_tex)
    table_tex = table_tex.replace("\n           &", " &")

    method_map = {
        "UKGE": "UKGE",
        "Energy": "Energy",
        "$U_{\\text{str}}$ (structural)": "CoverageOnly",
        "CAGP": "CAGP",
        "RelCondVar": "RelCondVar",
    }
    out: Dict[str, Dict[str, float]] = {}
    for raw_label, method in method_map.items():
        row_re = re.compile(
            rf"^{re.escape(raw_label)}\s*&\s*([^&]+?)\s*&\s*([^\\\\]+?)\\\\",
            flags=re.MULTILINE,
        )
        m = row_re.search(table_tex)
        if not m:
            continue
        std_cell = m.group(1).strip()
        tmp_cell = m.group(2).strip()
        std_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", std_cell)
        tmp_match = re.search(r"(-?\d+(?:\.\d+)?|\.\d+)", tmp_cell)
        if not std_match or not tmp_match:
            continue
        out[method] = {
            "standard_auroc": _parse_tex_number(std_match.group(1)),
            "temporal_auroc": _parse_tex_number(tmp_match.group(1)),
        }
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

    # Ensure the FB15k-237 "standard vs temporal" comparison table is internally
    # consistent with the canonical standard table (standard column) and
    # canonical temporal table (temporal column).
    standard_fb_vals = _extract_standard_table_fb_values(exp_tex)
    method_cmp_vals = _extract_method_comparison_values(exp_tex)
    fb_temporal_vals = temporal_tex_vals.get("fb15k237", {})
    for method, mm in method_cmp_vals.items():
        std_val = mm["standard_auroc"]
        tmp_val = mm["temporal_auroc"]

        std_ref = standard_fb_vals.get(method)
        if std_ref is None:
            issues.append(
                f"method comparison table has {method} standard={std_val} "
                "but no reference in tab:standard"
            )
        elif not math.isclose(float(std_ref), float(std_val), abs_tol=0.01):
            issues.append(
                f"method comparison mismatch standard FB15k-237/{method}: "
                f"table={std_val}, tab:standard={std_ref}"
            )

        tmp_ref = fb_temporal_vals.get(method, {}).get("overall_auroc")
        if tmp_ref is None:
            issues.append(
                f"method comparison table has {method} temporal={tmp_val} "
                "but no reference in tab:temporal_ood"
            )
        elif not math.isclose(float(tmp_ref), float(tmp_val), abs_tol=0.01):
            issues.append(
                f"method comparison mismatch temporal FB15k-237/{method}: "
                f"table={tmp_val}, tab:temporal_ood={tmp_ref}"
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
