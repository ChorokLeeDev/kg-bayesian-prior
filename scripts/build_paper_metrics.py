#!/usr/bin/env python3
"""
Build a canonical paper metrics artifact with explicit provenance.

This script intentionally avoids `canonical_temporal_results_v2.json` because that
artifact can contain patched summary values that do not match per-seed values.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

SEEDS = [42, 123, 456]
DATASETS = ["wn18rr", "fb15k237", "yago", "icews14"]

METHODS = [
    "UKGE",
    "Energy",
    "MCDropout",
    "DeepEnsemble",
    "SNGP",
    "GPOnly",
    "CoverageOnly",
    "CAGP",
    "RelCondVar",
]

TEMPORAL_METRICS = [
    "emerging_auroc",
    "novel_ctx_auroc",
    "overall_auroc",
    "overall_aupr",
]

MISSING_BASELINE_METHODS = {"MCDropout", "DeepEnsemble", "SNGP"}

YAGO_PER_SEED_METHOD_TO_STEM = {
    "UKGE": "ukge",
    "Energy": "energy",
    "GPOnly": "gponly",
    "CoverageOnly": "coverageonly",
    "CAGP": "cagp",
    "RelCondVar": "relcondvar",
}


def _load_json(path: Path, retries: int = 5, backoff_seconds: float = 0.05) -> dict:
    """
    Read JSON with short retries to tolerate concurrent non-atomic writers.
    """
    last_exc: Optional[Exception] = None
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


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values))


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _round2(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(f"{value:.2f}")


def _temporal_blob(blob: dict) -> dict:
    if "temporal" in blob and isinstance(blob["temporal"], dict):
        return blob["temporal"]
    return blob


def _collect_from_seed_dict(
    seed_dict: dict,
    method: str,
    source_file: Path,
    source_kind: str,
    provenance_tag: str,
) -> dict:
    seed_values = {metric: {} for metric in TEMPORAL_METRICS}
    for seed in SEEDS:
        seed_key = f"seed_{seed}"
        method_blob = seed_dict.get(seed_key, {}).get(method)
        if method_blob is None:
            continue
        t_blob = _temporal_blob(method_blob)
        for metric in TEMPORAL_METRICS:
            if metric in t_blob and t_blob[metric] is not None:
                seed_values[metric][str(seed)] = float(t_blob[metric])

    return _build_metric_block(
        seed_values=seed_values,
        source_files=[str(source_file)],
        source_kind=source_kind,
        provenance_tag=provenance_tag,
    )


def _build_metric_block(
    seed_values: Dict[str, Dict[str, float]],
    source_files: List[str],
    source_kind: str,
    provenance_tag: str,
    status_override: Optional[str] = None,
    allow_lt3_seeds: bool = False,
) -> dict:
    metrics_out = {}
    for metric in TEMPORAL_METRICS:
        per_seed = seed_values.get(metric, {})
        seeds_present = sorted(int(s) for s in per_seed.keys())
        values = [per_seed[str(s)] for s in seeds_present]
        status = status_override
        if status is None:
            status = "ok" if values else "not_evaluated"
        metrics_out[metric] = {
            "status": status,
            "value": _mean(values) if values else None,
            "std": _std(values) if values else None,
            "n_seeds": len(values),
            "seeds": seeds_present,
            "seed_values": {str(s): per_seed[str(s)] for s in seeds_present},
            "source_files": source_files,
            "source_kind": source_kind,
            "provenance_tag": provenance_tag,
            "allow_lt3_seeds": allow_lt3_seeds,
        }
    return metrics_out


def _collect_from_fixed_multiseed(
    path: Path,
    array_key: str,
    source_kind: str,
    provenance_tag: str,
) -> dict:
    data = _load_json(path)
    rows = data.get(array_key, [])
    seed_values = {metric: {} for metric in TEMPORAL_METRICS}
    for row in rows:
        seed = int(row["seed"])
        for metric in TEMPORAL_METRICS:
            if metric in row and row[metric] is not None:
                seed_values[metric][str(seed)] = float(row[metric])
    return _build_metric_block(
        seed_values=seed_values,
        source_files=[str(path)],
        source_kind=source_kind,
        provenance_tag=provenance_tag,
    )


def _load_temporal_dataset_blob(dataset: str) -> Tuple[dict, List[str]]:
    """
    Return dataset blob and list of source files used.
    """
    candidates: List[Path] = []
    if dataset == "wn18rr":
        candidates = [OUTPUTS / "wn18rr_temporal_results.json"]
    elif dataset == "fb15k237":
        # Primary file is dataset-specific. Fallback is combined file that includes
        # both wn18rr and fb15k237.
        candidates = [
            OUTPUTS / "fb15k237_temporal_results.json",
            OUTPUTS / "wn18rr_temporal_results.json",
        ]
    elif dataset == "icews14":
        candidates = [OUTPUTS / "icews14_temporal_results.json"]
    else:
        return {}, []

    for candidate in candidates:
        if not candidate.exists():
            continue
        data = _load_json(candidate)
        if dataset in data and isinstance(data[dataset], dict):
            return data[dataset], [str(candidate)]
        # ICEWS file can be top-level seed map.
        if dataset == "icews14" and "seed_42" in data:
            return data, [str(candidate)]
    return {}, [str(c) for c in candidates if c.exists()]


def _collect_from_temporal_results(
    dataset: str,
    method: str,
    provenance_tag: str,
) -> dict:
    ds_blob, source_files = _load_temporal_dataset_blob(dataset)
    if not ds_blob:
        return _build_metric_block(
            seed_values={metric: {} for metric in TEMPORAL_METRICS},
            source_files=source_files,
            source_kind="temporal_results",
            provenance_tag=provenance_tag,
            status_override="not_evaluated",
            allow_lt3_seeds=True,
        )
    return _collect_from_seed_dict(
        seed_dict=ds_blob,
        method=method,
        source_file=Path(source_files[0]),
        source_kind="temporal_results",
        provenance_tag=provenance_tag,
    )


def _collect_from_missing_baselines(dataset: str, method: str) -> dict:
    path = OUTPUTS / f"{dataset}_missing_baselines.json"
    if not path.exists():
        return _build_metric_block(
            seed_values={metric: {} for metric in TEMPORAL_METRICS},
            source_files=[str(path)],
            source_kind="missing_baselines",
            provenance_tag=f"{dataset}_missing_baselines",
            status_override="not_evaluated",
            allow_lt3_seeds=True,
        )
    metrics = _collect_from_seed_dict(
        seed_dict=_load_json(path),
        method=method,
        source_file=path,
        source_kind="missing_baselines",
        provenance_tag=f"{dataset}_missing_baselines",
    )

    # Missing-baseline files are written incrementally during long runs.
    # Treat partially filled methods as "not_evaluated" so canonical tables and
    # provenance checks remain stable until all 3 seeds are present.
    complete = all(
        int(metrics[metric]["n_seeds"]) == len(SEEDS) for metric in TEMPORAL_METRICS
    )
    if not complete:
        for metric in TEMPORAL_METRICS:
            metrics[metric]["status"] = "not_evaluated"
            metrics[metric]["value"] = None
            metrics[metric]["std"] = None
            metrics[metric]["allow_lt3_seeds"] = True
            metrics[metric]["provenance_tag"] = f"{dataset}_missing_baselines_incomplete"
    return metrics


def _collect_from_yago_per_seed(method: str) -> dict:
    stem = YAGO_PER_SEED_METHOD_TO_STEM[method]
    pattern = str(OUTPUTS / f"yago_temporal_{stem}_seed*.json")
    paths = sorted(Path(p) for p in glob.glob(pattern))
    seed_values = {metric: {} for metric in TEMPORAL_METRICS}
    for path in paths:
        data = _load_json(path)
        seed = int(data["seed"])
        t_blob = _temporal_blob(data)
        for metric in TEMPORAL_METRICS:
            if metric in t_blob and t_blob[metric] is not None:
                seed_values[metric][str(seed)] = float(t_blob[metric])
    return _build_metric_block(
        seed_values=seed_values,
        source_files=[str(p) for p in paths],
        source_kind="per_seed_files",
        provenance_tag="yago_per_seed_temporal",
        status_override="ok" if paths else "not_evaluated",
        allow_lt3_seeds=not bool(paths),
    )


def _collect_for_dataset_method(dataset: str, method: str) -> dict:
    # Precedence rules from the implementation plan.
    if dataset in {"wn18rr", "fb15k237"} and method in {"CAGP", "CoverageOnly"}:
        path = OUTPUTS / f"{dataset}_fixed_cagp_multiseed.json"
        key = "fixed_cagp" if method == "CAGP" else "coverage_only"
        if path.exists():
            primary = _collect_from_fixed_multiseed(
                path=path,
                array_key=key,
                source_kind="fixed_multiseed",
                provenance_tag=f"{dataset}_fixed_multiseed",
            )
            # Fixed multiseed files intentionally store AUROC metrics only; fill
            # missing AUPR from temporal results with explicit mixed provenance.
            fallback = _collect_from_temporal_results(
                dataset=dataset,
                method=method,
                provenance_tag=f"{dataset}_temporal_results",
            )
            for metric in TEMPORAL_METRICS:
                if primary[metric]["status"] != "ok" and fallback[metric]["status"] == "ok":
                    primary[metric] = fallback[metric]
                    primary[metric]["provenance_tag"] = (
                        f"{dataset}_fixed_plus_temporal_backfill"
                    )
            return primary

    if dataset == "yago" and method in {"CAGP", "CoverageOnly"}:
        path = OUTPUTS / "yago310_fixed_cagp_multiseed.json"
        key = "fixed_cagp" if method == "CAGP" else "coverage_only"
        if path.exists():
            primary = _collect_from_fixed_multiseed(
                path=path,
                array_key=key,
                source_kind="fixed_multiseed",
                provenance_tag="yago_fixed_multiseed",
            )
            fallback = _collect_from_yago_per_seed(method)
            for metric in TEMPORAL_METRICS:
                if primary[metric]["status"] != "ok" and fallback[metric]["status"] == "ok":
                    primary[metric] = fallback[metric]
                    primary[metric]["provenance_tag"] = "yago_fixed_plus_per_seed_backfill"
            return primary

    if dataset == "yago" and method in YAGO_PER_SEED_METHOD_TO_STEM:
        return _collect_from_yago_per_seed(method)

    if method in MISSING_BASELINE_METHODS:
        dataset_key = "icews14" if dataset == "icews14" else dataset
        return _collect_from_missing_baselines(dataset_key, method)

    # Remaining methods come from temporal result files.
    provenance = f"{dataset}_temporal_results"
    return _collect_from_temporal_results(dataset=dataset, method=method, provenance_tag=provenance)


def _summary_consistency_checks(result: dict, tolerance: float = 1e-9) -> List[str]:
    """
    Validate metric "value" equals recomputed mean(seed_values) exactly (within tolerance).
    """
    issues: List[str] = []
    for dataset, ds_blob in result["datasets"].items():
        for method, method_blob in ds_blob["methods"].items():
            for metric in TEMPORAL_METRICS:
                m = method_blob["metrics"][metric]
                if m["status"] != "ok":
                    continue
                seed_values = [float(v) for v in m["seed_values"].values()]
                if not seed_values:
                    continue
                recomputed = _mean(seed_values)
                if not math.isclose(recomputed, float(m["value"]), rel_tol=0.0, abs_tol=tolerance):
                    issues.append(
                        f"{dataset}/{method}/{metric}: value={m['value']} "
                        f"!= mean(seed_values)={recomputed}"
                    )
    return issues


def _build_paper_summary(datasets_blob: dict) -> dict:
    summary = {
        "temporal_ood": {},
        "complementarity": {},
        "aupr": {},
        "not_evaluated": {},
    }
    for dataset, ds_blob in datasets_blob.items():
        summary["temporal_ood"][dataset] = {}
        summary["complementarity"][dataset] = {}
        summary["aupr"][dataset] = {}
        summary["not_evaluated"][dataset] = {}
        for method, method_blob in ds_blob["methods"].items():
            metrics = method_blob["metrics"]
            summary["temporal_ood"][dataset][method] = {
                "emerging_auroc": _round2(metrics["emerging_auroc"]["value"]),
                "overall_auroc": _round2(metrics["overall_auroc"]["value"]),
                "emerging_auroc_std": _round2(metrics["emerging_auroc"]["std"]),
                "overall_auroc_std": _round2(metrics["overall_auroc"]["std"]),
                "status": metrics["overall_auroc"]["status"],
            }
            summary["complementarity"][dataset][method] = {
                "emerging_auroc": _round2(metrics["emerging_auroc"]["value"]),
                "novel_ctx_auroc": _round2(metrics["novel_ctx_auroc"]["value"]),
                "overall_auroc": _round2(metrics["overall_auroc"]["value"]),
                "status": metrics["overall_auroc"]["status"],
            }
            summary["aupr"][dataset][method] = {
                "overall_aupr": _round2(metrics["overall_aupr"]["value"]),
                "overall_aupr_std": _round2(metrics["overall_aupr"]["std"]),
                "status": metrics["overall_aupr"]["status"],
            }
            if metrics["overall_auroc"]["status"] != "ok":
                summary["not_evaluated"][dataset][method] = {
                    "reason": "missing_or_incomplete_source_data",
                    "sources": metrics["overall_auroc"]["source_files"],
                }
    return summary


def _collect_link_prediction() -> dict:
    path = OUTPUTS / "link_prediction_eval.json"
    if not path.exists():
        return {"status": "not_available", "source_file": str(path)}
    data = _load_json(path)
    out = {"status": "ok", "source_file": str(path), "datasets": {}}
    for dataset in ["fb15k237", "wn18rr"]:
        if dataset not in data:
            continue
        ds = data[dataset]
        out["datasets"][dataset] = {
            "vanilla": {
                "mrr": float(ds["vanilla"]["mrr"]),
                "hits@10": float(ds["vanilla"]["hits@10"]),
                "hits@1": float(ds["vanilla"]["hits@1"]),
            },
            "cagp": {
                "mrr": float(ds["cagp"]["mrr"]),
                "hits@10": float(ds["cagp"]["hits@10"]),
                "hits@1": float(ds["cagp"]["hits@1"]),
            },
        }
    return out


def build(strict: bool) -> dict:
    datasets_blob = {}
    for dataset in DATASETS:
        ds_methods = {}
        for method in METHODS:
            ds_methods[method] = {
                "metrics": _collect_for_dataset_method(dataset, method),
            }
        datasets_blob[dataset] = {"methods": ds_methods}

    payload = {
        "meta": {
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "policy": "reproducibility_over_legacy",
            "deprecated_sources": [str(OUTPUTS / "canonical_temporal_results_v2.json")],
            "strict": strict,
        },
        "datasets": datasets_blob,
        "paper_summary": _build_paper_summary(datasets_blob),
        "link_prediction": _collect_link_prediction(),
    }

    issues = _summary_consistency_checks(payload)
    payload["validation"] = {
        "issues": issues,
        "n_issues": len(issues),
    }

    if strict and issues:
        raise RuntimeError(
            "Strict mode failed consistency checks:\n- " + "\n- ".join(issues)
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical paper metrics artifact.")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUTS / "paper_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if internal consistency checks fail.",
    )
    args = parser.parse_args()

    result = build(strict=args.strict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Wrote canonical metrics to {args.out}")
    print(f"Consistency issues: {result['validation']['n_issues']}")


if __name__ == "__main__":
    main()
