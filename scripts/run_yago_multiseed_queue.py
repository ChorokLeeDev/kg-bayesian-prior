#!/usr/bin/env python3
"""Run YAGO temporal jobs with bounded concurrency to avoid system freeze."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_MODELS = ["UKGE", "Energy", "GPOnly", "CoverageOnly", "CAGP", "RelCondVar"]
DEFAULT_SEEDS = [123, 456]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch YAGO temporal runs with limited parallel workers."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Model list to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epoch count for each run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum number of concurrent runs (recommended: 2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs even if output JSON already exists.",
    )
    return parser.parse_args()


def run_one(
    project_root: Path,
    model: str,
    seed: int,
    epochs: int,
    log_dir: Path,
) -> tuple[str, int, int, float]:
    output_path = project_root / "outputs" / f"yago_temporal_{model.lower()}_seed{seed}.json"
    log_path = log_dir / f"yago_temporal_{model.lower()}_seed{seed}.log"

    cmd = [
        sys.executable,
        "-m",
        "scripts.run_yago_single_model",
        "--model",
        model,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
    ]

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# Command: {' '.join(cmd)}\n")
        log_file.write(f"# Output JSON: {output_path}\n\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            cwd=project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start
    return model, seed, result.returncode, elapsed


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, int]] = []
    for seed in args.seeds:
        for model in args.models:
            output_path = output_dir / f"yago_temporal_{model.lower()}_seed{seed}.json"
            if output_path.exists() and not args.force:
                print(f"SKIP existing: model={model}, seed={seed} ({output_path.name})")
                continue
            jobs.append((model, seed))

    if not jobs:
        print("No jobs to run.")
        return 0

    print(
        f"Launching {len(jobs)} jobs with workers={args.workers}, epochs={args.epochs}. "
        f"Logs: {log_dir}"
    )

    failures: list[tuple[str, int, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(run_one, project_root, model, seed, args.epochs, log_dir): (model, seed)
            for model, seed in jobs
        }
        for future in as_completed(future_map):
            model, seed = future_map[future]
            try:
                _, _, code, elapsed = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((model, seed, -1))
                print(f"FAIL model={model}, seed={seed}, error={exc}")
                continue

            if code == 0:
                print(f"DONE model={model}, seed={seed}, elapsed={elapsed/60:.1f}m")
            else:
                failures.append((model, seed, code))
                print(f"FAIL model={model}, seed={seed}, code={code}, elapsed={elapsed/60:.1f}m")

    if failures:
        print("\nFailed jobs:")
        for model, seed, code in failures:
            print(f"- model={model}, seed={seed}, code={code}")
        return 1

    print("All jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
