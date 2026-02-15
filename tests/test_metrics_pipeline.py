import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from create_fig1_minimal import build_figure_series  # noqa: E402
from src.ranking import compute_rank_from_scores  # noqa: E402


class MetricsPipelineTest(unittest.TestCase):
    def _run(self, *cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

    def test_metrics_consistency_and_seed_mean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "paper_metrics.json"
            self._run(
                sys.executable,
                str(SCRIPTS / "build_paper_metrics.py"),
                "--out",
                str(out),
            )
            payload = json.loads(out.read_text())

            for ds_blob in payload["datasets"].values():
                for method_blob in ds_blob["methods"].values():
                    metrics = method_blob["metrics"]
                    for metric_blob in metrics.values():
                        if metric_blob["status"] != "ok":
                            continue
                        seed_values = list(metric_blob["seed_values"].values())
                        if not seed_values:
                            continue
                        recomputed = float(np.mean(seed_values))
                        self.assertAlmostEqual(recomputed, metric_blob["value"], places=12)

    def test_ranking_uses_strict_greater_than(self) -> None:
        scores = np.array([0.8, 0.8, 0.7], dtype=float)
        self.assertEqual(compute_rank_from_scores(scores, 0), 1)
        self.assertEqual(compute_rank_from_scores(scores, 2), 3)

    def test_figure_series_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "paper_metrics.json"
            self._run(
                sys.executable,
                str(SCRIPTS / "build_paper_metrics.py"),
                "--out",
                str(out),
            )
            payload = json.loads(out.read_text())
            first = build_figure_series(payload, dataset="fb15k237")
            second = build_figure_series(payload, dataset="fb15k237")
            self.assertEqual(first["labels"], second["labels"])
            self.assertEqual(first["values"], second["values"])
            self.assertEqual(first["errors"], second["errors"])
            self.assertEqual(first["statuses"], second["statuses"])

    def test_validator_cross_checks_paper_tables(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "paper_metrics.json"
            self._run(
                sys.executable,
                str(SCRIPTS / "build_paper_metrics.py"),
                "--out",
                str(out),
            )
            self._run(
                sys.executable,
                str(SCRIPTS / "validate_metrics_provenance.py"),
                "--metrics",
                str(out),
                "--check-paper",
            )


if __name__ == "__main__":
    unittest.main()
