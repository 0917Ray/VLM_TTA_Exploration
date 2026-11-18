from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_alignment = importlib.import_module("8_evaluate_tpt_gradient_alignment")
assign_jobs = _alignment.assign_jobs
metric_stats = _alignment.metric_stats
summarize_records = _alignment.summarize_records
vector_metrics = _alignment.vector_metrics


class GradientMetricTest(unittest.TestCase):
    def test_vector_metrics_separate_raw_and_optimizer_updates(self):
        result = vector_metrics(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 2.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, -1.0]),
            1e-12,
        )
        self.assertEqual(result["raw_norm_T"], 1.0)
        self.assertEqual(result["raw_norm_C"], 2.0)
        self.assertAlmostEqual(result["raw_cosine"], 0.0)
        self.assertAlmostEqual(result["raw_dot"], 0.0)
        self.assertAlmostEqual(result["update_cosine"], 0.0)
        self.assertAlmostEqual(result["update_dot"], 0.0)

    def test_metric_stats_reports_sample_std_iqr_and_negative_ratio(self):
        result = metric_stats([-2.0, -1.0, 1.0, 2.0], "raw_dot")
        self.assertEqual(result["median"], 0.0)
        self.assertEqual(result["negative_count"], 2)
        self.assertEqual(result["negative_ratio"], 0.5)
        self.assertEqual(result["valid_count"], 4)
        self.assertAlmostEqual(result["standard_deviation"], 1.825741858)
        self.assertAlmostEqual(result["iqr"], 2.5)

    def test_summary_keeps_experiment_combinations_separate(self):
        base = {
            "sample_index": 0,
            "initial_correct": True,
            "raw_norm_T": 1.0,
            "raw_norm_C": 2.0,
            "raw_cosine": 0.5,
            "raw_dot": 1.0,
            "update_norm_T": 0.1,
            "update_norm_C": 0.2,
            "update_cosine": 0.25,
            "update_dot": 0.01,
        }
        records = [
            {**base, "model": "ViT-B/16", "dataset": "imagenet-a", "learning_rate": 0.003},
            {**base, "model": "ViT-B/16", "dataset": "imagenet-r", "learning_rate": 0.003},
        ]
        rows = summarize_records(records, "pooled")
        combinations = {(row["dataset"], row["learning_rate"]) for row in rows}
        self.assertEqual(combinations, {("imagenet-a", 0.003), ("imagenet-r", 0.003)})
        self.assertEqual(len(rows), 2 * 3 * 8)

    def test_jobs_are_assigned_round_robin(self):
        jobs = [("m", "d", 0.003, seed) for seed in (1, 2, 3)]
        self.assertEqual(assign_jobs(jobs, ["cuda:0", "cuda:1"]), [jobs[::2], jobs[1::2]])


if __name__ == "__main__":
    unittest.main()
