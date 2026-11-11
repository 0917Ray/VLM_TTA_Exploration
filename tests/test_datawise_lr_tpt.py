from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_datawise_lr_tpt import (  # noqa: E402
    DEFAULTS,
    assign_jobs,
    build_jobs,
    build_summary,
    first_step_adamw_context,
    parse_args,
    write_summary,
)
from evaluate_tpt import accumulate_topk_correct  # noqa: E402


class FirstStepAdamWTest(unittest.TestCase):
    def test_functional_update_matches_torch_adamw(self):
        theta0 = torch.tensor([[0.4, -0.7], [1.2, -0.2]], dtype=torch.float32)
        gradient = torch.tensor([[0.01, -2.0], [0.0, 0.4]], dtype=torch.float32)
        for learning_rate in (0.0, 0.0001, 0.005, 0.02):
            for weight_decay in (0.0, 0.01, 0.05):
                parameter = torch.nn.Parameter(theta0.clone())
                optimizer = torch.optim.AdamW(
                    [parameter], lr=learning_rate, weight_decay=weight_decay
                )
                parameter.grad = gradient.clone()
                optimizer.step()
                expected = first_step_adamw_context(
                    theta0, gradient, learning_rate, weight_decay,
                    float(optimizer.defaults["eps"]),
                )
                torch.testing.assert_close(parameter.detach(), expected)


class TopKAccountingTest(unittest.TestCase):
    def test_returned_top1_matches_accumulated_top1_with_tied_logits(self):
        logits = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
        global_indices = torch.tensor([10, 20, 30, 40, 50])
        counts = [0] * 5
        expected_top1 = int(global_indices[logits.topk(5).indices[0, 0]])
        predictions = accumulate_topk_correct(
            logits, torch.tensor([expected_top1]), global_indices, counts,
        )
        self.assertEqual(counts[0], 1)
        self.assertEqual(int(predictions[0, 0]), expected_top1)


class ConfigurationTest(unittest.TestCase):
    def test_default_plot_config_exists(self):
        self.assertTrue((PROJECT_ROOT / DEFAULTS["plot"]["config"]).is_file())

    def test_fraction_options_resolve_to_actual_counts(self):
        args, _ = parse_args([
            "--config", "configs/evaluation/datawise_lr_tpt.yaml",
            "--devices", "cpu",
            "--views", "64",
            "--selection-fractions", "0.1", "0.25",
        ])
        self.assertEqual(
            args.selection_specs,
            [
                {
                    "selection_mode": "fraction", "selection_value": 0.1,
                    "selection_count": 6, "selection_fraction": 0.1,
                },
                {
                    "selection_mode": "fraction", "selection_value": 0.25,
                    "selection_count": 16, "selection_fraction": 0.25,
                },
            ],
        )

    def test_seed_is_part_of_the_parallel_job_definition(self):
        args, _ = parse_args([
            "--config", "configs/evaluation/datawise_lr_tpt.yaml",
            "--devices", "cpu",
            "--seeds", "11", "42",
            "--selection-counts", "1", "2",
            "--tta-step-options", "1", "2",
        ])
        jobs = build_jobs(args)
        self.assertEqual(len(jobs), 8)
        self.assertEqual([job[0] for job in jobs[:2]], [11, 42])
        assignments = assign_jobs(jobs, ["cuda:0", "cuda:1"])
        self.assertEqual(assignments[0][0][0], 11)
        self.assertEqual(assignments[1][0][0], 42)
        self.assertTrue(all(len(device_jobs) == 4 for device_jobs in assignments))


class SummaryTest(unittest.TestCase):
    @staticmethod
    def result(
        seed: int, top1: float, counts: dict[str, int], selection_count: int = 1
    ) -> dict:
        return {
            "method": "Data-wise LR TPT",
            "idea": "test",
            "model": "RN50",
            "dataset": "imagenet-a",
            "seed": seed,
            "tta_steps": 1,
            "selection_mode": "count",
            "selection_value": selection_count,
            "selection_fraction": selection_count / 64,
            "samples": sum(counts.values()),
            "top1_accuracy": top1,
            "top5_accuracy": 0.8,
            "zero_shot_top1_accuracy": 0.4,
            "zero_shot_top5_accuracy": 0.7,
            "views": 64,
            "selection_count": selection_count,
            "weight_decay": 0.01,
            "average_initial_entropy": 1.0,
            "average_selected_entropy": 0.8,
            "average_entropy_reduction": 0.2,
            "elapsed_minutes": 1.0,
            "lr_selection_counts": counts,
        }

    def test_single_seed_uncertainty_is_undefined(self):
        results = [self.result(42, 0.5, {"0.001": 2, "0.005": 2})]
        selections = [{
            "selection_mode": "count", "selection_value": 1,
            "selection_count": 1, "selection_fraction": 1 / 64,
        }]
        averages, _, distribution = build_summary(
            results, ["RN50"], ["imagenet-a"], [1], selections,
            [0.001, 0.005],
        )
        self.assertIsNone(averages[0]["std_top1_accuracy"])
        self.assertTrue(all(row["selection_percent_std"] is None for row in distribution))

    def test_multi_seed_distribution_uses_seed_level_proportions(self):
        results = [
            self.result(11, 0.5, {"0.001": 1, "0.005": 3}),
            self.result(42, 0.7, {"0.001": 3, "0.005": 1}),
        ]
        selections = [{
            "selection_mode": "count", "selection_value": 1,
            "selection_count": 1, "selection_fraction": 1 / 64,
        }]
        averages, _, distribution = build_summary(
            results, ["RN50"], ["imagenet-a"], [1], selections,
            [0.001, 0.005],
        )
        self.assertAlmostEqual(averages[0]["average_top1_accuracy"], 0.6)
        by_lr = {row["candidate_lr"]: row for row in distribution}
        self.assertAlmostEqual(by_lr[0.001]["selection_percent"], 50.0)
        self.assertAlmostEqual(by_lr[0.005]["selection_percent"], 50.0)

    def test_selection_settings_are_aggregated_separately(self):
        results = [
            self.result(42, 0.25, {"0.001": 4, "0.005": 0}, selection_count=1),
            self.result(42, 0.75, {"0.001": 0, "0.005": 4}, selection_count=2),
        ]
        selections = [
            {
                "selection_mode": "count", "selection_value": count,
                "selection_count": count, "selection_fraction": count / 64,
            }
            for count in (1, 2)
        ]
        averages, _, distribution = build_summary(
            results, ["RN50"], ["imagenet-a"], [1], selections,
            [0.001, 0.005],
        )
        self.assertEqual([row["selection_count"] for row in averages], [1, 2])
        by_setting = {
            (row["selection_count"], row["candidate_lr"]): row["selection_percent"]
            for row in distribution
        }
        self.assertEqual(by_setting[(1, 0.001)], 100.0)
        self.assertEqual(by_setting[(2, 0.005)], 100.0)

    def test_seed_summary_has_explicit_seed_scope(self):
        results = [self.result(42, 0.5, {"0.001": 2, "0.005": 2})]
        selections = [{
            "selection_mode": "count", "selection_value": 1,
            "selection_count": 1, "selection_fraction": 1 / 64,
        }]
        averages, accuracy, distribution = build_summary(
            results, ["RN50"], ["imagenet-a"], [1], selections,
            [0.001, 0.005],
        )
        with tempfile.TemporaryDirectory() as directory:
            summary_root = Path(directory)
            write_summary(
                summary_root, results, averages, accuracy, distribution, seed=42
            )
            self.assertTrue((summary_root / "summary_results.json").is_file())
            self.assertFalse((summary_root / "average_results.json").exists())
            summary = (summary_root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Data-wise LR TPT Seed 42 Summary", summary)
            self.assertIn("Scope: **Seed 42**", summary)

    def test_overall_summary_combines_seed_rows_and_average(self):
        results = [
            self.result(11, 0.5, {"0.001": 1, "0.005": 3}),
            self.result(42, 0.7, {"0.001": 3, "0.005": 1}),
        ]
        selections = [{
            "selection_mode": "count", "selection_value": 1,
            "selection_count": 1, "selection_fraction": 1 / 64,
        }]
        averages, accuracy, distribution = build_summary(
            results, ["RN50"], ["imagenet-a"], [1], selections,
            [0.001, 0.005],
        )
        with tempfile.TemporaryDirectory() as directory:
            summary_root = Path(directory)
            write_summary(
                summary_root, results, averages, accuracy, distribution
            )
            summary = (summary_root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("All seeds and cross-seed mean", summary)
            self.assertIn("| RN50 | imagenet-a | count=1 | 11 |", summary)
            self.assertIn("| RN50 | imagenet-a | count=1 | 42 |", summary)
            self.assertIn("60.0000%", summary)
            self.assertTrue((summary_root / "average_results.json").is_file())


if __name__ == "__main__":
    unittest.main()
