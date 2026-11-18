from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

_transitions = importlib.import_module("7_evaluate_tpt_transition_stats")
ACCURACY_TRANSITIONS = _transitions.ACCURACY_TRANSITIONS
LOSS_TRANSITIONS = _transitions.LOSS_TRANSITIONS
accuracy_transition = _transitions.accuracy_transition
assign_jobs = _transitions.assign_jobs
build_jobs = _transitions.build_jobs
build_summary = _transitions.build_summary
loss_transition = _transitions.loss_transition
marginal_cross_entropy = _transitions.marginal_cross_entropy
parse_args = _transitions.parse_args
generate_plots = importlib.import_module("7_plot_tpt_transition_stats").generate_plots


class TransitionClassificationTest(unittest.TestCase):
    def test_accuracy_transitions_cover_binary_states(self):
        observed = {
            accuracy_transition(before, after)
            for before in (False, True)
            for after in (False, True)
        }
        self.assertEqual(observed, set(ACCURACY_TRANSITIONS))

    def test_loss_transitions_cover_decrease_and_non_decrease(self):
        cases = {
            loss_transition(1.0, entropy_after, 1.0, ce_after, 1e-8)
            for entropy_after in (0.5, 1.5)
            for ce_after in (0.5, 1.5)
        }
        self.assertEqual(cases, set(LOSS_TRANSITIONS))

    def test_equal_losses_enter_up_or_equal_bucket(self):
        self.assertEqual(
            loss_transition(1.0, 1.0, 1.0, 1.0, 1e-8),
            "entropy_up_or_equal_ce_up_or_equal",
        )

    def test_marginal_cross_entropy_matches_manual_average_probability(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
        expected = -torch.log(logits.softmax(dim=1).mean(dim=0)[1])
        actual = marginal_cross_entropy(logits, 1)
        torch.testing.assert_close(actual, expected)


class ConfigurationAndSchedulingTest(unittest.TestCase):
    def test_requested_grid_becomes_independent_tasks(self):
        args, _ = parse_args([
            "--config", "configs/evaluation/7_tpt_transition_stats.yaml",
            "--devices", "cpu",
            "--datasets", "imagenet-a", "imagenet-r",
            "--learning-rates", "0.001", "0.003",
            "--seeds", "42", "123",
            "--no-plot",
        ])
        jobs = build_jobs(args)
        self.assertEqual(len(jobs), 8)
        assignments = assign_jobs(jobs, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        self.assertTrue(all(len(device_jobs) == 2 for device_jobs in assignments))


class SummaryAndPlotTest(unittest.TestCase):
    @staticmethod
    def result(seed: int, accuracy_shift: float) -> dict:
        accuracy_proportions = {
            "correct_to_correct": 0.4 + accuracy_shift,
            "correct_to_wrong": 0.1,
            "wrong_to_correct": 0.2 - accuracy_shift,
            "wrong_to_wrong": 0.3,
        }
        loss_proportions = {
            "entropy_down_ce_down": 0.5,
            "entropy_down_ce_up_or_equal": 0.2,
            "entropy_up_or_equal_ce_up_or_equal": 0.2,
            "entropy_up_or_equal_ce_down": 0.1,
        }
        return {
            "model": "ViT-B/16",
            "dataset": "imagenet-a",
            "learning_rate": 0.003,
            "seed": seed,
            "samples": 10,
            "zero_shot_top1_accuracy": 0.5,
            "tpt_top1_accuracy": 0.6,
            "mean_entropy_delta": -0.1,
            "mean_cross_entropy_delta": -0.05,
            "accuracy_transition_proportions": accuracy_proportions,
            "accuracy_transition_counts": {
                key: round(value * 10) for key, value in accuracy_proportions.items()
            },
            "loss_transition_proportions": loss_proportions,
            "loss_transition_counts": {
                key: round(value * 10) for key, value in loss_proportions.items()
            },
        }

    def test_summary_is_seed_level_mean_with_sample_std(self):
        averages, accuracy_rows, loss_rows = build_summary([
            self.result(42, 0.0), self.result(123, 0.1),
        ])
        self.assertEqual(len(averages), 1)
        row = next(row for row in accuracy_rows if row["transition"] == "correct_to_correct")
        self.assertAlmostEqual(row["proportion_percent"], 45.0)
        self.assertIsNotNone(row["proportion_percent_std"])
        self.assertEqual(sum(row["total_count"] for row in accuracy_rows), 20)
        self.assertEqual(sum(row["total_count"] for row in loss_rows), 20)

    def test_plot_bundle_can_be_generated_from_saved_csv(self):
        _, accuracy_rows, loss_rows = build_summary([self.result(123, 0.0)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_csv(path: Path, rows: list[dict]) -> None:
                import csv
                import json
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({
                            key: json.dumps(value) if isinstance(value, list) else value
                            for key, value in row.items()
                        })

            accuracy_path = root / "accuracy.csv"
            loss_path = root / "loss.csv"
            write_csv(accuracy_path, accuracy_rows)
            write_csv(loss_path, loss_rows)
            paths = generate_plots(
                PROJECT_ROOT / "configs/visualization/7_tpt_transition_stats.yaml",
                accuracy_data=accuracy_path,
                loss_data=loss_path,
                output_dir=root / "plots",
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue((root / "plots/plot_config.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
