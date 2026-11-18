from __future__ import annotations

import csv
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_tpt import parse_args  # noqa: E402

generate_plots = importlib.import_module("5_plot_tpt_lr_steps_ablation").generate_plots
generate_lr_plots = importlib.import_module("4_plot_tpt_lr_ablation").generate_plots


class LearningRatePlotsTest(unittest.TestCase):
    def test_evaluation_configs_have_separate_protocols_and_outputs(self):
        cases = (
            ("4_tpt_lr_ablation.yaml", 1, 10, 0.01, "ablation", "idea_one_step_lr"),
            ("5_tpt_lr_steps_ablation.yaml", 10, 7, 0.05, "lr_steps", "idea_lr_steps"),
        )
        for filename, step_count, rate_count, weight_decay, plot_kind, idea in cases:
            config = PROJECT_ROOT / "configs/evaluation" / filename
            with self.subTest(filename=filename), patch.object(
                sys, "argv", ["evaluate_tpt.py", "--config", str(config)]
            ):
                args, _ = parse_args()
                self.assertEqual(len(args.tta_step_values), step_count)
                self.assertEqual(len(args.learning_rate_values), rate_count)
                self.assertEqual(args.weight_decay, weight_decay)
                self.assertEqual(args.plot["kind"], plot_kind)
                self.assertEqual(args.output_root.name, idea)

    def test_separate_one_step_and_joint_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for legacy, steps in ((True, (1,)), (False, (1, 2))):
                case = root / ("legacy" if legacy else "joint")
                case.mkdir()
                source_config = PROJECT_ROOT / ("configs/visualization/4_tpt_learning_rate_plots.json"
                                                if legacy else
                                                "configs/visualization/5_tpt_lr_steps_ablation_plots.json")
                config = json.loads(source_config.read_text(encoding="utf-8"))
                config["data"]["evaluation_config"] = str(
                    PROJECT_ROOT / ("configs/evaluation/4_tpt_lr_ablation.yaml" if legacy else
                                    "configs/evaluation/5_tpt_lr_steps_ablation.yaml")
                )
                config["figure"]["dpi"] = 50
                config["output"]["pdf"] = False
                if not legacy:
                    config["plots"].update({
                        "per_learning_rate": False,
                        "best_steps": False,
                        "best_accuracy": False,
                    })
                config_path = case / "config.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                fields = [
                    "model", "dataset", "selection_count", "learning_rate", "tta_steps",
                    "average_elapsed_minutes", "top1_accuracy_percent", "top1_std_percent",
                    "top5_accuracy_percent", "top5_std_percent",
                    "zero_shot_top1_accuracy_percent", "zero_shot_top1_std_percent",
                    "zero_shot_top5_accuracy_percent", "zero_shot_top5_std_percent",
                ]
                if not legacy:
                    fields.append("weight_decay")
                data_path = case / "results.csv"
                with data_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for rate in (0.001, 0.003):
                        for step in steps:
                            row = {
                                "model": "ViT-B/16", "dataset": "imagenet-a",
                                "selection_count": 1, "learning_rate": rate,
                                "tta_steps": step, "average_elapsed_minutes": 1,
                                "top1_accuracy_percent": 50 + 1000 * rate + step,
                                "top1_std_percent": 0.5,
                                "top5_accuracy_percent": 80 + 100 * rate + step,
                                "top5_std_percent": 0.3,
                                "zero_shot_top1_accuracy_percent": 48,
                                "zero_shot_top1_std_percent": 0,
                                "zero_shot_top5_accuracy_percent": 78,
                                "zero_shot_top5_std_percent": 0,
                            }
                            if not legacy:
                                row["weight_decay"] = 0.05
                            writer.writerow(row)
                output = case / "plots"
                (generate_lr_plots if legacy else generate_plots)(config_path, data_path, output)
                if legacy:
                    self.assertTrue((output / "learning_rate_line.png").is_file())
                    self.assertTrue((output / "learning_rate_bar.png").is_file())
                else:
                    self.assertTrue((output / "top1_lr_steps_heatmap.png").is_file())


if __name__ == "__main__":
    unittest.main()
