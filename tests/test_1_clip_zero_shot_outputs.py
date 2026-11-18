from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_clip_zeroshot import (  # noqa: E402
    average_results,
    clear_previous_outputs,
    parse_args,
    Result,
    write_average_results,
)


class ZeroShotOutputTest(unittest.TestCase):
    def test_idea_routes_output_and_rejects_path_components(self):
        config = PROJECT_ROOT / "configs/evaluation/1_clip_zero_shot.yaml"
        with patch.object(sys, "argv", ["evaluate_clip_zeroshot.py", "--config", str(config),
                                        "--idea", "baseline repeat", "--device", "cpu"]):
            args, _ = parse_args()
        self.assertEqual(args.idea, "baseline_repeat")
        self.assertEqual(args.output_root.name, "idea_baseline_repeat")

        with patch.object(sys, "argv", ["evaluate_clip_zeroshot.py", "--config", str(config),
                                        "--idea", "../other", "--device", "cpu"]):
            with self.assertRaisesRegex(SystemExit, "directory-safe"):
                parse_args()

    def test_same_idea_replaces_all_summary_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "idea_baseline"
            for accuracy in (0.1, 0.2):
                clear_previous_outputs(root)
                self.assertFalse((root / "average_results.md").exists())
                results = [Result(
                    model="RN50", dataset="imagenet-a", samples=10,
                    correct=int(accuracy * 10), top1_accuracy=accuracy,
                    candidate_classes=200, class_names_source="classes.json",
                    prompt_template="a photo of a {}.", seed=44,
                    device="cuda:0", idea="baseline",
                )]
                averages = average_results(results, ["RN50"], ["imagenet-a"])
                write_average_results(averages, root, ["RN50"], ["imagenet-a"])

            markdown = (root / "average_results.md").read_text(encoding="utf-8")
            self.assertIn("Idea: `baseline`", markdown)
            self.assertIn("Seeds: 44", markdown)
            self.assertIn("20.0000%", markdown)
            self.assertNotIn("10.0000%", markdown)
            with (root / "average_results.csv").open(encoding="utf-8", newline="") as stream:
                csv_row = next(csv.DictReader(stream))
            json_row = json.loads((root / "average_results.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(float(csv_row["average_top1_accuracy"]), 0.2)
            self.assertEqual(json_row["average_top1_accuracy"], 0.2)


if __name__ == "__main__":
    unittest.main()
