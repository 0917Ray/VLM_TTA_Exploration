from __future__ import annotations

import importlib
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

_alignment = importlib.import_module("8_evaluate_tpt_gradient_alignment")
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

class TaskLifecycleTest(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "smoke_test" / "outputs" / "test_8_lifecycle"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, datasets, seeds=(123, 456)):
        return argparse.Namespace(models=["m"], datasets=datasets, learning_rates=[0.003],
                                  seeds=list(seeds), output_root=self.root,
                                  plot={"enabled": False, "config": None})

    @staticmethod
    def launch(config_path, device_name, postprocess=False):
        stage = "postprocess" if postprocess else "compute"
        with (config_path.parent / f"{stage}.log").open("w") as stream:
            subprocess.run([sys.executable, str(ROOT / "smoke_test" / "8_task_worker_fixture.py"),
                            str(config_path), stage], check=True, stdout=stream, stderr=stream,
                           env=_alignment.task_environment(device_name))

    def run_fixture(self, args, devices=("cpu:0", "cpu:1")):
        with patch.object(_alignment, "run_task_process", side_effect=self.launch):
            return _alignment.run_jobs(args, list(devices))

    def test_process_exit_incremental_outputs_dynamic_queue_and_seed_order(self):
        results = self.run_fixture(self.args(["slow", "fast", "queued"]))
        self.assertEqual(len(results), 6)
        compute = [json.loads(p.read_text()) for p in self.root.glob("seed_*/*/compute_event.json")]
        posts = sorted((json.loads(p.read_text()) for p in self.root.glob("seed_*/*/postprocess_event.json")), key=lambda e: e["start"])
        self.assertEqual(len({e["pid"] for e in compute}), 6)
        for device in ("cpu:0", "cpu:1"):
            events = sorted((e for e in compute if e["device"] == device), key=lambda e: e["start"])
            self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(events, events[1:])))
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(posts, posts[1:])))
        for dataset in ("slow", "fast", "queued"):
            events = sorted((e for e in compute if e["dataset"] == dataset), key=lambda e: e["start"])
            self.assertEqual([e["seed"] for e in events], [123, 456])
            self.assertEqual(len({e["device"] for e in events}), 1)
        slow = next(e for e in compute if e["dataset"] == "slow" and e["seed"] == 123)
        queued = next(e for e in posts if e["dataset"] == "queued" and e["seed"] == 456)
        self.assertLess(queued["end"], slow["end"])

    def test_failures_keep_successful_tasks_and_do_not_mark_run_complete(self):
        with self.assertRaisesRegex(RuntimeError, "2 task stage"):
            self.run_fixture(self.args(["compute-fail", "plot-fail", "ok"], seeds=(123,)))
        self.assertFalse((self.root / "COMPLETED.json").exists())
        failure = json.loads((self.root / "FAILED.json").read_text())
        self.assertEqual(failure["computed_tasks"], 2)
        self.assertTrue((self.root / "seed_123/m__ok__lr_0p003/COMPLETED.json").is_file())
        self.assertTrue((self.root / "seed_123/m__plot-fail__lr_0p003/task_result.json").is_file())

    def test_device_mapping_respects_external_allocation_and_cpu_is_hidden(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}):
            self.assertEqual(_alignment.task_environment("cuda:1")["CUDA_VISIBLE_DEVICES"], "GPU-b")
            self.assertEqual(_alignment.task_environment(None)["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")

    def test_legacy_task_can_finalize_without_run_marker_and_retry_plot_failure(self):
        self.run_fixture(self.args(["ok"], seeds=(123,)))
        task = self.root / "seed_123/m__ok__lr_0p003"
        original = (task / "task_result.json").read_bytes()
        destination = self.root / "independent-output"
        result = _alignment.finalize_task(task, None, destination)
        self.assertEqual(result["sample_results"], str(task / "sample_gradient_metrics.json"))
        rows = json.loads((destination / "gradient_summary.json").read_text())
        self.assertEqual({row["scope"] for row in rows}, {"per_task"})
        plot_config = self.root / "plot.yaml"
        plot_config.write_text("enabled: true\n")
        plotting = importlib.import_module("8_plot_tpt_gradient_alignment")
        with patch.object(plotting, "generate_plots", side_effect=RuntimeError("plot failed")):
            with self.assertRaisesRegex(RuntimeError, "plot failed"):
                _alignment.finalize_task(task, plot_config, destination)
        self.assertFalse((destination / "COMPLETED.json").exists())
        self.assertTrue((destination / "POSTPROCESS_FAILED.json").exists())
        _alignment.finalize_task(task, None, destination)
        self.assertTrue((destination / "COMPLETED.json").exists())
        self.assertFalse((destination / "POSTPROCESS_FAILED.json").exists())
        self.assertEqual((task / "task_result.json").read_bytes(), original)

    def test_rebuild_rejects_partial_run_before_writing_final_summary(self):
        self.run_fixture(self.args(["ok"], seeds=(123,)))
        config = {"models": ["m"], "datasets": ["ok", "missing"], "learning_rates": [0.003], "seeds": [123]}
        (self.root / "resolved_config.yaml").write_text(json.dumps(config))
        rebuild = importlib.import_module("8_rebuild_tpt_gradient_alignment")
        with self.assertRaisesRegex(ValueError, "Task set does not match"):
            rebuild.rebuild(self.root, ROOT / "configs/visualization/8_tpt_gradient_alignment.yaml")
        self.assertFalse((self.root / "COMPLETED.json").exists())
        self.assertFalse((self.root / "summary").exists())

    def test_main_aggregates_only_after_all_task_outputs_and_keeps_groups(self):
        with patch.object(_alignment, "run_task_process", side_effect=self.launch):
            _alignment.main([
                "--devices", "cpu", "--datasets", "imagenet-a", "imagenet-r",
                "--seeds", "11", "42", "--no-plot", "--run-name", "fixture",
                "--output-root", str(self.root),
            ])
        run = self.root / "idea_Four_Datasets/run_fixture"
        completed = json.loads((run / "COMPLETED.json").read_text())
        self.assertEqual(completed["total_tasks"], 4)
        rows = json.loads((run / "summary/gradient_summary.json").read_text())
        all_rows = [row for row in rows if row["group"] == "all"]
        self.assertEqual({row["dataset"] for row in all_rows}, {"imagenet-a", "imagenet-r"})
        self.assertEqual({row["valid_count"] for row in all_rows}, {2})
        self.assertEqual({row["scope"] for row in all_rows}, {"pooled"})

    def test_rebuild_recovers_all_computed_tasks_after_plot_failure(self):
        self.run_fixture(self.args(["ok"], seeds=(123,)))
        (self.root / "resolved_config.yaml").write_text(json.dumps({
            "models": ["m"], "datasets": ["ok"], "learning_rates": [0.003], "seeds": [123],
        }))
        (self.root / "FAILED.json").write_text('{"stage": "postprocess"}')
        config = self.root / "plot.yaml"
        config.write_text("enabled: false\n")
        rebuild = importlib.import_module("8_rebuild_tpt_gradient_alignment")
        rebuild.rebuild(self.root, config)
        self.assertTrue((self.root / "COMPLETED.json").exists())
        self.assertFalse((self.root / "FAILED.json").exists())
        self.assertTrue((self.root / "summary/distribution_data.csv").exists())


if __name__ == "__main__":
    unittest.main()
