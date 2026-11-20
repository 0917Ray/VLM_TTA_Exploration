#!/usr/bin/env python3
"""Rebuild experiment 8 summaries and figures from saved per-sample metrics only."""

from __future__ import annotations

import argparse
import importlib
import os
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
_evaluation = importlib.import_module("8_evaluate_tpt_gradient_alignment")
_plotting = importlib.import_module("8_plot_tpt_gradient_alignment")


def rebuild(run_dir: Path, plot_config: Path) -> None:
    run_dir = run_dir.resolve()
    plot_config = plot_config.resolve()
    task_paths = sorted(run_dir.glob("seed_*/*/task_result.json"))
    if not task_paths:
        raise ValueError(f"No saved task results under {run_dir}")
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.is_file():
        config = _evaluation.load_yaml(resolved_path)
        expected = {
            run_dir / f"seed_{seed}" / _evaluation.task_slug(model, dataset, rate) / "task_result.json"
            for model, dataset, rate, seed in product(config["models"], config["datasets"], config["learning_rates"], config["seeds"])
        }
        if set(task_paths) != expected:
            missing = sorted(str(path.relative_to(run_dir)) for path in expected - set(task_paths))
            raise ValueError(f"Task set does not match resolved config; missing={missing}. Use --task-dir for an individual completed task.")
    elif not (run_dir / "COMPLETED.json").is_file():
        raise ValueError("Need resolved_config.yaml to verify all expected tasks, or a legacy COMPLETED.json")
    by_seed: dict[int, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    # Validate all inputs before rewriting any summaries or completion markers.
    for task_path in task_paths:
        result, records = _evaluation.load_task_data(task_path.parent)
        expected_root = run_dir / f"seed_{result['seed']}" / _evaluation.task_slug(result["model"], result["dataset"], result["learning_rate"])
        if task_path.parent != expected_root:
            raise ValueError(f"Task identity does not match its directory: {task_path}")
        by_seed[int(result["seed"])].append((result, records))
    (run_dir / "COMPLETED.json").unlink(missing_ok=True)
    for task_path in task_paths:
        result = _evaluation.finalize_task(task_path.parent, plot_config)
        _evaluation.atomic_write_json(task_path, result)
        print(f"[SUCCESS] rebuilt task outputs: {task_path.parent}", flush=True)
    all_results: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    for seed, tasks in sorted(by_seed.items()):
        results = [result for result, _ in tasks]
        records = [record for _, task_records in tasks for record in task_records]
        root = run_dir / f"seed_{seed}" / "summary"
        _evaluation.write_summary_files(root, results, records, f"Seed {seed}")
        paths = _plotting.generate_plots(plot_config, root / "distribution_data.csv", root / "plots")
        print(f"[SUCCESS] seed {seed}: {len(records)} samples, {len(paths)} figure files", flush=True)
        all_results.extend(results)
        all_records.extend(records)
    root = run_dir / "summary"
    _evaluation.write_summary_files(root, all_results, all_records, "All tasks")
    paths = _plotting.generate_plots(plot_config, root / "distribution_data.csv", root / "plots")
    _evaluation.atomic_write_json(run_dir / "COMPLETED.json", {
        "status": "complete", "completed_at": datetime.now().astimezone().isoformat(),
        "total_tasks": len(all_results), "expected_seeds": sorted(by_seed),
        "completed_seeds": sorted(by_seed), "summary": str(root), "rebuilt": True,
    })
    (run_dir / "FAILED.json").unlink(missing_ok=True)
    print(f"[SUCCESS] pooled: {len(all_records)} samples, {len(paths)} figure files at {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--task-dir", type=Path, help="Rebuild one completed task, even while other tasks are running")
    parser.add_argument("--output-dir", type=Path, help="Separate destination for --task-dir; source data is left unchanged")
    parser.add_argument("--plot-config", type=Path, default=PROJECT_ROOT / "configs/visualization/8_tpt_gradient_alignment.yaml")
    args = parser.parse_args()
    if args.output_dir is not None and args.task_dir is None:
        parser.error("--output-dir requires --task-dir")
    if args.task_dir is not None:
        destination = args.output_dir.resolve() if args.output_dir else args.task_dir.resolve()
        _evaluation.finalize_task(args.task_dir.resolve(), args.plot_config.resolve(), destination)
        print(f"[SUCCESS] task summary and figures: {destination}", flush=True)
    else:
        rebuild(args.run_dir, args.plot_config)


if __name__ == "__main__":
    main()
