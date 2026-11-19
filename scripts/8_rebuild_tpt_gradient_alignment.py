#!/usr/bin/env python3
"""Rebuild experiment 8 summaries and figures from saved per-sample metrics only."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_evaluation = importlib.import_module("8_evaluate_tpt_gradient_alignment")
_plotting = importlib.import_module("8_plot_tpt_gradient_alignment")


def rebuild(run_dir: Path, plot_config: Path) -> None:
    run_dir = run_dir.resolve()
    plot_config = plot_config.resolve()
    if not (run_dir / "COMPLETED.json").is_file():
        raise ValueError(f"Completed run not found: {run_dir}")
    task_paths = sorted(run_dir.glob("seed_*/*/task_result.json"))
    if not task_paths:
        raise ValueError(f"No saved task results under {run_dir}")
    by_seed: dict[int, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    for task_path in task_paths:
        result = json.loads(task_path.read_text(encoding="utf-8"))
        sample_path = task_path.parent / "sample_gradient_metrics.json"
        records = json.loads(sample_path.read_text(encoding="utf-8"))
        if len(records) != int(result["samples"]):
            raise ValueError(f"Sample count mismatch in {task_path}")
        if any(int(row["seed"]) != int(result["seed"]) for row in records):
            raise ValueError(f"Seed mismatch in {sample_path}")
        result["summary_rows"] = _evaluation.summarize_records(records, "per_task")
        _evaluation.atomic_write_json(task_path, result)
        by_seed[int(result["seed"])].append((result, records))
        print(f"[INFO] loaded {len(records)} saved samples from {sample_path}", flush=True)
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
    print(f"[SUCCESS] pooled: {len(all_records)} samples, {len(paths)} figure files at {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--plot-config", type=Path, default=PROJECT_ROOT / "configs/visualization/8_tpt_gradient_alignment.yaml")
    args = parser.parse_args()
    rebuild(args.run_dir, args.plot_config)


if __name__ == "__main__":
    main()
