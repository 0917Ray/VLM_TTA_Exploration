#!/usr/bin/env python3
"""Plot per-sample LR proportions and accuracy for data-wise LR TPT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tpt-matplotlib-cache")
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "data": {
        "evaluation_config": "../evaluation/datawise_lr_tpt.yaml",
        "accuracy": None,
        "distribution": None,
        "source": None,
    },
    "figure": {
        "figsize": [9.5, 5.2],
        "dpi": 300,
        "transparent": True,
        "title_pad": 30,
    },
    "bars": {
        "group_width": 0.76,
        "face_alpha": 0.5,
        "edge_alpha": 0.95,
        "edge_width": 1.8,
    },
    "errorbar": {
        "enabled": True,
        "color": "#3A3D42",
        "alpha": 0.82,
        "linewidth": 1.05,
        "capsize": 3.0,
    },
    "value_labels": {
        "accuracy": True,
        "distribution": True,
        "format": "{:.1f}",
        "fontsize": 8.0,
    },
    "axis": {
        "accuracy_ylim": [0, 110],
        "distribution_ylim": [0, 110],
        "grid": True,
        "x_tick_rotation": 0,
    },
    "spines": {"linewidth": 2.2, "color": "#3A3D42"},
    "legend": {"location": "upper center", "anchor": [0.5, 1.02], "columns": 3},
    "colors": ["#4F7C65", "#A75B73", "#4C78A8", "#8F6AAE", "#D08C3F"],
    "output": {"pdf": True, "directory": None},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        override = json.load(stream)
    if not isinstance(override, dict):
        raise ValueError("Plot config must contain a JSON object")
    unknown = set(override) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown plot config keys: {', '.join(sorted(unknown))}")
    config = deep_merge(DEFAULT_CONFIG, override)
    if not isinstance(config["enabled"], bool):
        raise ValueError("enabled must be true or false")
    if (not isinstance(config["colors"], list) or not config["colors"] or
            any(not isinstance(color, str) for color in config["colors"])):
        raise ValueError("colors must be a non-empty list")
    for color in config["colors"]:
        mcolors.to_rgba(color)
    for field in ("accuracy_ylim", "distribution_ylim"):
        limits = config["axis"][field]
        if (not isinstance(limits, list) or len(limits) != 2 or
                not all(isinstance(value, (int, float)) for value in limits) or
                limits[0] >= limits[1]):
            raise ValueError(f"axis.{field} must contain increasing numeric limits")
    return config


def resolve_relative(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def infer_latest_run(config: dict[str, Any], config_path: Path) -> Path:
    evaluation_value = config["data"]["evaluation_config"]
    if evaluation_value is None:
        raise ValueError("data.evaluation_config is required when data paths are omitted")
    evaluation_path = resolve_relative(evaluation_value, config_path.parent)
    with evaluation_path.open(encoding="utf-8") as stream:
        evaluation = yaml.safe_load(stream) or {}
    output_root = Path(evaluation["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    idea = evaluation.get("idea")
    if idea:
        output_root = output_root / f"idea_{'_'.join(str(idea).split())}"
    completed = sorted(
        path for path in output_root.glob("run_*")
        if (path / "COMPLETED.json").is_file()
    )
    if not completed:
        raise FileNotFoundError(f"No completed run found under {output_root}")
    return completed[-1]


def infer_paths(
    config: dict[str, Any], config_path: Path
) -> tuple[Path, Path, Path]:
    data = config["data"]
    accuracy = data["accuracy"]
    distribution = data["distribution"]
    if (accuracy is None) != (distribution is None):
        raise ValueError("data.accuracy and data.distribution must be set together")
    if accuracy is not None:
        accuracy_path = resolve_relative(accuracy, config_path.parent)
        distribution_path = resolve_relative(distribution, config_path.parent)
        output = config["output"]["directory"]
        output_dir = (
            resolve_relative(output, config_path.parent)
            if output is not None else config_path.parent
        )
        return accuracy_path, distribution_path, output_dir
    run_root = infer_latest_run(config, config_path)
    summary = run_root / "summary"
    return (
        summary / "accuracy_plot_data.csv",
        summary / "lr_distribution_plot_data.csv",
        summary / "plots",
    )


def read_csv(path: Path, numeric: set[str], integer: set[str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = (numeric | integer | {"model", "dataset", "selection_mode"}) - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    for row_number, row in enumerate(rows, start=2):
        for key in numeric:
            value = row[key]
            row[key] = None if value in {"", "None", "null"} else float(value)
            if row[key] is not None and not math.isfinite(row[key]):
                raise ValueError(f"Non-finite {key} at {path}:{row_number}")
        for key in integer:
            row[key] = int(row[key])
    return rows


def configure_style(config: dict[str, Any]) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": config["spines"]["linewidth"],
    })


def style_axis(axis, config: dict[str, Any], ylabel: str, ylim: list[float]) -> None:
    axis.set_ylabel(ylabel)
    axis.set_ylim(*ylim)
    if config["axis"]["grid"]:
        axis.grid(axis="y", color="#DADDE2", linestyle="--", linewidth=0.75, zorder=0)
    for spine in axis.spines.values():
        spine.set_linewidth(config["spines"]["linewidth"])
        spine.set_color(config["spines"]["color"])


def add_legend(axis, config: dict[str, Any], columns: int) -> None:
    axis.legend(
        loc=config["legend"]["location"],
        bbox_to_anchor=config["legend"]["anchor"],
        ncol=columns,
        frameon=False,
    )


def error_values(values: list[float | None]) -> np.ndarray | None:
    if all(value is None for value in values):
        return None
    return np.asarray([0.0 if value is None else value for value in values], dtype=float)


def add_value_labels(axis, bars, config: dict[str, Any]) -> None:
    label_config = config["value_labels"]
    for bar in bars:
        value = bar.get_height()
        axis.annotate(
            label_config["format"].format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=label_config["fontsize"],
        )


def selection_label(row: dict[str, Any]) -> str:
    if row["selection_mode"] == "count":
        return f"count={row['selection_count']}"
    return f"fraction={row['selection_fraction']:g} (count={row['selection_count']})"


def selection_directory(row: dict[str, Any]) -> str:
    value = f"{row['selection_value']:g}".replace(".", "p")
    return f"{row['selection_mode']}_{value}"


def save_figure(figure, base: Path, config: dict[str, Any]) -> list[Path]:
    figure.tight_layout()
    paths = [base.with_suffix(".png")]
    figure.savefig(
        paths[0], dpi=config["figure"]["dpi"],
        transparent=config["figure"]["transparent"], bbox_inches="tight",
    )
    if config["output"]["pdf"]:
        paths.append(base.with_suffix(".pdf"))
        figure.savefig(paths[-1], transparent=config["figure"]["transparent"], bbox_inches="tight")
    plt.close(figure)
    return paths


def scope_title(row: dict[str, Any]) -> str:
    scope = str(row.get("summary_scope", "")).strip()
    return f" / {scope}" if scope else ""


def plot_distribution(
    rows: list[dict[str, Any]], output_dir: Path, config: dict[str, Any]
) -> list[Path]:
    rates = sorted({row["candidate_lr"] for row in rows})
    steps = sorted({row["tta_steps"] for row in rows})
    keys = [(row["candidate_lr"], row["tta_steps"]) for row in rows]
    if len(keys) != len(set(keys)) or len(rows) != len(rates) * len(steps):
        raise ValueError("LR distribution must contain one row per LR/step pair")
    positions = np.arange(len(rates), dtype=float)
    width = config["bars"]["group_width"] / len(steps)
    figure, axis = plt.subplots(figsize=config["figure"]["figsize"])
    for index, step_count in enumerate(steps):
        step_rows = [
            next(row for row in rows if row["candidate_lr"] == rate and row["tta_steps"] == step_count)
            for rate in rates
        ]
        values = [row["selection_percent"] for row in step_rows]
        errors = error_values([row["selection_percent_std"] for row in step_rows])
        color = config["colors"][index % len(config["colors"])]
        bars = axis.bar(
            positions + (index - (len(steps) - 1) / 2) * width,
            values,
            width=width,
            label=f"{step_count} step" if step_count == 1 else f"{step_count} steps",
            color=mcolors.to_rgba(color, config["bars"]["face_alpha"]),
            edgecolor=mcolors.to_rgba(color, config["bars"]["edge_alpha"]),
            linewidth=config["bars"]["edge_width"],
            yerr=errors if config["errorbar"]["enabled"] else None,
            capsize=config["errorbar"]["capsize"],
            error_kw={
                "ecolor": config["errorbar"]["color"],
                "alpha": config["errorbar"]["alpha"],
                "elinewidth": config["errorbar"]["linewidth"],
            },
            zorder=3,
        )
        if config["value_labels"]["distribution"]:
            add_value_labels(axis, bars, config)
    axis.set_xticks(positions, [f"{rate:g}" for rate in rates])
    axis.tick_params(axis="x", rotation=config["axis"]["x_tick_rotation"])
    axis.set_xlabel("Candidate learning rate")
    axis.set_title(
        f"Per-sample LR Selection: {rows[0]['model']} / {rows[0]['dataset']} / "
        f"{selection_label(rows[0])}{scope_title(rows[0])}",
        pad=config["figure"]["title_pad"],
    )
    style_axis(axis, config, "Selected samples (%)", config["axis"]["distribution_ylim"])
    add_legend(
        axis, config, min(config["legend"]["columns"], len(steps))
    )
    return save_figure(figure, output_dir / "lr_selection_distribution", config)


def plot_accuracy(
    rows: list[dict[str, Any]], output_dir: Path, config: dict[str, Any]
) -> list[Path]:
    rows = sorted(rows, key=lambda row: row["tta_steps"])
    if len({row["tta_steps"] for row in rows}) != len(rows):
        raise ValueError("Accuracy data contains duplicate step rows")
    labels = ["CLIP\nzero-shot"] + [
        f"Data-wise LR\n{row['tta_steps']} step" + ("" if row["tta_steps"] == 1 else "s")
        for row in rows
    ]
    series = [
        ("Top-1", "top1_accuracy_percent", "top1_std_percent", "zero_shot_top1_accuracy_percent", "zero_shot_top1_std_percent"),
        ("Top-5", "top5_accuracy_percent", "top5_std_percent", "zero_shot_top5_accuracy_percent", "zero_shot_top5_std_percent"),
    ]
    positions = np.arange(len(labels), dtype=float)
    width = config["bars"]["group_width"] / len(series)
    figure, axis = plt.subplots(figsize=config["figure"]["figsize"])
    line_values = []
    for index, (name, value_key, std_key, zero_key, zero_std_key) in enumerate(series):
        values = [rows[0][zero_key]] + [row[value_key] for row in rows]
        errors = error_values([rows[0][zero_std_key]] + [row[std_key] for row in rows])
        color = config["colors"][index % len(config["colors"])]
        bars = axis.bar(
            positions + (index - 0.5) * width,
            values,
            width=width,
            label=name,
            color=mcolors.to_rgba(color, config["bars"]["face_alpha"]),
            edgecolor=mcolors.to_rgba(color, config["bars"]["edge_alpha"]),
            linewidth=config["bars"]["edge_width"],
            yerr=errors if config["errorbar"]["enabled"] else None,
            capsize=config["errorbar"]["capsize"],
            error_kw={"ecolor": config["errorbar"]["color"], "elinewidth": config["errorbar"]["linewidth"]},
            zorder=3,
        )
        if config["value_labels"]["accuracy"]:
            add_value_labels(axis, bars, config)
        line_values.append((name, values, errors, color))
    axis.set_xticks(positions, labels)
    axis.set_title(
        f"Data-wise LR TPT Accuracy: {rows[0]['model']} / {rows[0]['dataset']} / "
        f"{selection_label(rows[0])}{scope_title(rows[0])}",
        pad=config["figure"]["title_pad"],
    )
    style_axis(axis, config, "Accuracy (%)", config["axis"]["accuracy_ylim"])
    add_legend(axis, config, 2)
    paths = save_figure(figure, output_dir / "accuracy_bar", config)

    figure, axis = plt.subplots(figsize=config["figure"]["figsize"])
    markers = ["o", "s"]
    for index, (name, values, errors, color) in enumerate(line_values):
        axis.errorbar(
            positions, values, yerr=errors if config["errorbar"]["enabled"] else None,
            label=name, color=color, marker=markers[index], linewidth=2.0,
            markersize=6, capsize=config["errorbar"]["capsize"], zorder=3,
        )
        if config["value_labels"]["accuracy"]:
            for x, value in zip(positions, values):
                axis.annotate(
                    config["value_labels"]["format"].format(value), (x, value),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=config["value_labels"]["fontsize"],
                )
    axis.set_xticks(positions, labels)
    axis.set_title(
        f"Data-wise LR TPT Accuracy: {rows[0]['model']} / {rows[0]['dataset']} / "
        f"{selection_label(rows[0])}{scope_title(rows[0])}",
        pad=config["figure"]["title_pad"],
    )
    style_axis(axis, config, "Accuracy (%)", config["axis"]["accuracy_ylim"])
    add_legend(axis, config, 2)
    paths.extend(save_figure(figure, output_dir / "accuracy_line", config))
    return paths


def write_effective_bundle(
    output_dir: Path,
    config: dict[str, Any],
    accuracy_path: Path,
    distribution_path: Path,
) -> tuple[Path, Path, Path]:
    bundled_accuracy = output_dir / "accuracy_plot_data.csv"
    bundled_distribution = output_dir / "lr_distribution_plot_data.csv"
    if accuracy_path.resolve() != bundled_accuracy.resolve():
        shutil.copyfile(accuracy_path, bundled_accuracy)
    if distribution_path.resolve() != bundled_distribution.resolve():
        shutil.copyfile(distribution_path, bundled_distribution)
    bundled_script = output_dir / Path(__file__).name
    if Path(__file__).resolve() != bundled_script.resolve():
        shutil.copyfile(__file__, bundled_script)
    effective = deep_merge(config, {
        "data": {
            "accuracy": bundled_accuracy.name,
            "distribution": bundled_distribution.name,
            "source": config["data"].get("source") or str(accuracy_path.parent),
        },
        "output": {"directory": "."},
    })
    effective_path = output_dir / "datawise_lr_tpt_plots_config.json"
    effective_path.write_text(
        json.dumps(effective, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bundled_accuracy, bundled_distribution, effective_path


def generate_plots(
    config_path: Path,
    accuracy_data: Path | None = None,
    distribution_data: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if not config["enabled"]:
        return []
    if accuracy_data is not None and distribution_data is not None and output_dir is not None:
        accuracy_path = Path(accuracy_data).resolve()
        distribution_path = Path(distribution_data).resolve()
        output_dir = Path(output_dir).resolve()
    else:
        inferred_accuracy, inferred_distribution, inferred_output = infer_paths(
            config, config_path
        )
        accuracy_path = (
            Path(accuracy_data).resolve() if accuracy_data is not None else inferred_accuracy
        )
        distribution_path = (
            Path(distribution_data).resolve()
            if distribution_data is not None else inferred_distribution
        )
        output_dir = Path(output_dir).resolve() if output_dir is not None else inferred_output
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style(config)

    accuracy_rows = read_csv(
        accuracy_path,
        numeric={
            "top1_accuracy_percent", "top1_std_percent", "top5_accuracy_percent",
            "top5_std_percent", "zero_shot_top1_accuracy_percent",
            "zero_shot_top1_std_percent", "zero_shot_top5_accuracy_percent",
            "zero_shot_top5_std_percent", "selection_value", "selection_fraction",
        },
        integer={"tta_steps", "num_seeds", "selection_count"},
    )
    distribution_rows = read_csv(
        distribution_path,
        numeric={
            "candidate_lr", "selection_percent", "selection_percent_std",
            "selection_value", "selection_fraction",
        },
        integer={
            "tta_steps", "num_seeds", "total_samples", "total_selected",
            "selection_count",
        },
    )
    combinations = sorted({
        (
            row["model"], row["dataset"], row["selection_mode"],
            row["selection_value"], row["selection_count"], row["selection_fraction"],
        )
        for row in accuracy_rows
    })
    distribution_combinations = sorted({
        (
            row["model"], row["dataset"], row["selection_mode"],
            row["selection_value"], row["selection_count"], row["selection_fraction"],
        )
        for row in distribution_rows
    })
    if combinations != distribution_combinations:
        raise ValueError("Accuracy and LR-distribution data cover different model/dataset groups")

    write_effective_bundle(output_dir, config, accuracy_path, distribution_path)
    paths: list[Path] = []
    for (
        model_name, dataset_name, selection_mode, selection_value,
        selection_count, selection_fraction,
    ) in combinations:
        selection_key = {
            "selection_mode": selection_mode,
            "selection_value": selection_value,
            "selection_count": selection_count,
            "selection_fraction": selection_fraction,
        }
        group_dir = (
            output_dir / f"{model_name.replace('/', '-')}_{dataset_name}"
            / selection_directory(selection_key)
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        accuracy_group = [
            row for row in accuracy_rows
            if row["model"] == model_name and row["dataset"] == dataset_name and
            row["selection_mode"] == selection_mode and
            row["selection_value"] == selection_value
        ]
        distribution_group = [
            row for row in distribution_rows
            if row["model"] == model_name and row["dataset"] == dataset_name and
            row["selection_mode"] == selection_mode and
            row["selection_value"] == selection_value
        ]
        paths.extend(plot_accuracy(accuracy_group, group_dir, config))
        paths.extend(plot_distribution(distribution_group, group_dir, config))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--accuracy-data", type=Path)
    parser.add_argument("--distribution-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_plots(
        args.config, args.accuracy_data, args.distribution_data, args.output_dir
    )
    print(f"[SUCCESS] generated {len(paths)} data-wise LR TPT figures")


if __name__ == "__main__":
    main()
