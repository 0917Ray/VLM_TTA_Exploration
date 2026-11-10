#!/usr/bin/env python3
"""Plot a two-dimensional TPT learning-rate and update-step ablation."""

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
        "evaluation_config": None,
        "model": None,
        "dataset": None,
        "selection_count": None,
        "selection_fraction": None,
        "weight_decay": None,
    },
    "series": [
        {
            "name": "top1",
            "column": "top1_accuracy_percent",
            "std_column": "top1_std_percent",
            "zero_shot_column": "zero_shot_top1_accuracy_percent",
            "zero_shot_std_column": "zero_shot_top1_std_percent",
            "label": "Top-1",
            "color": "#4F7C65",
            "marker": "o",
        },
        {
            "name": "top5",
            "column": "top5_accuracy_percent",
            "std_column": "top5_std_percent",
            "zero_shot_column": "zero_shot_top5_accuracy_percent",
            "zero_shot_std_column": "zero_shot_top5_std_percent",
            "label": "Top-5",
            "color": "#A75B73",
            "marker": "s",
        },
    ],
    "figure": {
        "figsize": [9, 5],
        "bar_figsize": [11, 5],
        "heatmap_figsize": [8, 5.5],
        "dpi": 300,
        "transparent": True,
        "value_labels": True,
        "value_format": "{:.1f}",
        "value_fontsize": 8.5,
        "errorbars": True,
        "highlight_best": True,
        "heatmap_best_marker_size": 90,
    },
    "plots": {
        "per_learning_rate": True,
        "best_steps": True,
        "best_accuracy": True,
        "heatmaps": True,
    },
    "axis": {"accuracy_ylim": [0, 100], "grid": True},
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
    if not config["series"]:
        raise ValueError("series must not be empty")
    for series in config["series"]:
        required = {
            "name", "column", "std_column", "zero_shot_column",
            "zero_shot_std_column", "label", "color", "marker",
        }
        if not isinstance(series, dict) or not required <= set(series):
            raise ValueError("Every series entry is missing required fields")
        mcolors.to_rgba(series["color"])
    return config


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def infer_paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path]:
    evaluation_value = config["data"]["evaluation_config"]
    if evaluation_value is None:
        raise ValueError("data.evaluation_config is required without CLI path overrides")
    evaluation_path = resolve_config_path(evaluation_value, config_path)
    with evaluation_path.open(encoding="utf-8") as stream:
        evaluation = yaml.safe_load(stream) or {}
    run_root = Path(evaluation["output_root"])
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    idea = evaluation.get("idea")
    if idea:
        run_root = run_root / f"idea_{'_'.join(str(idea).split())}"
    summary = run_root / "summary"
    return summary / "selection_fraction_chart_data.csv", summary / "lr_steps_plots"


def read_rows(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = config["data"]
    for key in ("model", "dataset"):
        if data[key] is not None:
            rows = [row for row in rows if row.get(key) == data[key]]
    for key in ("selection_count", "selection_fraction", "weight_decay"):
        if data[key] is not None:
            target = float(data[key])
            rows = [
                row for row in rows
                if math.isclose(float(row.get(key, "nan")), target,
                                rel_tol=1e-12, abs_tol=1e-12)
            ]
    rows = [row for row in rows if row.get("only_origin", "false").lower() != "true"]
    if not rows:
        raise ValueError(f"No rows remain after filtering {path}")

    numeric = {
        "learning_rate", "tta_steps", "weight_decay", "average_elapsed_minutes",
    }
    for series in config["series"]:
        numeric.update({
            series["column"], series["std_column"],
            series["zero_shot_column"], series["zero_shot_std_column"],
        })
    for row_number, row in enumerate(rows, start=2):
        missing = numeric - set(row)
        if missing:
            raise ValueError(f"Missing columns at row {row_number}: {sorted(missing)}")
        for column in numeric:
            row[column] = float(row[column])
            if not math.isfinite(row[column]):
                raise ValueError(f"Non-finite {column} at row {row_number}")
        row["tta_steps"] = int(row["tta_steps"])
    weight_decays = sorted({row["weight_decay"] for row in rows})
    if len(weight_decays) != 1:
        values = ", ".join(f"{value:g}" for value in weight_decays)
        raise ValueError(
            "Multiple weight_decay values remain after filtering: "
            f"{values}. Set data.weight_decay in the plot config."
        )
    keys = [(row["learning_rate"], row["tta_steps"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate learning_rate/tta_steps rows remain after filtering")
    return sorted(rows, key=lambda row: (row["learning_rate"], row["tta_steps"]))


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.8,
    })


def style_axis(axis, config: dict[str, Any], accuracy: bool = True) -> None:
    if accuracy:
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(*config["axis"]["accuracy_ylim"])
    if config["axis"]["grid"]:
        axis.grid(axis="y", color="#DADDE2", linestyle="--", linewidth=0.75, zorder=0)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("#3A3D42")


def save_figure(figure, path: Path, config: dict[str, Any]) -> list[Path]:
    figure.tight_layout()
    args = {
        "dpi": config["figure"]["dpi"], "transparent": config["figure"]["transparent"],
        "bbox_inches": "tight", "pad_inches": 0.05,
    }
    figure.savefig(path, **args)
    paths = [path]
    if config["output"]["pdf"]:
        pdf_path = path.with_suffix(".pdf")
        figure.savefig(pdf_path, **args)
        paths.append(pdf_path)
    plt.close(figure)
    return paths


def annotate(axis, x_values, y_values, color: str, config: dict[str, Any],
             direction: int = 1) -> None:
    if not config["figure"]["value_labels"]:
        return
    for x_value, y_value in zip(x_values, y_values):
        axis.annotate(
            config["figure"]["value_format"].format(y_value),
            (x_value, y_value), xytext=(0, 7 * direction), textcoords="offset points",
            ha="center", va="bottom" if direction > 0 else "top",
            fontsize=config["figure"]["value_fontsize"], color=color, zorder=6,
        )


def highlight_max(axis, x_values, y_values, color: str, config: dict[str, Any]) -> None:
    if not config["figure"]["highlight_best"]:
        return
    values = np.asarray(y_values, dtype=float)
    maximum = float(values.max())
    mask = np.isclose(values, maximum, rtol=1e-12, atol=1e-12)
    axis.scatter(
        np.asarray(x_values)[mask], values[mask], marker="*", s=210,
        facecolor=color, edgecolor="#1F2328", linewidth=1.1, zorder=7,
    )


def zero_shot_value(rows: list[dict[str, Any]], column: str) -> float:
    values = np.asarray([row[column] for row in rows], dtype=float)
    if not np.allclose(values, values[0], rtol=1e-8, atol=1e-8):
        raise ValueError(f"Zero-shot column varies across ablation rows: {column}")
    return float(values[0])


def plot_metric_categories(rows: list[dict[str, Any]], labels: list[str],
                           title: str, stem: Path, config: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    positions = np.arange(len(labels), dtype=float)
    figure, axis = plt.subplots(figsize=config["figure"]["figsize"])
    for index, series in enumerate(config["series"]):
        values = [zero_shot_value(rows, series["zero_shot_column"])] + [
            row[series["column"]] for row in rows
        ]
        errors = [zero_shot_value(rows, series["zero_shot_std_column"])] + [
            row[series["std_column"]] for row in rows
        ]
        axis.errorbar(
            positions, values, yerr=errors if config["figure"]["errorbars"] else None,
            label=series["label"], color=series["color"], marker=series["marker"],
            linewidth=2, markersize=6, capsize=3, zorder=3,
        )
        annotate(axis, positions, values, series["color"], config, 1 if index == 0 else -1)
        highlight_max(axis, positions, values, series["color"], config)
    axis.set_xticks(positions, labels)
    axis.set_xlabel("TTA update steps")
    axis.set_title(title)
    axis.legend(loc="best", frameon=False)
    style_axis(axis, config)
    paths.extend(save_figure(figure, stem.with_name(stem.name + "_line.png"), config))

    figure, axis = plt.subplots(figsize=config["figure"]["bar_figsize"])
    width = 0.72 / len(config["series"])
    for index, series in enumerate(config["series"]):
        values = [zero_shot_value(rows, series["zero_shot_column"])] + [
            row[series["column"]] for row in rows
        ]
        errors = [zero_shot_value(rows, series["zero_shot_std_column"])] + [
            row[series["std_column"]] for row in rows
        ]
        x_values = positions + (index - (len(config["series"]) - 1) / 2) * width
        bars = axis.bar(
            x_values, values, width=width * 0.88, yerr=errors if config["figure"]["errorbars"] else None,
            capsize=3, label=series["label"], color=mcolors.to_rgba(series["color"], 0.5),
            edgecolor=series["color"], linewidth=1.8, zorder=3,
        )
        if config["figure"]["value_labels"]:
            axis.bar_label(
                bars, fmt=config["figure"]["value_format"], padding=3,
                fontsize=config["figure"]["value_fontsize"],
            )
        highlight_max(axis, x_values, values, series["color"], config)
    axis.set_xticks(positions, labels)
    axis.set_xlabel("TTA update steps")
    axis.set_title(title)
    axis.legend(loc="best", frameon=False)
    style_axis(axis, config)
    paths.extend(save_figure(figure, stem.with_name(stem.name + "_bar.png"), config))
    return paths


def select_best(rows: list[dict[str, Any]], series: dict[str, Any]) -> dict[str, Any]:
    maximum = max(row[series["column"]] for row in rows)
    candidates = [
        row for row in rows
        if math.isclose(row[series["column"]], maximum, rel_tol=1e-12, abs_tol=1e-12)
    ]
    return min(candidates, key=lambda row: row["tta_steps"])


def plot_best_steps(by_lr: dict[float, list[dict[str, Any]]], output_dir: Path,
                    config: dict[str, Any]) -> list[Path]:
    rates = sorted(by_lr)
    positions = np.arange(len(rates), dtype=float)
    labels = [f"{rate:g}" for rate in rates]
    paths: list[Path] = []
    best = {
        series["name"]: [select_best(by_lr[rate], series) for rate in rates]
        for series in config["series"]
    }
    for kind in ("line", "bar"):
        figsize = config["figure"]["figsize" if kind == "line" else "bar_figsize"]
        figure, axis = plt.subplots(figsize=figsize)
        width = 0.72 / len(config["series"])
        for index, series in enumerate(config["series"]):
            values = [row["tta_steps"] for row in best[series["name"]]]
            if kind == "line":
                axis.plot(
                    positions, values, label=series["label"], color=series["color"],
                    marker=series["marker"], linewidth=2, markersize=6, zorder=3,
                )
                if config["figure"]["value_labels"]:
                    direction = 1 if index == 0 else -1
                    for position, value in zip(positions, values):
                        axis.annotate(
                            str(value), (position, value), xytext=(0, 7 * direction),
                            textcoords="offset points", ha="center",
                            va="bottom" if direction > 0 else "top",
                            fontsize=config["figure"]["value_fontsize"],
                            color=series["color"], zorder=6,
                        )
            else:
                x_values = positions + (index - (len(config["series"]) - 1) / 2) * width
                bars = axis.bar(
                    x_values, values, width=width * 0.88, label=series["label"],
                    color=mcolors.to_rgba(series["color"], 0.5),
                    edgecolor=series["color"], linewidth=1.8, zorder=3,
                )
                if config["figure"]["value_labels"]:
                    axis.bar_label(bars, fmt="%g", padding=3,
                                   fontsize=config["figure"]["value_fontsize"])
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Learning rate")
        axis.set_ylabel("Best TTA update steps")
        axis.set_title("Best TTA Step Count at Each Learning Rate")
        axis.yaxis.get_major_locator().set_params(integer=True)
        all_steps = [
            row["tta_steps"] for series in config["series"]
            for row in best[series["name"]]
        ]
        step_span = max(all_steps) - min(all_steps)
        padding = max(0.35, 0.12 * max(step_span, 1))
        axis.set_ylim(min(all_steps) - padding, max(all_steps) + padding)
        axis.legend(loc="best", frameon=False)
        style_axis(axis, config, accuracy=False)
        paths.extend(save_figure(figure, output_dir / f"best_steps_by_lr_{kind}.png", config))
    return paths


def plot_best_accuracy(by_lr: dict[float, list[dict[str, Any]]], output_dir: Path,
                       config: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    rates = sorted(by_lr)
    best = {
        series["name"]: [select_best(by_lr[rate], series) for rate in rates]
        for series in config["series"]
    }
    labels = ["CLIP\nzero-shot"] + [f"{rate:g}" for rate in rates]
    positions = np.arange(len(labels), dtype=float)
    paths: list[Path] = []
    for kind in ("line", "bar"):
        figsize = config["figure"]["figsize" if kind == "line" else "bar_figsize"]
        figure, axis = plt.subplots(figsize=figsize)
        width = 0.72 / len(config["series"])
        for index, series in enumerate(config["series"]):
            selected = best[series["name"]]
            values = [zero_shot_value(sum(by_lr.values(), []), series["zero_shot_column"])] + [
                row[series["column"]] for row in selected
            ]
            errors = [zero_shot_value(sum(by_lr.values(), []), series["zero_shot_std_column"])] + [
                row[series["std_column"]] for row in selected
            ]
            if kind == "line":
                axis.errorbar(
                    positions, values, yerr=errors if config["figure"]["errorbars"] else None,
                    label=series["label"], color=series["color"], marker=series["marker"],
                    linewidth=2, markersize=6, capsize=3, zorder=3,
                )
                annotate(axis, positions, values, series["color"], config,
                         1 if index == 0 else -1)
                highlight_max(axis, positions, values, series["color"], config)
            else:
                x_values = positions + (index - (len(config["series"]) - 1) / 2) * width
                bars = axis.bar(
                    x_values, values, width=width * 0.88,
                    yerr=errors if config["figure"]["errorbars"] else None, capsize=3,
                    label=series["label"], color=mcolors.to_rgba(series["color"], 0.5),
                    edgecolor=series["color"], linewidth=1.8, zorder=3,
                )
                if config["figure"]["value_labels"]:
                    axis.bar_label(bars, fmt=config["figure"]["value_format"], padding=3,
                                   fontsize=config["figure"]["value_fontsize"])
                highlight_max(axis, x_values, values, series["color"], config)
        axis.set_xticks(positions, labels)
        axis.set_xlabel("Learning rate (best step selected independently)")
        axis.set_title("Best TPT Accuracy at Each Learning Rate")
        axis.legend(loc="best", frameon=False)
        style_axis(axis, config)
        paths.extend(save_figure(figure, output_dir / f"best_accuracy_by_lr_{kind}.png", config))

    table = []
    for rate_index, rate in enumerate(rates):
        row = {"learning_rate": rate}
        for series in config["series"]:
            selected = best[series["name"]][rate_index]
            row[f"{series['name']}_best_steps"] = selected["tta_steps"]
            row[f"{series['name']}_best_accuracy_percent"] = selected[series["column"]]
            row[f"{series['name']}_std_percent"] = selected[series["std_column"]]
        table.append(row)
    return paths, table


def plot_heatmaps(rows: list[dict[str, Any]], output_dir: Path,
                  config: dict[str, Any]) -> list[Path]:
    rates = sorted({row["learning_rate"] for row in rows})
    steps = sorted({row["tta_steps"] for row in rows})
    lookup = {(row["learning_rate"], row["tta_steps"]): row for row in rows}
    expected = {(rate, step) for rate in rates for step in steps}
    if set(lookup) != expected:
        raise ValueError("Heatmaps require a complete learning-rate/step grid")
    paths: list[Path] = []
    for series in config["series"]:
        matrix = np.array([
            [lookup[(rate, step)][series["column"]] for step in steps]
            for rate in rates
        ])
        figure, axis = plt.subplots(figsize=config["figure"]["heatmap_figsize"])
        color_map = mcolors.LinearSegmentedColormap.from_list(
            f"{series['name']}_heatmap", ["#F7F7F7", series["color"]]
        )
        image = axis.imshow(matrix, aspect="auto", cmap=color_map)
        for row_index in range(len(rates)):
            for column_index in range(len(steps)):
                normalized = image.norm(matrix[row_index, column_index])
                axis.text(
                    column_index, row_index,
                    config["figure"]["value_format"].format(matrix[row_index, column_index]),
                    ha="center", va="center", fontsize=config["figure"]["value_fontsize"],
                    color="white" if normalized > 0.55 else "#1F2328",
                )
        if config["figure"]["highlight_best"]:
            maximum = float(matrix.max())
            for row_index, column_index in np.argwhere(
                np.isclose(matrix, maximum, rtol=1e-12, atol=1e-12)
            ):
                axis.scatter(
                    column_index + 0.3, row_index - 0.3,
                    marker="*", s=config["figure"]["heatmap_best_marker_size"],
                    facecolor=series["color"], edgecolor="#1F2328",
                    linewidth=1.1, zorder=4,
                )
        axis.set_xticks(range(len(steps)), [str(step) for step in steps])
        axis.set_yticks(range(len(rates)), [f"{rate:g}" for rate in rates])
        axis.set_xlabel("TTA update steps")
        axis.set_ylabel("Learning rate")
        axis.set_title(f"{series['label']} Accuracy: Learning Rate x TTA Steps")
        figure.colorbar(image, ax=axis, label="Accuracy (%)")
        paths.extend(save_figure(
            figure, output_dir / f"{series['name']}_lr_steps_heatmap.png", config
        ))
    return paths


def persist_best_table(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "best_steps_and_accuracy.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_plots(config_path: Path, data_path: Path | None = None,
                   output_dir: Path | None = None) -> list[Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if not config["enabled"]:
        return []
    inferred_data, inferred_output = infer_paths(config, config_path)
    data_path = Path(data_path).resolve() if data_path is not None else inferred_data
    if output_dir is None:
        configured = config["output"]["directory"]
        output_dir = (resolve_config_path(configured, config_path)
                      if configured is not None else inferred_output)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(data_path, config)
    configure_style()

    bundled_data = output_dir / "lr_steps_plot_data.csv"
    if data_path != bundled_data:
        shutil.copyfile(data_path, bundled_data)
    shutil.copyfile(config_path, output_dir / "lr_steps_plots_config.json")
    if Path(__file__).resolve() != (output_dir / Path(__file__).name).resolve():
        shutil.copyfile(__file__, output_dir / Path(__file__).name)

    by_lr = {
        rate: [row for row in rows if row["learning_rate"] == rate]
        for rate in sorted({row["learning_rate"] for row in rows})
    }
    paths: list[Path] = []
    if config["plots"]["per_learning_rate"]:
        per_lr_dir = output_dir / "per_learning_rate"
        per_lr_dir.mkdir(exist_ok=True)
        for rate, rate_rows in by_lr.items():
            rate_rows.sort(key=lambda row: row["tta_steps"])
            labels = ["CLIP\nzero-shot"] + [str(row["tta_steps"]) for row in rate_rows]
            slug = f"{rate:g}".replace("-", "m").replace(".", "p")
            paths.extend(plot_metric_categories(
                rate_rows, labels, f"TPT Accuracy at Learning Rate {rate:g}",
                per_lr_dir / f"lr_{slug}_steps", config,
            ))
    if config["plots"]["best_steps"]:
        paths.extend(plot_best_steps(by_lr, output_dir, config))
    if config["plots"]["best_accuracy"]:
        best_paths, table = plot_best_accuracy(by_lr, output_dir, config)
        paths.extend(best_paths)
        paths.append(persist_best_table(table, output_dir))
    if config["plots"]["heatmaps"]:
        paths.extend(plot_heatmaps(rows, output_dir, config))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_plots(args.config, args.data, args.output_dir)
    print(f"[INFO] generated {len(paths)} LR/step ablation artifacts")


if __name__ == "__main__":
    main()
